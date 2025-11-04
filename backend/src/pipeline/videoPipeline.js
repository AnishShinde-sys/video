import path from 'path';
import fs from 'fs-extra';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
import ffprobeInstaller from '@ffprobe-installer/ffprobe';
import axios from 'axios';
import crypto from 'crypto';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { GoogleAIFileManager } from '@google/generative-ai/files';

import config from '../config.js';
import logger from '../logger.js';
import jobStore from '../jobStore.js';
import { prepareJobDirectories, writeJson, relativeToJob } from '../utils/fileUtils.js';

const GEMINI_TEXT_MODEL = 'gemini-2.0-flash';
const GEMINI_IMAGE_MODEL = 'gemini-2.5-flash-image';
const MAX_GEMINI_REFERENCE_FRAMES = 5;

ffmpeg.setFfmpegPath(ffmpegInstaller.path);
ffmpeg.setFfprobePath(ffprobeInstaller.path);

function parseFps(stream) {
  const rate = stream.avg_frame_rate || stream.r_frame_rate;
  if (!rate || rate === '0/0') return 30;
  const [num, den] = rate.split('/').map((value) => Number(value));
  if (!num || !den) return 30;
  return num / den;
}

function sectionLabel(index, total) {
  const ratio = index / Math.max(total - 1, 1);
  if (ratio < 0.2) return 'start';
  if (ratio > 0.8) return 'end';
  return 'middle';
}

function normalizeFrameList(frames) {
  if (!frames) return [];
  if (Array.isArray(frames)) return frames;
  if (Array.isArray(frames.frames)) return frames.frames;
  return [];
}

function selectRepresentativeFrames(frames, maxFrames = MAX_GEMINI_REFERENCE_FRAMES) {
  const list = normalizeFrameList(frames);
  if (list.length <= maxFrames) {
    return list;
  }

  const lastIndex = list.length - 1;
  const step = lastIndex / Math.max(maxFrames - 1, 1);
  const selected = [];
  const seen = new Set();

  for (let i = 0; i < maxFrames; i += 1) {
    const index = Math.round(step * i);
    const frame = list[Math.min(index, lastIndex)];
    if (frame && !seen.has(frame.filePath)) {
      selected.push(frame);
      seen.add(frame.filePath);
    }
  }

  return selected;
}

function shouldRetryRequest(error) {
  const status = error?.response?.status;
  if (status === 429) return true;
  if (status && status >= 500) return true;
  if (error?.code === 'ECONNRESET' || error?.code === 'ETIMEDOUT') return true;
  if (error?.message && /timeout/i.test(error.message)) return true;
  return !status;
}

async function withRetry(requestFn, {
  retries = 3,
  initialDelay = 1000,
  backoffFactor = 2,
  description = 'external request'
} = {}) {
  let attempt = 0;
  let delay = initialDelay;

  while (true) {
    try {
      return await requestFn();
    } catch (error) {
      attempt += 1;
      if (attempt > retries || !shouldRetryRequest(error)) {
        throw error;
      }

      logger.warn({ err: error, attempt, delay, description }, `${description} failed; retrying`);
      await sleep(delay);
      delay *= backoffFactor;
    }
  }
}

async function runFfmpeg(command, { description = 'ffmpeg job' } = {}) {
  return new Promise((resolve, reject) => {
    let stderr = '';
    let stdout = '';

    command
      .on('start', (cli) => {
        logger.debug({ cli }, `${description} started`);
      })
      .on('progress', (progress) => {
        logger.debug({ progress }, `${description} progress`);
      })
      .on('stderr', (line) => {
        stderr += `${line}\n`;
      })
      .on('stdout', (line) => {
        stdout += `${line}\n`;
      })
      .on('end', () => {
        logger.debug(`${description} completed`);
        resolve({ stdout, stderr });
      })
      .on('error', (error, _stdout, ffmpegStderr) => {
        const combinedStderr = (ffmpegStderr || '') + stderr;
        const wrappedError = error instanceof Error ? error : new Error(error?.message || 'ffmpeg failed');
        wrappedError.stderr = combinedStderr;
        wrappedError.stdout = stdout;
        wrappedError.description = description;
        logger.error({ err: wrappedError }, `${description} failed`);
        reject(wrappedError);
      })
      .run();
  });
}

function ensureGeminiConfigured() {
  if (!config.geminiApiKey) {
    throw new Error('GEMINI_API_KEY is not configured');
  }
}

function ensureSeedanceConfigured() {
  if (!config.seedanceApiKey) {
    throw new Error('EACHLABS_API_KEY is not configured');
  }
  const { keyId, keyName, applicationKey } = config.backblaze;
  if (!keyId || !keyName || !applicationKey) {
    throw new Error('Backblaze B2 credentials (keyID, keyName, applicationKey) are required for video generation');
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default class VideoPipeline {
  constructor(jobId) {
    this.jobId = jobId;
    this.geminiClient = null;
    this.visionModel = null;
    this.fileManager = null;
    this.jobPaths = null;
  }

  async initializeDirectories() {
    this.jobPaths = await prepareJobDirectories(this.jobId, config.storage.jobs);
  }

  async run({ videoPath, prompt }) {
    await this.initializeDirectories();

    try {
      jobStore.progress(this.jobId, {
        status: 'processing',
        progress: 5,
        step: 'Extracting key frames from video'
      });

      const frames = await this.extractFrames(videoPath);

      jobStore.progress(this.jobId, {
        progress: 20,
        step: 'Generating transformed image with Gemini'
      });

      const generatedImage = await this.generateImage(frames, prompt);

      jobStore.progress(this.jobId, {
        progress: 35,
        step: 'Extracting audio from original video'
      });

      const extractedAudio = await this.extractAudio(videoPath);

      let analysisResult = null;
      if (!config.pipeline.skipAnalysis) {
        jobStore.progress(this.jobId, {
          progress: 55,
          step: 'Analyzing original video for timing and transcript'
        });

        analysisResult = await this.analyzeVideo(videoPath);
      }

      let videoResult = null;
      const shouldGenerateVideo = !config.pipeline.skipVideoGeneration && Boolean(analysisResult);

      if (shouldGenerateVideo && !config.seedanceApiKey) {
        logger.warn({ jobId: this.jobId }, 'Seedance API key missing; skipping video generation step');
      }

      if (shouldGenerateVideo && config.seedanceApiKey) {
        jobStore.progress(this.jobId, {
          progress: 70,
          step: 'Generating animated video from Seedance'
        });

        const seedanceResult = await this.generateSeedanceVideo({
          originalVideo: videoPath,
          generatedImage: generatedImage.generatedImagePath,
          analysisText: analysisResult.analysisText,
          userPrompt: prompt
        });

        videoResult = { seedance: seedanceResult, latentsync: null };

        if (extractedAudio?.audioPath) {
          jobStore.progress(this.jobId, {
            progress: 85,
            step: 'Applying LatentSync lip sync alignment'
          });

          try {
            const latentsyncResult = await this.runLatentSync({
              videoPath: seedanceResult.videoPath,
              audioPath: extractedAudio.audioPath,
              guidanceScale: 1
            });

            videoResult.latentsync = latentsyncResult;
          } catch (latentsyncError) {
            logger.warn({ err: latentsyncError, jobId: this.jobId }, 'LatentSync processing failed; falling back to Seedance output');
          }
        }
      }

      const resultPayload = this.buildResultPayload({
        prompt,
        frames,
        generatedImage,
        extractedAudio,
        analysisResult,
        videoResult
      });

      jobStore.complete(this.jobId, resultPayload);
    } catch (error) {
      logger.error({ err: error, jobId: this.jobId }, 'Pipeline failed');
      jobStore.fail(this.jobId, error);
      throw error;
    }
  }

  buildResultPayload({ prompt, frames, generatedImage, extractedAudio, analysisResult, videoResult }) {
    const base = {
      jobId: this.jobId,
      prompt,
      frames: frames?.frames?.map((frame) => ({
        path: this.assetPath(frame.filePath),
        timestamp: frame.timestamp
      })),
      generatedImage: generatedImage ? this.assetPath(generatedImage.generatedImagePath) : null,
      generatedImageMetadata: generatedImage?.metadataFile ? this.assetPath(generatedImage.metadataFile) : null,
      extractedAudio: extractedAudio ? this.assetPath(extractedAudio.audioPath) : null,
      createdAt: new Date().toISOString()
    };

    if (analysisResult) {
      base.analysisFile = this.assetPath(analysisResult.analysisFile);
      base.analysisJson = this.assetPath(analysisResult.analysisJson);
    }

    if (videoResult?.seedance) {
      const { seedance, latentsync } = videoResult;
      base.video = {
        finalVideo: this.assetPath(latentsync?.videoPath || seedance.videoPath),
        finalSource: latentsync ? 'latentsync' : 'seedance',
        seedance: {
          rawVideo: this.assetPath(seedance.videoPath),
          metadata: this.assetPath(seedance.metadataFile),
          predictionId: seedance.predictionId,
          duration: seedance.duration,
          sizeMb: seedance.sizeMb
        },
        latentsync: latentsync ? {
          video: this.assetPath(latentsync.videoPath),
          metadata: this.assetPath(latentsync.metadataFile),
          predictionId: latentsync.predictionId,
          guidanceScale: latentsync.guidanceScale,
          duration: latentsync.duration,
          sizeMb: latentsync.sizeMb
        } : null
      };
    }

    return base;
  }

  assetPath(targetPath) {
    if (!targetPath) return null;
    const relative = relativeToJob(this.jobPaths.jobDir, targetPath);
    return `/api/jobs/${this.jobId}/files/${relative}`;
  }

  async extractFrames(videoPath) {
    const metadata = await new Promise((resolve, reject) => {
      ffmpeg.ffprobe(videoPath, (err, data) => {
        if (err) {
          reject(err);
        } else {
          resolve(data);
        }
      });
    });

    const videoStream = metadata.streams.find((stream) => stream.codec_type === 'video');
    if (!videoStream) {
      throw new Error('Unable to read video stream metadata');
    }

    const fps = parseFps(videoStream);
    const duration = Number(metadata.format.duration || videoStream.duration || 0);
    const frameCount = Number(videoStream.nb_frames || 0);
    const numFrames = config.pipeline.frameCount;

    const timestamps = Array.from({ length: numFrames }, (_, index) => {
      if (!duration) return index;
      return Math.min((duration * index) / Math.max(numFrames - 1, 1), Math.max(duration - 0.001, 0));
    });

    const frames = [];

    for (let index = 0; index < timestamps.length; index += 1) {
      const timestamp = timestamps[index];
      const section = sectionLabel(index, timestamps.length);
      const fileName = `frame_${String(index + 1).padStart(2, '0')}_${section}_${Math.round(timestamp * fps).toString().padStart(4, '0')}.jpg`;
      const filePath = path.join(this.jobPaths.framesDir, fileName);

      const command = ffmpeg(videoPath)
        .seekInput(Math.max(timestamp - 0.01, 0))
        .frames(1)
        .outputOptions([
          '-vf', 'scale=iw:ih:flags=bicubic,format=yuvj422p',
          '-pix_fmt', 'yuvj422p',
          '-q:v', '2',
          '-strict', 'unofficial'
        ])
        .output(filePath);

      await runFfmpeg(command, {
        description: `extract frame ${index + 1}/${timestamps.length} at ${timestamp.toFixed(2)}s`
      });

      if (!(await fs.pathExists(filePath))) {
        logger.warn({ jobId: this.jobId, filePath }, 'Expected frame not created; reusing previous frame if available');

        const previous = frames[frames.length - 1];
        if (!previous) {
          throw new Error(`FFmpeg did not produce frame ${index + 1}; no previous frame to fall back to`);
        }

        await fs.copy(previous.filePath, filePath);
      }

      frames.push({ filePath, timestamp, section });
    }

    const metadataFile = path.join(this.jobPaths.framesDir, 'metadata.json');
    const metadataPayload = {
      videoPath,
      fps,
      duration,
      frameCount,
      frames: frames.map((frame) => ({
        file: path.basename(frame.filePath),
        timestamp: frame.timestamp,
        section: frame.section
      }))
    };
    await writeJson(metadataFile, metadataPayload);

    return { frames, metadataFile };
  }

  async generateImage(frames, userPrompt) {
    ensureGeminiConfigured();
    const frameList = normalizeFrameList(frames);
    const selectedFrames = selectRepresentativeFrames(frameList);
    const referenceFrame = selectedFrames[0];

    if (!referenceFrame) {
      throw new Error('No reference frame available for image generation');
    }

    const promptDescription = `Create a photorealistic image that reflects the requested edit: ${userPrompt}`;
    const frameSummaries = selectedFrames
      .map((frame, index) => {
        const timestamp = Number.isFinite(frame.timestamp) ? `${frame.timestamp.toFixed(2)}s` : 'unknown time';
        const section = frame.section || 'unknown section';
        return `Frame ${index + 1}: captured at ${timestamp}, section ${section}.`;
      })
      .join('\n');

    const base64Frames = await Promise.all(
      selectedFrames.map(async (frame) => {
        const buffer = await fs.readFile(frame.filePath);
        return buffer.toString('base64');
      })
    );

    const prompt = `You are creating a photorealistic still frame for a video edit.

Edit Goal: ${userPrompt}

Reference frames provided (${selectedFrames.length} total) capture the subject across the clip. Maintain continuity with their appearance, lighting, and composition.

Reference frames:
${frameSummaries}`;

    const parts = [
      { text: prompt },
      ...base64Frames.map((data) => ({
        inlineData: {
          mimeType: 'image/jpeg',
          data
        }
      }))
    ];

    const response = await withRetry(
      () => axios.post(
        `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_IMAGE_MODEL}:generateContent?key=${config.geminiApiKey}`,
        {
          contents: [
            {
              parts
            }
          ],
          generationConfig: {
            response_modalities: ['IMAGE'],
            image_config: {
              aspect_ratio: '1:1'
            }
          }
        },
        { timeout: 45000 }
      ),
      {
        description: 'Gemini image generation request',
        retries: 4,
        initialDelay: 2000
      }
    );

    const candidate = response.data?.candidates?.[0];
    if (!candidate) {
      throw new Error('Gemini image generation produced no candidates');
    }

    const imagePart = candidate.content?.parts?.find((part) => part.inlineData);
    if (!imagePart) {
      throw new Error('Gemini image generation returned no image data');
    }

    const generatedImagePath = path.join(this.jobPaths.generatedDir, 'generated_output.jpg');
    const metadataFile = path.join(this.jobPaths.generatedDir, 'metadata.json');
    const imageBuffer = Buffer.from(imagePart.inlineData.data, 'base64');

    await fs.writeFile(generatedImagePath, imageBuffer);
    await writeJson(metadataFile, {
      user_prompt: userPrompt,
      reference_frame: path.basename(referenceFrame.filePath),
      reference_frame_count: selectedFrames.length,
      reference_frames: selectedFrames.map((frame, index) => ({
        index: index + 1,
        file: path.basename(frame.filePath),
        timestamp: Number.isFinite(frame.timestamp) ? frame.timestamp : null,
        section: frame.section || null
      })),
      description: promptDescription,
      created_at: new Date().toISOString()
    });

    return { generatedImagePath, metadataFile };
  }

  async extractAudio(videoPath) {
    const audioPath = path.join(this.jobPaths.audioDir, 'original_audio.wav');
    const command = ffmpeg(videoPath)
      .noVideo()
      .audioChannels(2)
      .audioCodec('pcm_s16le')
      .audioFrequency(44100)
      .format('wav')
      .output(audioPath);

    await runFfmpeg(command, { description: 'extract audio track' });

    return {
      audioPath,
      sizeMb: (await fs.stat(audioPath)).size / (1024 * 1024)
    };
  }

  async ensureGeminiClient() {
    if (!this.geminiClient) {
      ensureGeminiConfigured();
      this.geminiClient = new GoogleGenerativeAI(config.geminiApiKey);
      this.visionModel = this.geminiClient.getGenerativeModel({ model: GEMINI_TEXT_MODEL });
    }

    if (!this.fileManager) {
      ensureGeminiConfigured();
      this.fileManager = new GoogleAIFileManager(config.geminiApiKey);
    }
  }

  async waitForFileReady(file) {
    let current = file;
    while (current?.state === 'PROCESSING' || current?.state === 'UPLOADING') {
      await sleep(2000);
      current = await this.fileManager.getFile(current.name);
    }

    if (current?.state !== 'ACTIVE') {
      throw new Error(`Gemini file processing failed: ${current?.state}`);
    }

    return current;
  }

  async analyzeVideo(videoPath) {
    await this.ensureGeminiClient();

    const upload = await this.fileManager.uploadFile(videoPath, {
      mimeType: 'video/mp4',
      displayName: path.basename(videoPath)
    });

    const readyFile = await this.waitForFileReady(upload.file);

    const analysisPrompt = `Analyze this video in EXTREME detail. Provide second-by-second tracking of EVERYTHING that happens.

## PART 1: COMPLETE SPEECH TRANSCRIPTION - **CRITICAL REQUIREMENT**
**IMPORTANT: This is the MOST CRITICAL part of the analysis. The video generation will use this transcript.**

Transcribe EVERY SINGLE WORD spoken with EXACT timestamps in [MM:SS.milliseconds] format.
**YOU MUST TRANSCRIBE EXACTLY WHAT IS SAID - DO NOT SKIP OR SUMMARIZE ANY WORDS.**

For each word/phrase, include:
- [Timestamp] "Exact words spoken" - Tone, Volume, Pace
- Tone (happy, sad, excited, angry, sarcastic, neutral, etc.)
- Volume (loud, normal, soft, whisper)
- Pace (fast, normal, slow)
- Emphasis or stress on specific words
- Pauses and their duration

After the detailed transcription, provide:
**FULL TRANSCRIPT (for video generation):**
Provide the complete spoken text as one continuous paragraph with all the exact words that must be said in the generated video.

## PART 2: SECOND-BY-SECOND MOUTH MOVEMENTS
For EVERY second of the video, describe mouth, lips, tongue, teeth, jaw, and whether speaking.

## PART 3: SECOND-BY-SECOND EYE TRACKING
For EVERY second track direction, blinks, movement, pupil size, eyelid position, and eye contact.

## PART 4: SECOND-BY-SECOND HEAD MOVEMENTS
Document head position, movement, and angles every second.

## PART 5: SECOND-BY-SECOND BODY MOVEMENTS
Document posture, arms, hands, torso, and distance every second.

## PART 6: FACIAL EXPRESSIONS (DETAILED)
Track all facial expression changes with intensity and micro-expressions.

## PART 7: SYNCHRONIZED TIMELINE
Combine speech, mouth, eyes, head, body, expression for every second.

## PART 8: SCENE & ENVIRONMENT
Describe duration, frames, resolution, lighting, background, foreground, camera angle, movement, depth of field, composition.

## PART 9: AUDIO ANALYSIS
Document quality, background noise, acoustics, levels, non-speech sounds, silence with timestamps.

## PART 10: METADATA & SUMMARY
Summary metrics including duration, word count, actions, emotions, key moments, narrative.`;

    const response = await this.visionModel.generateContent([
      { text: analysisPrompt },
      {
        fileData: {
          mimeType: readyFile.mimeType,
          fileUri: readyFile.uri
        }
      }
    ]);

    const analysisText = response?.response?.text();
    if (!analysisText) {
      throw new Error('Gemini video analysis returned no content');
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const analysisFile = path.join(this.jobPaths.analysisDir, `analysis_${timestamp}.txt`);
    const analysisJson = path.join(this.jobPaths.analysisDir, `analysis_${timestamp}.json`);

    await fs.writeFile(analysisFile, analysisText, 'utf8');
    await writeJson(analysisJson, {
      analysis: analysisText,
      video_path: videoPath,
      created_at: new Date().toISOString()
    });

    await this.fileManager.deleteFile(readyFile.name).catch((error) => {
      logger.warn({ err: error }, 'Failed to delete uploaded video from Gemini');
    });

    return { analysisFile, analysisJson, analysisText };
  }

  extractTranscript(analysis) {
    const transcriptMarker = analysis.includes('**FULL TRANSCRIPT')
      ? '**FULL TRANSCRIPT'
      : 'FULL TRANSCRIPT:';

    if (!analysis.includes(transcriptMarker)) {
      return 'Speak naturally and expressively';
    }

    const section = analysis.split(transcriptMarker)[1];
    if (!section) return 'Speak naturally and expressively';

    const text = section.split('##')[0].split('\n\n')[0];
    return text.replace(/\*\*|:/g, '').trim() || 'Speak naturally and expressively';
  }

  extractCameraInfo(analysis) {
    const line = analysis.split('\n').find((entry) => entry.toLowerCase().includes('camera angle'));
    if (!line) return 'straight on, medium close-up';
    const [, value] = line.split(':');
    return value ? value.trim() : 'straight on, medium close-up';
  }

  extractKeyMovements(analysis) {
    const movements = [];
    for (const line of analysis.split('\n')) {
      if (line.trim().startsWith('[') && (line.includes('Speech:') || line.includes('Eyes:') || line.includes('Head:'))) {
        movements.push(line.trim());
        if (movements.length >= 10) break;
      }
    }
    return movements.join('\n') || 'Maintain natural, subtle movements';
  }

  extractDuration(analysis) {
    const match = analysis.match(/Total video duration:\s*Approximately\s*(\d+(?:\.\d+)*)\s*seconds/i);
    if (!match) return 12;
    const duration = Math.floor(Number(match[1]));
    return Math.min(Math.max(duration, 5), 12);
  }

  async authorizeB2() {
    const { keyId, applicationKey } = config.backblaze;
    const authString = Buffer.from(`${keyId}:${applicationKey}`).toString('base64');

    const response = await withRetry(
      () => axios.get('https://api.backblazeb2.com/b2api/v2/b2_authorize_account', {
        headers: {
          Authorization: `Basic ${authString}`
        },
        timeout: 15000
      }),
      {
        description: 'Backblaze authorize request',
        retries: 3,
        initialDelay: 2000
      }
    );

    return response.data;
  }

  async getUploadTarget(authData) {
    const response = await withRetry(
      () => axios.post(
        `${authData.apiUrl}/b2api/v2/b2_list_buckets`,
        {
          accountId: authData.accountId
        },
        {
          headers: {
            Authorization: authData.authorizationToken
          },
          timeout: 15000
        }
      ),
      {
        description: 'Backblaze list buckets request',
        retries: 3,
        initialDelay: 2000
      }
    );

    const bucket = response.data.buckets.find((item) => item.bucketName === config.backblaze.keyName)
      || response.data.buckets[0];
    if (!bucket) {
      throw new Error('No Backblaze buckets available');
    }

    const upload = await withRetry(
      () => axios.post(
        `${authData.apiUrl}/b2api/v2/b2_get_upload_url`,
        { bucketId: bucket.bucketId },
        {
          headers: {
            Authorization: authData.authorizationToken
          },
          timeout: 15000
        }
      ),
      {
        description: 'Backblaze get upload url request',
        retries: 3,
        initialDelay: 2000
      }
    );

    return {
      bucketName: bucket.bucketName,
      uploadUrl: upload.data.uploadUrl,
      uploadAuthToken: upload.data.authorizationToken,
      downloadUrl: authData.downloadUrl
    };
  }

  async uploadToB2(filePath, { contentType = 'application/octet-stream', filePrefix = 'video_transformer' } = {}) {
    const authData = await this.authorizeB2();
    const target = await this.getUploadTarget(authData);

    const fileContent = await fs.readFile(filePath);
    const sha1 = crypto.createHash('sha1').update(fileContent).digest('hex');
    const safePrefix = filePrefix.replace(/[^a-zA-Z0-9/_-]/g, '') || 'video_transformer';
    const fileName = `${safePrefix}/${Date.now()}_${path.basename(filePath)}`;

    await withRetry(
      () => axios.post(target.uploadUrl, fileContent, {
        headers: {
          Authorization: target.uploadAuthToken,
          'X-Bz-File-Name': fileName,
          'Content-Type': contentType,
          'Content-Length': fileContent.length,
          'X-Bz-Content-Sha1': sha1
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: 60000
      }),
      {
        description: 'Backblaze upload image request',
        retries: 3,
        initialDelay: 2000
      }
    );

    return `${target.downloadUrl}/file/${target.bucketName}/${fileName}`;
  }

  async generateSeedanceVideo({ originalVideo, generatedImage, analysisText, userPrompt }) {
    ensureSeedanceConfigured();

    const imageUrl = await this.uploadToB2(generatedImage, {
      contentType: 'image/jpeg',
      filePrefix: 'video_transformer/generated'
    });
    const transcript = this.extractTranscript(analysisText);
    const keyMovements = this.extractKeyMovements(analysisText);
    const cameraInfo = this.extractCameraInfo(analysisText);
    const duration = this.extractDuration(analysisText);

    const generationPrompt = `${userPrompt}

SPEECH TRANSCRIPT (THE PERSON MUST SAY THESE EXACT WORDS):
${transcript}

MOTION & TIMING (from original video):
${keyMovements}

VIDEO REQUIREMENTS:
- Duration: ${duration} seconds
- Style: Cinematic, photorealistic, high-fidelity
- Motion: Smooth, natural movements matching original timing
- Camera: ${cameraInfo} - Static camera
- Lighting: Realistic lighting matching the reference image
- Audio: The person must speak the transcript above with natural lip sync and voice

Animate this image with natural, lifelike motion. THE PERSON MUST SAY THE EXACT WORDS FROM THE TRANSCRIPT. Maintain the exact appearance from the image while adding subtle movements, facial expressions, natural lip movements that match the speech, and the actions described above. Ensure photorealistic quality with smooth, natural motion and accurate lip synchronization.`;

    const requestPayload = {
      model: 'seedance-v1-pro-fast-image-to-video',
      version: '0.0.1',
      input: {
        prompt: generationPrompt,
        image_url: imageUrl,
        aspect_ratio: '9:16',
        resolution: '1080p',
        duration: String(duration)
      },
      webhook_url: ''
    };

    const headers = {
      'X-API-Key': config.seedanceApiKey,
      'Content-Type': 'application/json'
    };

    const createResponse = await withRetry(
      () => axios.post('https://api.eachlabs.ai/v1/prediction/', requestPayload, {
        headers,
        timeout: 20000
      }),
      {
        description: 'Seedance create prediction request',
        retries: 3,
        initialDelay: 3000
      }
    );
    const predictionId = createResponse.data.predictionID || createResponse.data.id;

    if (!predictionId) {
      throw new Error('Seedance API did not return a prediction ID');
    }

    let attempts = 0;
    const maxAttempts = 180;
    let videoUrl;

    while (attempts < maxAttempts) {
      attempts += 1;
      await sleep(1000);

      const pollResponse = await withRetry(
        () => axios.get(`https://api.eachlabs.ai/v1/prediction/${predictionId}`, {
          headers,
          timeout: 15000
        }),
        {
          description: 'Seedance poll request',
          retries: 2,
          initialDelay: 1500
        }
      );
      const status = pollResponse.data.status;

      if (status === 'succeeded' || status === 'success') {
        videoUrl = pollResponse.data.output;
        break;
      }

      if (status === 'failed') {
        throw new Error(pollResponse.data.error || 'Seedance video generation failed');
      }
    }

    if (!videoUrl) {
      throw new Error('Seedance video generation timed out');
    }

    const videoResponse = await withRetry(
      () => axios.get(videoUrl, {
        responseType: 'arraybuffer',
        timeout: 60000
      }),
      {
        description: 'Seedance download video request',
        retries: 3,
        initialDelay: 3000
      }
    );
    const timestamp = Date.now();
    const videoPath = path.join(this.jobPaths.videosDir, `generated_video_${timestamp}.mp4`);
    await fs.writeFile(videoPath, videoResponse.data);

    const metadataFile = path.join(this.jobPaths.videosDir, `metadata_${timestamp}.json`);
    const metadataPayload = {
      predictionId,
      video_url: videoUrl,
      original_video: originalVideo,
      generated_image: generatedImage,
      duration,
      created_at: new Date().toISOString()
    };
    await writeJson(metadataFile, metadataPayload);

    return {
      predictionId,
      videoPath,
      metadataFile,
      duration,
      sizeMb: videoResponse.data.byteLength / (1024 * 1024)
    };
  }

  async runLatentSync({ videoPath, audioPath, guidanceScale = 1 }) {
    ensureSeedanceConfigured();

    const videoUrl = await this.uploadToB2(videoPath, {
      contentType: 'video/mp4',
      filePrefix: 'video_transformer/seedance_output'
    });

    const audioUrl = await this.uploadToB2(audioPath, {
      contentType: 'audio/wav',
      filePrefix: 'video_transformer/original_audio'
    });

    const requestPayload = {
      model: 'latentsync',
      version: '0.0.1',
      input: {
        video_url: videoUrl,
        audio_url: audioUrl,
        guidance_scale: guidanceScale
      },
      webhook_url: ''
    };

    const headers = {
      'X-API-Key': config.seedanceApiKey,
      'Content-Type': 'application/json'
    };

    const createResponse = await withRetry(
      () => axios.post('https://api.eachlabs.ai/v1/prediction/', requestPayload, {
        headers,
        timeout: 20000
      }),
      {
        description: 'LatentSync create prediction request',
        retries: 3,
        initialDelay: 3000
      }
    );

    const predictionId = createResponse.data.predictionID || createResponse.data.id;

    if (!predictionId) {
      throw new Error('LatentSync API did not return a prediction ID');
    }

    let attempts = 0;
    const maxAttempts = 300;
    let resultUrl;
    let duration = createResponse.data.duration || null;

    while (attempts < maxAttempts) {
      attempts += 1;
      await sleep(1000);

      const pollResponse = await withRetry(
        () => axios.get(`https://api.eachlabs.ai/v1/prediction/${predictionId}`, {
          headers,
          timeout: 20000
        }),
        {
          description: 'LatentSync poll request',
          retries: 2,
          initialDelay: 2000
        }
      );

      duration = pollResponse.data.duration || duration;

      const status = pollResponse.data.status;
      if (status === 'succeeded' || status === 'success') {
        resultUrl = pollResponse.data.output;
        break;
      }
      if (status === 'failed') {
        throw new Error(pollResponse.data.error || 'LatentSync processing failed');
      }
    }

    if (!resultUrl) {
      throw new Error('LatentSync processing timed out');
    }

    const videoResponse = await withRetry(
      () => axios.get(resultUrl, {
        responseType: 'arraybuffer',
        timeout: 60000
      }),
      {
        description: 'LatentSync download video request',
        retries: 3,
        initialDelay: 3000
      }
    );

    const timestamp = Date.now();
    const finalVideoPath = path.join(this.jobPaths.finalDir, `latentsync_video_${timestamp}.mp4`);
    await fs.writeFile(finalVideoPath, videoResponse.data);

    const metadataFile = path.join(this.jobPaths.finalDir, `latentsync_metadata_${timestamp}.json`);
    await writeJson(metadataFile, {
      predictionId,
      video_url: resultUrl,
      source_video: videoPath,
      source_audio: audioPath,
      guidance_scale: guidanceScale,
      duration,
      created_at: new Date().toISOString()
    });

    return {
      predictionId,
      videoPath: finalVideoPath,
      metadataFile,
      duration,
      guidanceScale,
      sizeMb: videoResponse.data.byteLength / (1024 * 1024)
    };
  }
}
