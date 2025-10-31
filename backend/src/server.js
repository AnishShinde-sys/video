import fs from "fs";
import fsp from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

import axios from "axios";
import cors from "cors";
import express from "express";
import FormData from "form-data";
import multer from "multer";
import { nanoid } from "nanoid";
import dotenv from "dotenv";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const uploadDir = path.resolve(__dirname, "..", "uploads");
const storageDir = path.resolve(__dirname, "..", "storage");

const envCandidates = [
  path.resolve(__dirname, "../../.env"),
  path.resolve(__dirname, "../.env"),
  path.resolve(process.cwd(), ".env"),
];

for (const candidate of envCandidates) {
  if (fs.existsSync(candidate)) {
    dotenv.config({ path: candidate });
    break;
  }
}

for (const dir of [uploadDir, storageDir]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

const app = express();

app.use(cors());
app.use(express.json({ limit: "1mb" }));

const DECART_API_KEY = process.env.DECART_API_KEY;
if (!DECART_API_KEY) {
  console.warn("⚠️  DECART_API_KEY is not set. Requests to Decart API will fail until it is configured.");
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB as per Decart specs
const allowedMimeTypes = new Set([
  "video/mp4",
  "video/quicktime",
  "video/webm",
  "video/x-matroska",
  "video/avi",
]);

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, uploadDir),
    filename: (_req, file, cb) => {
      const ext = path.extname(file.originalname) || ".mp4";
      cb(null, `${Date.now()}_${nanoid()}${ext}`);
    },
  }),
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: (_req, file, cb) => {
    if (allowedMimeTypes.has(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new multer.MulterError("LIMIT_UNEXPECTED_FILE", "Unsupported file type"));
    }
  },
});

const jobs = new Map();

const publicJob = (job) => {
  if (!job) return null;
  const { outputPath, inputPath, ...rest } = job;
  return rest;
};

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.post("/api/video-edits", upload.single("video"), async (req, res) => {
  try {
    const prompt = req.body?.prompt?.trim();
    const resolution = req.body?.resolution?.trim() || "720p";

    if (!req.file) {
      return res.status(400).json({ error: "Video file is required" });
    }

    if (!prompt) {
      await safeUnlink(req.file.path);
      return res.status(400).json({ error: "Prompt cannot be empty" });
    }

    const jobId = nanoid();
    const createdAt = new Date().toISOString();

    jobs.set(jobId, {
      id: jobId,
      status: "queued",
      progress: 0,
      message: "Waiting to start",
      prompt,
      resolution,
      createdAt,
      updatedAt: createdAt,
      outputPath: null,
      error: null,
      inputPath: req.file.path,
      originalName: originalName(req.file.originalname),
    });

    processVideoJob({
      jobId,
      filePath: req.file.path,
      originalName: req.file.originalname,
      prompt,
      resolution,
    }).catch((error) => {
      console.error(`Job ${jobId} crashed`, error);
    });

    res.status(202).json({ jobId });
  } catch (error) {
    console.error("Upload handler error", error);
    res.status(500).json({ error: "Failed to start video edit" });
  }
});

app.get("/api/video-edits/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: "Job not found" });
  }
  res.json(publicJob(job));
});

app.get("/api/video-edits/:jobId/result", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: "Job not found" });
  }

  if (job.status !== "completed" || !job.outputPath) {
    return res.status(409).json({ error: "Job not completed" });
  }

  res.download(job.outputPath, job.outputFilename ?? `${job.id}.mp4`);
});

app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(413).json({ error: "Video is too large. Max size is 10MB." });
    }
    return res.status(400).json({ error: err.message || "Upload error" });
  }

  console.error("Unhandled error", err);
  res.status(500).json({ error: "Unexpected server error" });
});

const port = Number(process.env.PORT ?? 4000);
app.listen(port, () => {
  console.log(`Decart proxy listening on port ${port}`);
});

async function processVideoJob({ jobId, filePath, originalName, prompt, resolution }) {
  updateJob(jobId, { status: "processing", message: "Uploading to Decart", progress: 15 });

  try {
    if (!DECART_API_KEY) {
      throw new Error("DECART_API_KEY missing. Update your .env.");
    }

    const outputFilename = `${jobId}_${sanitizeFilename(originalName).replace(/\.[^.]+$/, "")}.mp4`;
    const outputPath = path.join(storageDir, outputFilename);

    await callDecartApi({
      jobId,
      filePath,
      prompt,
      resolution,
      outputPath,
    });

    updateJob(jobId, {
      status: "completed",
      progress: 100,
      message: "Video ready",
      outputPath,
      outputFilename,
      inputPath: null,
    });
  } catch (error) {
    console.error(`Job ${jobId} failed`, error?.response?.data ?? error);
    updateJob(jobId, {
      status: "failed",
      progress: 100,
      message: "Video edit failed",
      error: errorMessage(error),
      inputPath: null,
    });
  } finally {
    await safeUnlink(filePath);
  }
}

async function callDecartApi({ jobId, filePath, prompt, resolution, outputPath }) {
  updateJob(jobId, { message: "Contacting Decart", progress: 40 });

  const form = new FormData();
  form.append("prompt", prompt);
  if (resolution) {
    form.append("resolution", resolution);
  }
  form.append("data", fs.createReadStream(filePath));

  const requestConfig = {
    method: "post",
    url: "https://api.decart.ai/v1/generate/lucy-pro-v2v",
    headers: {
      ...form.getHeaders(),
      "X-API-KEY": DECART_API_KEY,
    },
    responseType: "stream",
    maxBodyLength: Infinity,
    maxContentLength: Infinity,
    data: form,
  };

  const response = await axios(requestConfig);
  updateJob(jobId, { message: "Processing", progress: 70 });

  await streamToFile(response.data, outputPath);
  updateJob(jobId, { message: "Finalizing", progress: 90 });
}

async function streamToFile(stream, outputPath) {
  const writer = fs.createWriteStream(outputPath);
  return new Promise((resolve, reject) => {
    stream.pipe(writer);
    stream.on("error", reject);
    writer.on("error", reject);
    writer.on("finish", resolve);
  });
}

function sanitizeFilename(name) {
  return name.replace(/[^a-zA-Z0-9-_\.]/g, "_");
}

function originalName(name = "video") {
  return sanitizeFilename(name).slice(0, 64) || "video";
}

async function safeUnlink(filePath) {
  if (!filePath) return;
  try {
    await fsp.unlink(filePath);
  } catch (error) {
    if (error.code !== "ENOENT") {
      console.warn(`Failed to delete ${filePath}:`, error.message);
    }
  }
}

function errorMessage(error) {
  if (!error) return "Unknown error";
  if (axios.isAxiosError(error)) {
    const status = error.response?.status;
    const detail = error.response?.data;
    if (detail && typeof detail === "object") {
      return detail.error ?? JSON.stringify(detail);
    }
    if (typeof detail === "string") {
      return detail;
    }
    return status ? `Decart API error (${status})` : "Decart API error";
  }
  return error.message ?? String(error);
}

function updateJob(jobId, patch) {
  const job = jobs.get(jobId);
  if (!job) return;
  const next = {
    ...job,
    ...patch,
    updatedAt: new Date().toISOString(),
  };
  jobs.set(jobId, next);
}


