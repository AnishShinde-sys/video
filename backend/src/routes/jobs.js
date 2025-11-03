import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs-extra';
import { v4 as uuidV4 } from 'uuid';

import config from '../config.js';
import jobStore from '../jobStore.js';
import VideoPipeline from '../pipeline/videoPipeline.js';
import logger from '../logger.js';

const router = express.Router();

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, config.storage.uploads);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname);
    const base = path.basename(file.originalname, ext).replace(/[^a-zA-Z0-9_-]/g, '');
    cb(null, `${base || 'video'}_${Date.now()}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 500 * 1024 * 1024 // 500MB
  },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype.startsWith('video/')) {
      cb(null, true);
    } else {
      cb(new Error('Only video uploads are supported'));
    }
  }
});

router.post('/', upload.single('video'), async (req, res, next) => {
  try {
    const { prompt } = req.body;
    if (!req.file) {
      return res.status(400).json({ error: 'Video file is required' });
    }
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    const jobId = `job_${uuidV4()}`;

    jobStore.create(jobId, {
      prompt,
      originalFileName: req.file.originalname,
      videoPath: req.file.path
    });

    res.status(202).json({
      jobId,
      message: 'Video uploaded successfully. Processing started.'
    });

    const pipeline = new VideoPipeline(jobId);
    pipeline
      .run({
        videoPath: req.file.path,
        prompt
      })
      .catch((error) => {
        logger.error({ err: error, jobId }, 'Pipeline execution error');
      });
  } catch (error) {
    next(error);
  }
});

router.get('/', (_req, res) => {
  res.json({ jobs: jobStore.list() });
});

router.get('/:jobId', (req, res) => {
  const job = jobStore.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  return res.json(job);
});

router.get('/:jobId/files/*', async (req, res) => {
  const job = jobStore.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  const jobDir = path.join(config.storage.jobs, req.params.jobId);
  const requestedPath = req.params[0];
  const absolutePath = path.join(jobDir, requestedPath);

  if (!absolutePath.startsWith(jobDir)) {
    return res.status(400).json({ error: 'Invalid path' });
  }

  if (!(await fs.pathExists(absolutePath))) {
    return res.status(404).json({ error: 'File not found' });
  }

  return res.sendFile(absolutePath);
});

export default router;

