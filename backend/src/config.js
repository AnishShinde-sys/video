import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import fs from 'fs-extra';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..');

dotenv.config({ path: path.join(projectRoot, '.env') });

const storageRoot = process.env.STORAGE_ROOT || path.join(projectRoot, 'storage');
const uploadsDir = path.join(storageRoot, 'uploads');
const jobsDir = path.join(storageRoot, 'jobs');

fs.ensureDirSync(uploadsDir);
fs.ensureDirSync(jobsDir);

const config = {
  env: process.env.NODE_ENV || 'development',
  port: Number(process.env.PORT) || 4000,
  host: process.env.HOST || '0.0.0.0',
  geminiApiKey: process.env.GEMINI_API_KEY,
  seedanceApiKey: process.env.EACHLABS_API_KEY,
  backblaze: {
    keyId: process.env.keyID,
    keyName: process.env.keyName,
    applicationKey: process.env.applicationKey
  },
  storage: {
    root: storageRoot,
    uploads: uploadsDir,
    jobs: jobsDir
  },
  pipeline: {
    frameCount: Number(process.env.FRAME_COUNT) || 7,
    skipAnalysis: process.env.SKIP_ANALYSIS === 'true',
    skipVideoGeneration: process.env.SKIP_VIDEO_GENERATION === 'true'
  }
};

export default config;

