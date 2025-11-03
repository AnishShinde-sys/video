import path from 'path';
import fs from 'fs-extra';

export async function prepareJobDirectories(jobId, jobsRoot) {
  const jobDir = path.join(jobsRoot, jobId);
  const dirs = {
    jobDir,
    framesDir: path.join(jobDir, 'frames'),
    promptsDir: path.join(jobDir, 'prompts'),
    generatedDir: path.join(jobDir, 'generated'),
    analysisDir: path.join(jobDir, 'analysis'),
    videosDir: path.join(jobDir, 'videos'),
    audioDir: path.join(jobDir, 'audio'),
    finalDir: path.join(jobDir, 'final')
  };

  await Promise.all(Object.values(dirs).map((dir) => fs.ensureDir(dir)));
  return dirs;
}

export async function writeJson(filePath, data) {
  await fs.writeJson(filePath, data, { spaces: 2 });
  return filePath;
}

export function sanitizeFileName(filename) {
  return filename.replace(/[^a-zA-Z0-9_.-]/g, '_');
}

export function relativeToJob(jobDir, targetPath) {
  return path.relative(jobDir, targetPath);
}

export async function deleteIfExists(targetPath) {
  if (await fs.pathExists(targetPath)) {
    await fs.remove(targetPath);
  }
}

