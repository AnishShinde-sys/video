import logger from './logger.js';

class JobStore {
  constructor() {
    this.jobs = new Map();
  }

  create(jobId, payload) {
    const job = {
      jobId,
      status: 'queued',
      progress: 0,
      step: 'Job queued',
      error: null,
      result: null,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      ...payload
    };

    this.jobs.set(jobId, job);
    logger.debug({ jobId }, 'Job queued');
    return job;
  }

  get(jobId) {
    return this.jobs.get(jobId);
  }

  list() {
    return Array.from(this.jobs.values()).sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  }

  update(jobId, fields) {
    const job = this.jobs.get(jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    const updated = {
      ...job,
      ...fields,
      updatedAt: new Date().toISOString()
    };

    this.jobs.set(jobId, updated);
    return updated;
  }

  progress(jobId, { progress, step, status }) {
    const job = this.jobs.get(jobId);
    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    const nextProgress = Math.max(job.progress, progress);
    const nextStatus = status || job.status;

    return this.update(jobId, {
      status: nextStatus,
      progress: nextProgress,
      step: step || job.step
    });
  }

  complete(jobId, result) {
    logger.info({ jobId }, 'Job completed');
    return this.update(jobId, {
      status: 'completed',
      progress: 100,
      step: 'Job completed successfully',
      result
    });
  }

  fail(jobId, error) {
    const message = typeof error === 'string' ? error : error.message;
    logger.error({ jobId, error: message }, 'Job failed');
    return this.update(jobId, {
      status: 'failed',
      step: 'Job failed',
      error: message,
      progress: 100
    });
  }
}

const jobStore = new JobStore();

export default jobStore;

