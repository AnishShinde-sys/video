import pino from 'pino';
import config from './config.js';

const logger = pino({
  level: process.env.LOG_LEVEL || (config.env === 'development' ? 'debug' : 'info'),
  base: {
    service: 'ai-video-transformer-backend'
  }
});

export default logger;

