#!/usr/bin/env python3
"""
Central Configuration System for AI Video Editor
Handles both localhost development and production deployment configurations
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""

    # Application Settings
    APP_NAME = "AI Video Editor"
    APP_VERSION = "1.0.0"

    # Flask Settings
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

    # Server Configuration
    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", "5001"))
    DEBUG = False
    TESTING = False

    # File Upload Settings
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    JOBS_FOLDER = os.getenv("JOBS_FOLDER", "jobs")
    TEMP_FOLDER = os.getenv("TEMP_FOLDER", "temp")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 500 * 1024 * 1024))  # 500MB
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

    # API Keys (from environment variables only)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EACHLABS_API_KEY = os.getenv("EACHLABS_API_KEY")

    # API Endpoints
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    EACHLABS_BASE_URL = os.getenv("EACHLABS_BASE_URL", "https://api.eachlabs.ai/v1")

    # API Configuration
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    API_RETRY_COUNT = int(os.getenv("API_RETRY_COUNT", "3"))
    API_BACKOFF_FACTOR = float(os.getenv("API_BACKOFF_FACTOR", "1.0"))

    # Model Settings
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
    EACHLABS_MODEL = os.getenv("EACHLABS_MODEL", "seedance-v1-pro-fast-image-to-video")

    # Security Headers (for production)
    SEND_FILE_MAX_AGE_DEFAULT = 0
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

    # CORS Settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS = ["Content-Type", "Authorization"]

    # Task Queue (for future implementation)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

    @classmethod
    def init_app(cls, app):
        """Initialize application with configuration"""
        # Create necessary directories
        for folder in [cls.UPLOAD_FOLDER, cls.JOBS_FOLDER, cls.TEMP_FOLDER]:
            Path(folder).mkdir(parents=True, exist_ok=True)

        # Create subdirectories for temp processing
        temp_subdirs = [
            "step1_frames",
            "step2_prompt",
            "step3_single",
            "step4_complete_analysis",
            "step5_output"
        ]
        for subdir in temp_subdirs:
            (Path(cls.TEMP_FOLDER) / subdir).mkdir(parents=True, exist_ok=True)

        # Create logs directory
        Path("logs").mkdir(exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    FLASK_ENV = "development"
    HOST = "127.0.0.1"  # localhost only

    # Less strict security for development
    SESSION_COOKIE_SECURE = False

    # More verbose logging
    LOG_LEVEL = "DEBUG"

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        print(f"Running in DEVELOPMENT mode on {cls.HOST}:{cls.PORT}")


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False
    FLASK_ENV = "production"
    HOST = "0.0.0.0"  # All interfaces for production

    # Strict security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'

    # Production logging
    LOG_LEVEL = "WARNING"
    LOG_FILE = "/var/log/ai-video-editor/app.log"

    # CORS - Should be restricted in production
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "https://yourdomain.com")

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Add production-specific initialization
        import logging
        from logging.handlers import RotatingFileHandler

        # Set up file logging
        if not app.debug:
            # Ensure log directory exists
            log_dir = Path(cls.LOG_FILE).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                cls.LOG_FILE,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(getattr(logging, cls.LOG_LEVEL))
            app.logger.addHandler(file_handler)

            app.logger.setLevel(getattr(logging, cls.LOG_LEVEL))
            app.logger.info('AI Video Editor startup')

        print(f"Running in PRODUCTION mode on {cls.HOST}:{cls.PORT}")


class TestingConfig(Config):
    """Testing configuration"""

    TESTING = True
    DEBUG = True
    FLASK_ENV = "testing"

    # Use test database/storage
    UPLOAD_FOLDER = "test_uploads"
    JOBS_FOLDER = "test_jobs"
    TEMP_FOLDER = "test_temp"

    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])