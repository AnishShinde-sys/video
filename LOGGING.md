# Logging Configuration Guide

## Environment Variables

### Enable/Disable Logging
```bash
export ENABLE_LOGGING=true    # Enable logging
export ENABLE_LOGGING=false   # Disable logging (default)
```

### Log Level
```bash
export LOG_LEVEL=DEBUG    # Most verbose
export LOG_LEVEL=INFO     # Default level
export LOG_LEVEL=WARNING  # Warnings and errors only
export LOG_LEVEL=ERROR    # Errors only
```

### Log File Location
```bash
export LOG_FILE=logs/app.log              # Custom log file
export LOG_FILE=logs/job_specific.log     # Job-specific logs
# Default: logs/app.log for Flask app, logs/job_{job_id}.log for processing
```

## Usage Examples

### Web Application with Logging
```bash
# Enable logging for Flask app
export ENABLE_LOGGING=true
export LOG_LEVEL=INFO
export LOG_FILE=logs/video_transformer.log
python3 app.py
```

### Command Line with Logging
```bash
# Enable logging for CLI
python3 video_transformer.py video.mp4 "person with hat" \
  --enable-logging \
  --log-level DEBUG \
  --log-file logs/transformation.log
```

### Production Logging
```bash
# Production setup with detailed logging
export ENABLE_LOGGING=true
export LOG_LEVEL=INFO
export LOG_FILE=logs/production.log
python3 app.py
```

## Log Output Format

```
2025-10-28 12:34:56 - root - INFO - STEP: Starting frame extraction for: video.mp4
2025-10-28 12:34:57 - root - DEBUG - DETAIL: Video metadata retrieved successfully
2025-10-28 12:34:58 - root - INFO - SUCCESS: Extracted 7 frames
2025-10-28 12:35:00 - root - ERROR - ERROR: API call failed: 422 Unprocessable Entity
```

## Log Levels Explained

- **DEBUG**: Detailed information for debugging
  - API payloads, file operations, resource usage
- **INFO**: General information about progress
  - Step completions, successful operations
- **WARNING**: Important notices
  - Missing API keys, deprecated features
- **ERROR**: Error conditions
  - API failures, file not found, processing errors

## Log Files

### Default Locations
- **Flask App**: `logs/app.log`
- **Job Processing**: `logs/job_{job_id}.log`
- **CLI Tool**: `logs/video_transformer.log`

### Log Rotation
- Logs are appended to files
- No automatic rotation (implement if needed)
- Consider using `logrotate` for production

## Debugging Common Issues

### Enable Debug Logging
```bash
export ENABLE_LOGGING=true
export LOG_LEVEL=DEBUG
python3 app.py
```

### Check Specific Job Logs
```bash
# Job-specific logs are created automatically
tail -f logs/job_20251028_123456.log
```

### Disable Logging for Performance
```bash
export ENABLE_LOGGING=false
# Or simply don't set the environment variable
python3 app.py
```

## Integration with External Tools

### Syslog Integration
```python
# Add to setup_logging function for syslog
import logging.handlers
syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
logger.addHandler(syslog_handler)
```

### JSON Logging
```python
# For structured logging, modify formatter
import json
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        })
```

## Performance Impact

- **Disabled**: No performance impact
- **INFO Level**: Minimal impact (~1-2%)
- **DEBUG Level**: Moderate impact (~5-10%)
- **File Logging**: Additional I/O overhead

## Security Considerations

- Log files may contain sensitive information
- Ensure proper file permissions (600 or 640)
- Consider log encryption for sensitive data
- Rotate and archive logs regularly
