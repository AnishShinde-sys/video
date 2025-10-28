# AI Video Transformer - Web Application

A beautiful web interface for transforming videos using AI. Upload a video, describe what you want, and get a transformed video with AI-generated visuals and synced speech.

## Features

- 🎬 **Drag & Drop Interface** - Easy video upload with drag and drop support
- 🎨 **AI-Powered Transformation** - Uses Gemini AI for image generation and analysis
- 🗣️ **Speech Synthesis** - Extracts transcript and generates video with lip-synced speech
- 📱 **9:16 Vertical Format** - Perfect for social media and mobile viewing
- ⏱️ **12-Second Videos** - Optimized duration for SeedDance API
- 📊 **Real-time Progress** - Live updates during processing
- 💾 **Easy Downloads** - Download both video and image results

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **AI Models**:
  - Gemini 2.0 Flash (vision & text analysis)
  - Gemini 2.5 Flash Image (image generation)
  - SeedDance V1 Pro Fast (image-to-video)
- **Storage**: Backblaze B2

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Make sure your `.env` file contains:

```bash
GEMINI_API_KEY=your_gemini_api_key
EACHLABS_API_KEY=your_eachlabs_api_key
keyID=your_backblaze_key_id
keyName=your_backblaze_bucket_name
applicationKey=your_backblaze_app_key
```

### 3. Run the Application

```bash
python3 app.py
```

The server will start on **http://localhost:5001**

### 4. Open in Browser

Navigate to http://localhost:5001 in your web browser.

## How to Use

1. **Upload Video**:
   - Click the upload area or drag and drop a video file
   - Supported formats: MP4, MOV, AVI, MKV, WebM
   - Max file size: 500MB

2. **Enter Prompt**:
   - Describe what you want to see in the video
   - Example: "person wearing sunglasses and a red baseball cap"
   - Be descriptive about the final scene, not modifications

3. **Transform**:
   - Click "Transform Video" button
   - Wait for processing (typically 3-5 minutes)
   - Watch real-time progress updates

4. **Download Results**:
   - Once complete, view the generated video
   - Download both the video and the generated image
   - Video specs: 1080p, 9:16, 12 seconds

## Processing Pipeline

The application runs through 5 steps:

1. **Extract Frames** - Extracts 7 key frames from your video
2. **Enhance Prompt** - AI analyzes frames and creates action-focused prompt
3. **Generate Image** - Creates a single transformed image using Gemini
4. **Analyze Video** - Extracts complete transcript and movement data
5. **Generate Video** - Creates final video with SeedDance API

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload video and start processing
- `GET /status/<job_id>` - Get processing status
- `GET /download/<filename>` - Download generated files
- `GET /jobs` - List all jobs

## Configuration

### Change Port

Edit [app.py](app.py:211):
```python
app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
```

### Change Max File Size

Edit [app.py](app.py:15):
```python
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
```

### Adjust Video Duration

The video duration is hardcoded to 12 seconds in [video_transformer.py](video_transformer.py:855):
```python
"duration": "12"
```

### Change Aspect Ratio

Edit [video_transformer.py](video_transformer.py:853):
```python
"aspect_ratio": "9:16"  # Change to "16:9" for landscape
```

## Troubleshooting

### Port Already in Use

If port 5001 is in use:
```bash
# Find process using port
lsof -ti:5001

# Kill the process
kill -9 $(lsof -ti:5001)
```

Or change the port in app.py.

### API Key Errors

Make sure all API keys are set in `.env`:
```bash
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GEMINI:', os.getenv('GEMINI_API_KEY')[:10]); print('EACHLABS:', os.getenv('EACHLABS_API_KEY')[:10])"
```

### Processing Fails

Check the Flask console output for detailed error messages. Common issues:
- Invalid video format
- API rate limits
- Network connectivity
- Insufficient API credits

### Upload Fails

Ensure your video is:
- Under 500MB
- In a supported format (MP4, MOV, AVI, MKV, WebM)
- Not corrupted

## Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

Also consider:
- Using Redis for job queue management
- Adding user authentication
- Setting up HTTPS with nginx
- Using a production database for job tracking
- Implementing rate limiting
- Adding file cleanup cron jobs

## File Structure

```
imageediting/
├── app.py                  # Flask application
├── video_transformer.py    # AI pipeline
├── templates/
│   └── index.html         # Web interface
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── app.js         # Frontend logic
├── uploads/               # Uploaded videos
└── output/                # Generated files
```

## Performance Tips

- Use SSD storage for faster frame extraction
- Ensure good internet connection for API calls
- Close other applications to free up memory
- Consider adding caching for repeated transformations

## Credits

Built with:
- [Flask](https://flask.palletsprojects.com/)
- [Google Gemini AI](https://ai.google.dev/)
- [SeedDance by Eachlabs](https://eachlabs.ai/)
- [Backblaze B2](https://www.backblaze.com/b2/)

## License

See LICENSE file for details.

## Support

For issues or questions:
1. Check the Flask console for error messages
2. Review the troubleshooting section
3. Verify all API keys are valid
4. Ensure dependencies are installed correctly
