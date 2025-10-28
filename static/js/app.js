// Global state
let currentJobId = null;
let pollInterval = null;

// DOM Elements
const uploadForm = document.getElementById('upload-form');
const videoInput = document.getElementById('video-input');
const promptInput = document.getElementById('prompt-input');
const submitBtn = document.getElementById('submit-btn');
const dropZone = document.getElementById('drop-zone');
const fileNameDisplay = document.getElementById('file-name');

const uploadSection = document.getElementById('upload-section');
const progressSection = document.getElementById('progress-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');

const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const statusText = document.getElementById('status-text');
const errorMessage = document.getElementById('error-message');

// File input handling
videoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileNameDisplay.textContent = `Selected: ${file.name}`;
        fileNameDisplay.style.display = 'block';
    }
});

// Drag and drop handling
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('video/')) {
            videoInput.files = files;
            fileNameDisplay.textContent = `Selected: ${file.name}`;
            fileNameDisplay.style.display = 'block';
        } else {
            alert('Please drop a video file');
        }
    }
});

// Form submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append('video', videoInput.files[0]);
    formData.append('prompt', promptInput.value);

    // Disable form
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').textContent = 'Uploading...';
    submitBtn.querySelector('.btn-loader').style.display = 'inline-block';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        // Start monitoring job
        currentJobId = data.job_id;
        showProgress();
        startPolling();

    } catch (error) {
        showError(error.message);
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').textContent = 'Transform Video';
        submitBtn.querySelector('.btn-loader').style.display = 'none';
    }
});

// Show progress section
function showProgress() {
    uploadSection.style.display = 'none';
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

// Show results section
function showResults(result) {
    uploadSection.style.display = 'none';
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';
    errorSection.style.display = 'none';

    // Set video (use view URL for streaming)
    const resultVideo = document.getElementById('result-video');
    resultVideo.src = result.video_view_url || result.video_url;
    resultVideo.load(); // Force reload

    // Set image (use view URL for display)
    const resultImage = document.getElementById('result-image');
    resultImage.src = result.image_view_url || result.image_url;

    // Set download links (use download URLs)
    document.getElementById('download-video-btn').href = result.video_url;
    document.getElementById('download-image-btn').href = result.image_url;

    // Set info
    document.getElementById('video-duration').textContent = `${result.duration || 0}s`;
    document.getElementById('video-size').textContent = `${(result.size_mb || 0).toFixed(2)} MB`;
}

// Show error section
function showError(error) {
    uploadSection.style.display = 'none';
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'block';

    errorMessage.textContent = error;
}

// Update progress
function updateProgress(progress, step, status) {
    progressFill.style.width = `${progress}%`;
    progressText.textContent = `${progress}%`;
    statusText.textContent = step;

    // Update step indicators
    const steps = document.querySelectorAll('.step');
    steps.forEach((stepEl, index) => {
        const stepNum = index + 1;
        if (stepNum < getCurrentStep(progress)) {
            stepEl.classList.add('completed');
            stepEl.classList.remove('active');
        } else if (stepNum === getCurrentStep(progress)) {
            stepEl.classList.add('active');
            stepEl.classList.remove('completed');
        } else {
            stepEl.classList.remove('active', 'completed');
        }
    });
}

// Get current step based on progress
function getCurrentStep(progress) {
    if (progress < 20) return 1;
    if (progress < 35) return 2;
    if (progress < 50) return 3;
    if (progress < 65) return 4;
    return 5;
}

// Poll job status
function startPolling() {
    // Poll every 2 seconds
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/status/${currentJobId}`);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to get status');
            }

            // Update progress
            updateProgress(
                data.progress || 0,
                data.step || 'Processing...',
                data.status
            );

            // Check if completed
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                showResults(data.result);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                showError(data.error || 'Video processing failed');
            }

        } catch (error) {
            clearInterval(pollInterval);
            showError(error.message);
        }
    }, 2000);
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (pollInterval) {
        clearInterval(pollInterval);
    }
});
