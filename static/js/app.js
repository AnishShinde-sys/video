// Global variables
let selectedVideo = null;
let currentJobId = null;
let pollInterval = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const videoInput = document.getElementById('videoInput');
const uploadPrompt = document.getElementById('uploadPrompt');
const videoPreview = document.getElementById('videoPreview');
const previewVideo = document.getElementById('previewVideo');
const videoName = document.getElementById('videoName');
const videoSize = document.getElementById('videoSize');
const uploadSection = document.getElementById('uploadSection');
const promptSection = document.getElementById('promptSection');
const pipelineSection = document.getElementById('pipelineSection');
const resultsSection = document.getElementById('resultsSection');
const userPrompt = document.getElementById('userPrompt');
const errorModal = document.getElementById('errorModal');
const errorMessage = document.getElementById('errorMessage');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingMessage = document.getElementById('loadingMessage');

// Initialize drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('video/')) {
            handleVideoSelection(file);
        } else {
            showError('Please select a valid video file');
        }
    }
});

// Handle file input change
videoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleVideoSelection(file);
    }
});

// Handle video selection
function handleVideoSelection(file) {
    // Check file size (500MB limit)
    const maxSize = 500 * 1024 * 1024; // 500MB in bytes
    if (file.size > maxSize) {
        showError('File size exceeds 500MB limit. Please select a smaller video.');
        return;
    }

    selectedVideo = file;

    // Show preview
    const videoURL = URL.createObjectURL(file);
    previewVideo.src = videoURL;
    videoName.textContent = file.name;
    videoSize.textContent = formatFileSize(file.size);

    uploadPrompt.style.display = 'none';
    videoPreview.style.display = 'block';

    // Show prompt section
    promptSection.style.display = 'block';

    // Scroll to prompt section
    promptSection.scrollIntoView({ behavior: 'smooth' });
}

// Remove video
function removeVideo() {
    selectedVideo = null;
    previewVideo.src = '';
    uploadPrompt.style.display = 'block';
    videoPreview.style.display = 'none';
    promptSection.style.display = 'none';
    videoInput.value = '';
}

// Set prompt from suggestion
function setPrompt(text) {
    userPrompt.value = text;
    userPrompt.focus();
}

// Start processing
async function startProcessing() {
    if (!selectedVideo) {
        showError('Please select a video first');
        return;
    }

    const prompt = userPrompt.value.trim();
    if (!prompt) {
        showError('Please enter a transformation description');
        return;
    }

    // Show pipeline section
    pipelineSection.style.display = 'block';
    resultsSection.style.display = 'none';

    // Reset pipeline UI
    resetPipeline();

    // Scroll to pipeline
    pipelineSection.scrollIntoView({ behavior: 'smooth' });

    // Disable process button
    document.getElementById('processBtn').disabled = true;

    // Create FormData
    const formData = new FormData();
    formData.append('video', selectedVideo);
    formData.append('prompt', prompt);

    try {
        // Show loading
        showLoading('Uploading video and starting processing...');

        // Send to backend
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        currentJobId = data.job_id;
        hideLoading();

        // Start polling for status
        startPolling();

    } catch (error) {
        hideLoading();
        showError(`Failed to start processing: ${error.message}`);
        document.getElementById('processBtn').disabled = false;
    }
}

// Start polling for job status
function startPolling() {
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${currentJobId}`);
            const data = await response.json();

            updatePipelineStatus(data);

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                await showResults();
            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                showError(`Processing failed: ${data.error}`);
                document.getElementById('processBtn').disabled = false;
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000); // Poll every 2 seconds
}

// Update pipeline status
function updatePipelineStatus(data) {
    const steps = data.steps || {};

    Object.keys(steps).forEach(stepKey => {
        const stepNum = stepKey.replace('step', '');
        const step = steps[stepKey];

        const stepElement = document.getElementById(`step${stepNum}`);
        const progress = document.getElementById(`progress${stepNum}`);
        const status = document.getElementById(`status${stepNum}`);

        if (stepElement) {
            // Remove all status classes
            stepElement.classList.remove('active', 'completed', 'error');

            if (step.status === 'processing') {
                stepElement.classList.add('active');
                progress.style.width = `${step.progress || 50}%`;
                status.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                status.className = 'step-status';
            } else if (step.status === 'completed') {
                stepElement.classList.add('completed');
                progress.style.width = '100%';
                status.innerHTML = '<i class="fas fa-check-circle"></i>';
                status.className = 'step-status completed';
            } else if (step.status === 'error') {
                stepElement.classList.add('error');
                status.innerHTML = '<i class="fas fa-times-circle"></i>';
                status.className = 'step-status error';
            }
        }
    });
}

// Show results
async function showResults() {
    try {
        showLoading('Loading results...');

        const response = await fetch(`/api/results/${currentJobId}`);
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Set videos
        document.getElementById('originalVideo').src = URL.createObjectURL(selectedVideo);
        document.getElementById('transformedVideo').src = data.video_url;

        // Set prompts
        document.getElementById('originalPrompt').textContent = userPrompt.value;
        document.getElementById('enhancedPrompt').textContent = data.enhanced_prompt || 'Not available';

        // Set frames
        const framesGrid = document.getElementById('framesGrid');
        framesGrid.innerHTML = '';
        if (data.frames && data.frames.length > 0) {
            data.frames.forEach(frameUrl => {
                const img = document.createElement('img');
                img.src = frameUrl;
                framesGrid.appendChild(img);
            });
        }

        // Set analysis
        document.getElementById('analysisText').textContent = data.analysis || 'Analysis not available';

        // Set download button
        document.getElementById('downloadBtn').onclick = () => {
            window.open(data.video_url, '_blank');
        };

        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        hideLoading();

    } catch (error) {
        hideLoading();
        showError(`Failed to load results: ${error.message}`);
    }
}

// Reset pipeline UI
function resetPipeline() {
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        const progress = document.getElementById(`progress${i}`);
        const status = document.getElementById(`status${i}`);

        if (step) {
            step.classList.remove('active', 'completed', 'error');
            progress.style.width = '0%';
            status.innerHTML = '<i class="fas fa-clock"></i>';
            status.className = 'step-status';
        }
    }
}

// Reset app
function resetApp() {
    // Clear interval if running
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    // Reset variables
    selectedVideo = null;
    currentJobId = null;

    // Reset UI
    removeVideo();
    userPrompt.value = '';
    pipelineSection.style.display = 'none';
    resultsSection.style.display = 'none';
    document.getElementById('processBtn').disabled = false;

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show tab
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}Tab`).style.display = 'block';

    // Add active class to clicked button
    event.target.classList.add('active');
}

// Show error
function showError(message) {
    errorMessage.textContent = message;
    errorModal.style.display = 'flex';
}

// Hide error
function hideError() {
    errorModal.style.display = 'none';
}

// Show loading
function showLoading(message = 'Processing...') {
    loadingMessage.textContent = message;
    loadingOverlay.style.display = 'flex';
}

// Hide loading
function hideLoading() {
    loadingOverlay.style.display = 'none';
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Handle keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Escape key to close modals
    if (e.key === 'Escape') {
        hideError();
    }
});