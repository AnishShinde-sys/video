const getApiBaseUrl = () => {
  const dataApiBase = document.body.getAttribute('data-api-base');
  if (dataApiBase) {
    return dataApiBase;
  }
  
  const config = window.__APP_CONFIG__ || { apiBaseUrl: '/api' };
  return config.apiBaseUrl;
};

const apiBaseUrl = (() => {
  try {
    const baseUrl = getApiBaseUrl();
    return new URL(baseUrl, window.location.origin);
  } catch (error) {
    console.error('Invalid API base URL, falling back to /api', error);
    return new URL('/api', window.location.origin);
  }
})();

const API_ORIGIN = apiBaseUrl.origin;
const RAW_API_PREFIX = apiBaseUrl.pathname.replace(/\/$/, '');
const API_PREFIX = RAW_API_PREFIX === '' ? '/api' : RAW_API_PREFIX;

const buildApiUrl = (path = '') => {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${API_ORIGIN}${API_PREFIX}${normalized}`;
};

const resolveFileUrl = (path) => {
  if (!path) return null;
  if (/^https?:/i.test(path)) {
    return path;
  }
  if (path.startsWith('/api/')) {
    return `${API_ORIGIN}${path}`;
  }
  return buildApiUrl(path);
};

const state = {
  jobId: null,
  pollInterval: null,
  videoObjectUrl: null
};

const elements = {
  form: document.getElementById('edit-form'),
  videoInput: document.getElementById('video'),
  videoDropzone: document.getElementById('video-dropzone'),
  videoPlaceholder: document.getElementById('video-placeholder'),
  previewVideo: document.getElementById('preview-video'),
  promptInput: document.getElementById('prompt'),
  resolutionSelect: document.getElementById('resolution'),
  submitBtn: document.getElementById('submit-btn'),
  statusPanel: document.getElementById('status'),
  statusPill: document.getElementById('status-pill'),
  statusMessage: document.getElementById('status-message'),
  progressBar: document.getElementById('progress-bar'),
  resultPanel: document.getElementById('result'),
  resultVideo: document.getElementById('result-video'),
  downloadLink: document.getElementById('download-link'),
  errorPanel: document.getElementById('error'),
  errorMessage: document.getElementById('error-message'),
  retryBtn: document.getElementById('retry-btn'),
  resetBtn: document.getElementById('reset-btn'),
  resultPreviewVideo: document.getElementById('result-preview-video'),
  resultPreviewTitle: document.getElementById('result-preview-title'),
  resultPreviewSubtitle: document.getElementById('result-preview-subtitle'),
  resultPreviewPlaceholder: document.querySelector('#result-preview-box .video-placeholder')
};

const setHidden = (element, hidden) => {
  if (!element) return;
  element.classList.toggle('hidden', hidden);
};

const revokeUrl = (url) => {
  if (url) {
    URL.revokeObjectURL(url);
  }
};

const resetPreview = () => {
  revokeUrl(state.videoObjectUrl);
  state.videoObjectUrl = null;

  if (elements.previewVideo) {
    elements.previewVideo.pause();
    elements.previewVideo.removeAttribute('src');
    elements.previewVideo.load();
    setHidden(elements.previewVideo, true);
  }

  setHidden(elements.videoPlaceholder, false);
};

const resetResultPreview = () => {
  if (elements.resultPreviewVideo) {
    elements.resultPreviewVideo.pause();
    elements.resultPreviewVideo.removeAttribute('src');
    elements.resultPreviewVideo.load();
    setHidden(elements.resultPreviewVideo, true);
  }

  if (elements.resultPreviewPlaceholder) {
    setHidden(elements.resultPreviewPlaceholder, false);
  }

  if (elements.resultPreviewTitle) {
    elements.resultPreviewTitle.textContent = 'Awaiting result';
  }
  if (elements.resultPreviewSubtitle) {
    elements.resultPreviewSubtitle.textContent = 'Submit an edit to see the processed video here.';
  }
};

const resetInterface = () => {
  state.jobId = null;
  if (state.pollInterval) {
    clearInterval(state.pollInterval);
    state.pollInterval = null;
  }

  elements.form.reset();
  resetPreview();
  resetResultPreview();

  setHidden(elements.statusPanel, true);
  setHidden(elements.resultPanel, true);
  setHidden(elements.errorPanel, true);

  if (elements.progressBar) {
    elements.progressBar.style.width = '0%';
  }
  if (elements.statusPill) {
    elements.statusPill.textContent = 'queued';
  }
  if (elements.statusMessage) {
    elements.statusMessage.textContent = 'Preparing…';
  }
};

const setSubmitting = (isSubmitting) => {
  if (!elements.submitBtn) return;
  elements.submitBtn.disabled = isSubmitting;
  elements.submitBtn.textContent = isSubmitting ? 'Submitting…' : 'Submit Edit';
};

const updateStatusPanel = ({ status, progress, message }) => {
  setHidden(elements.statusPanel, false);
  if (elements.statusPill && status) {
    elements.statusPill.textContent = status;
  }
  if (elements.statusMessage && message) {
    elements.statusMessage.textContent = message;
  }
  if (elements.progressBar) {
    const safeProgress = Math.max(0, Math.min(progress ?? 0, 100));
    elements.progressBar.style.width = `${safeProgress}%`;
  }
};

const showResult = (result) => {
  if (!result) {
    showError('Result payload missing from job response.');
    return;
  }

  const videoInfo = result.video || {};
  const finalUrl = resolveFileUrl(videoInfo.finalVideo || videoInfo.rawVideo);

  if (!finalUrl) {
    showError('No video URL received from the backend.');
    return;
  }

  setHidden(elements.resultPanel, false);
  setHidden(elements.errorPanel, true);

  elements.resultVideo.src = finalUrl;
  elements.resultVideo.load();
  elements.downloadLink.href = finalUrl;

  if (elements.resultPreviewVideo) {
    elements.resultPreviewVideo.src = finalUrl;
    elements.resultPreviewVideo.load();
    elements.resultPreviewVideo.play().catch(() => {
      /* autoplay can be blocked; ignore */
    });
    setHidden(elements.resultPreviewVideo, false);
  }

  if (elements.resultPreviewPlaceholder) {
    setHidden(elements.resultPreviewPlaceholder, true);
  }

  if (elements.resultPreviewTitle) {
    elements.resultPreviewTitle.textContent = 'Edit complete';
  }
  if (elements.resultPreviewSubtitle) {
    elements.resultPreviewSubtitle.textContent = 'Review the video or download it below.';
  }
};

const showError = (message) => {
  setHidden(elements.errorPanel, false);
  setHidden(elements.statusPanel, true);
  setHidden(elements.resultPanel, true);
  if (elements.errorMessage) {
    elements.errorMessage.textContent = message || 'Something went wrong.';
  }
};

const pollStatus = async () => {
  if (!state.jobId) return;

  try {
    const response = await fetch(buildApiUrl(`/jobs/${state.jobId}`));
    if (!response.ok) {
      throw new Error(`Failed to fetch job status (HTTP ${response.status})`);
    }

    const job = await response.json();
    updateStatusPanel({ status: job.status, progress: job.progress, message: job.step });

    if (job.status === 'completed') {
      clearInterval(state.pollInterval);
      state.pollInterval = null;
      showResult(job.result);
    } else if (job.status === 'failed') {
      clearInterval(state.pollInterval);
      state.pollInterval = null;
      showError(job.error || 'Video processing failed');
    }
  } catch (error) {
    console.error('Status polling failed', error);
    clearInterval(state.pollInterval);
    state.pollInterval = null;
    showError(error.message || 'Status polling failed');
  }
};

const startPolling = () => {
  if (state.pollInterval) {
    clearInterval(state.pollInterval);
  }
  state.pollInterval = setInterval(pollStatus, 2500);
};

const handleFileSelection = (file) => {
  if (!file) {
    resetPreview();
    return;
  }

  if (!file.type.startsWith('video/')) {
    showError('Please choose a valid video file.');
    resetPreview();
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showError('Please choose a video that is 10MB or smaller.');
    resetPreview();
    return;
  }

  revokeUrl(state.videoObjectUrl);
  state.videoObjectUrl = URL.createObjectURL(file);

  if (elements.previewVideo) {
    elements.previewVideo.src = state.videoObjectUrl;
    elements.previewVideo.load();
    elements.previewVideo.play().catch(() => {
      /* autoplay can be blocked; ignore */
    });
    setHidden(elements.previewVideo, false);
  }

  setHidden(elements.videoPlaceholder, true);
};

const handleSubmit = async (event) => {
  event.preventDefault();

  const file = elements.videoInput.files?.[0];
  if (!file) {
    showError('Please choose an original video first.');
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showError('Please choose a video that is 10MB or smaller.');
    return;
  }

  const prompt = elements.promptInput.value.trim();
  if (!prompt) {
    showError('Describe the edit you would like to apply.');
    return;
  }

  const formData = new FormData();
  formData.append('video', file);
  formData.append('prompt', prompt);
  if (elements.resolutionSelect?.value) {
    formData.append('resolution', elements.resolutionSelect.value);
  }

  setSubmitting(true);
  setHidden(elements.errorPanel, true);
  setHidden(elements.resultPanel, true);
  updateStatusPanel({ status: 'queued', progress: 5, message: 'Uploading video…' });

  try {
    const response = await fetch(buildApiUrl('/jobs'), {
      method: 'POST',
      body: formData
    });

    const payload = await response.json().catch(() => ({ error: 'Unexpected response from server.' }));

    if (!response.ok) {
      throw new Error(payload.error || 'Failed to start the edit.');
    }

    state.jobId = payload.jobId;
    startPolling();
  } catch (error) {
    console.error('Upload failed', error);
    showError(error.message || 'Unable to submit edit.');
  } finally {
    setSubmitting(false);
  }
};

const wireDropzone = () => {
  if (!elements.videoDropzone) return;

  elements.videoDropzone.addEventListener('click', () => {
    elements.videoInput.click();
  });

  elements.videoDropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      elements.videoInput.click();
    }
  });

  ['dragenter', 'dragover'].forEach((type) => {
    elements.videoDropzone.addEventListener(type, (event) => {
      event.preventDefault();
      elements.videoDropzone.classList.add('active');
    });
  });

  ['dragleave', 'dragend', 'drop'].forEach((type) => {
    elements.videoDropzone.addEventListener(type, () => {
      elements.videoDropzone.classList.remove('active');
    });
  });

  elements.videoDropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    const [file] = event.dataTransfer.files || [];
    if (file) {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      elements.videoInput.files = transfer.files;
      handleFileSelection(file);
    }
  });
};

elements.form.addEventListener('submit', handleSubmit);

elements.videoInput.addEventListener('change', (event) => {
  handleFileSelection(event.target.files?.[0] || null);
});

elements.retryBtn?.addEventListener('click', () => {
  resetInterface();
});

elements.resetBtn?.addEventListener('click', () => {
  resetInterface();
});

window.addEventListener('beforeunload', () => {
  revokeUrl(state.videoObjectUrl);
  if (state.pollInterval) {
    clearInterval(state.pollInterval);
  }
});

wireDropzone();
resetInterface();
