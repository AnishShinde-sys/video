const form = document.getElementById("edit-form");
const submitBtn = document.getElementById("submit-btn");
const videoInput = document.getElementById("video");
const videoDropzone = document.getElementById("video-dropzone");
const videoPlaceholder = document.getElementById("video-placeholder");
const videoPlaceholderTitle = document.getElementById("video-placeholder-title");
const videoPlaceholderSubtitle = document.getElementById("video-placeholder-subtitle");
const previewVideo = document.getElementById("preview-video");
const resultPreviewBox = document.getElementById("result-preview-box");
const resultPreviewVideo = document.getElementById("result-preview-video");
const resultPreviewTitle = document.getElementById("result-preview-title");
const resultPreviewSubtitle = document.getElementById("result-preview-subtitle");
const resultPreviewPlaceholder = document.querySelector("#result-preview-box .video-placeholder");
const statusSection = document.getElementById("status");
const statusMessageEl = document.getElementById("status-message");
const statusPill = document.getElementById("status-pill");
const progressBar = document.getElementById("progress-bar");
const resultSection = document.getElementById("result");
const resultVideo = document.getElementById("result-video");
const downloadLink = document.getElementById("download-link");
const resetBtn = document.getElementById("reset-btn");
const errorSection = document.getElementById("error");
const errorMessageEl = document.getElementById("error-message");
const retryBtn = document.getElementById("retry-btn");

const HARDCODED_API_BASE = document.body.dataset.apiBase || "https://video-536c.onrender.com";
const params = new URLSearchParams(window.location.search);
const queryApiBase = params.get("apiBase") || params.get("api") || params.get("backend");
const storedApiBase = window.localStorage.getItem("huemo_api_base");

let resolvedApiBase = queryApiBase || storedApiBase || HARDCODED_API_BASE;

if (queryApiBase) {
  window.localStorage.setItem("huemo_api_base", queryApiBase);
}

const API_BASE_URL = resolvedApiBase.replace(/\/$/, "");

function ensureJson(response) {
  const contentType = response.headers.get("content-type") || "";
  return contentType.toLowerCase().includes("application/json");
}

async function parseJsonResponse(response) {
  if (ensureJson(response)) {
    return response.json();
  }

  const fallbackText = await response.text();
  const message = fallbackText?.trim() || `Unexpected response (status ${response.status})`;
  throw new Error(message);
}

let currentJobId = null;
let pollTimer = null;
let previewUrl = null;

if (videoDropzone) {
  videoDropzone.addEventListener("click", () => videoInput?.click());
  videoDropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      videoInput?.click();
    }
  });
}

if (videoInput) {
  videoInput.addEventListener("change", handleVideoSelection);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const video = form.video.files[0];
  const prompt = form.prompt.value.trim();

  if (!video) {
    showError("Please choose a video file before submitting.");
    return;
  }

  if (!prompt) {
    showError("Prompt cannot be empty.");
    return;
  }

  resetUI();
  resetResultPreview();
  toggleForm(false);
  showStatus({ status: "queued", message: "Uploading video...", progress: 5 });

  try {
    const body = new FormData(form);
    const response = await fetch(`${API_BASE_URL}/api/video-edits`, {
      method: "POST",
      body,
    });

    const data = await parseJsonResponse(response);

    if (!response.ok) {
      throw new Error(data?.error || "Failed to start edit");
    }

    currentJobId = data.jobId;
    pollTimer = startPolling(currentJobId);
  } catch (error) {
    showError(error.message || "Something went wrong");
    toggleForm(true);
  }
});

resetBtn.addEventListener("click", () => {
  if (pollTimer) clearInterval(pollTimer);
  currentJobId = null;
  form.reset();
  resetVideoPreview();
  resetResultPreview();
  resetUI();
  toggleForm(true);
});

retryBtn.addEventListener("click", () => {
  errorSection.classList.add("hidden");
  toggleForm(true);
});

function handleVideoSelection() {
  if (!videoInput?.files?.length) {
    resetVideoPreview();
    return;
  }

  const file = videoInput.files[0];
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  if (previewVideo) {
    previewVideo.src = previewUrl;
    previewVideo.classList.remove("hidden");
    previewVideo.load();
    previewVideo.play().catch(() => {});
  }

  if (videoDropzone) {
    videoDropzone.classList.add("has-video");
  }

  if (videoPlaceholderTitle) {
    videoPlaceholderTitle.textContent = "Selected";
  }
  if (videoPlaceholderSubtitle) {
    videoPlaceholderSubtitle.textContent = file.name;
  }
}

function resetVideoPreview() {
  if (previewVideo) {
    previewVideo.pause();
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewVideo.classList.add("hidden");
  }

  if (videoDropzone) {
    videoDropzone.classList.remove("has-video");
  }

  if (videoPlaceholderTitle) {
    videoPlaceholderTitle.textContent = "Click to upload";
  }
  if (videoPlaceholderSubtitle) {
    videoPlaceholderSubtitle.textContent = "Choose a short video (≤10MB) to preview here.";
  }

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function startPolling(jobId) {
  return setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/video-edits/${jobId}`);
      const data = await parseJsonResponse(response);

      if (!response.ok) {
        throw new Error(data?.error || "Failed to fetch status");
      }

      updateStatus(data);

      if (data.status === "completed") {
        clearInterval(pollTimer);
        pollTimer = null;
        showResult(jobId);
        toggleForm(true);
      }

      if (data.status === "failed") {
        clearInterval(pollTimer);
        pollTimer = null;
        showError(data.error || "Video edit failed.");
        toggleForm(true);
      }
    } catch (error) {
      clearInterval(pollTimer);
      pollTimer = null;
      showError(error.message || "Lost connection to server.");
      toggleForm(true);
    }
  }, 2000);
}

function toggleForm(enabled) {
  submitBtn.disabled = !enabled;
  submitBtn.textContent = enabled ? "Submit Edit" : "Submitting...";
}

function updateStatus(job) {
  showStatus(job);
}

function showStatus({ status, message, progress }) {
  statusSection.classList.remove("hidden");
  errorSection.classList.add("hidden");
  resultSection.classList.add("hidden");

  statusPill.textContent = status;
  statusMessageEl.textContent = message || "Processing";
  progressBar.style.width = `${Math.min(progress ?? 0, 100)}%`;
}

function showResult(jobId) {
  statusSection.classList.remove("hidden");
  resultSection.classList.remove("hidden");
  errorSection.classList.add("hidden");

  const fileUrl = `${API_BASE_URL}/api/video-edits/${jobId}/result`;
  resultVideo.src = fileUrl;
  resultVideo.load();
  downloadLink.href = fileUrl;
  updateResultPreview(fileUrl);
}

function showError(message) {
  statusSection.classList.add("hidden");
  resultSection.classList.add("hidden");
  errorSection.classList.remove("hidden");
  errorMessageEl.textContent = message;
  resetResultPreview();
}

function resetUI() {
  statusSection.classList.add("hidden");
  resultSection.classList.add("hidden");
  errorSection.classList.add("hidden");
  progressBar.style.width = "0%";
  statusPill.textContent = "queued";
  statusMessageEl.textContent = "Preparing...";
}

function updateResultPreview(fileUrl) {
  if (!resultPreviewVideo || !resultPreviewBox) return;

  resultPreviewVideo.src = fileUrl;
  resultPreviewVideo.classList.remove("hidden");
  resultPreviewVideo.load();
  resultPreviewVideo.play().catch(() => {});

  resultPreviewBox.classList.add("has-video");
  if (resultPreviewPlaceholder) {
    resultPreviewPlaceholder.classList.add("hidden");
  }
  setResultPreviewMessage("Preview ready", "Play the edited video above.");
}

function resetResultPreview() {
  if (resultPreviewVideo) {
    resultPreviewVideo.pause();
    resultPreviewVideo.removeAttribute("src");
    resultPreviewVideo.load();
    resultPreviewVideo.classList.add("hidden");
  }

  if (resultPreviewBox) {
    resultPreviewBox.classList.remove("has-video");
  }

  if (resultPreviewPlaceholder) {
    resultPreviewPlaceholder.classList.remove("hidden");
  }

  setResultPreviewMessage("Awaiting result", "Submit an edit to see the processed video here.");
}

function setResultPreviewMessage(title, subtitle) {
  if (resultPreviewTitle) {
    resultPreviewTitle.textContent = title;
  }
  if (resultPreviewSubtitle) {
    resultPreviewSubtitle.textContent = subtitle;
  }
}

