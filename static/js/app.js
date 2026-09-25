/**
 * ASLO — American Sign Language Online Studio Engine
 * Orchestrates real-time telemetry, Web Speech API (STT & TTS),
 * Active Learning model correction, camera HUD, and interactive controls.
 */

// ── DOM References ──────────────────────────────────────────────────────────
const dom = {
    // Camera & HUD
    cameraFeed: document.getElementById("camera-feed"),
    camFallback: document.getElementById("cam-fallback"),
    hudConfidence: document.getElementById("hud-confidence"),
    hudTrackingPill: document.getElementById("hud-tracking-pill"),
    hudCamName: document.getElementById("hud-cam-name"),
    btnFullscreen: document.getElementById("btn-fullscreen"),
    btnMirror: document.getElementById("btn-mirror"),
    btnSwitchCam: document.getElementById("btn-switch-cam"),
    camTelemetryHud: document.getElementById("cam-telemetry-hud"),
    camHudToggleBtn: document.getElementById("cam-hud-toggle-btn"),
    camRecentTrail: document.getElementById("cam-recent-trail"),
    hudRecTag: document.getElementById("hud-rec-tag"),
    hudRecDot: document.getElementById("hud-rec-dot"),
    hudRecStatus: document.getElementById("hud-rec-status"),

    // Telemetry & Hero Gesture
    statusBadge: document.getElementById("status-badge"),
    gestureLabel: document.getElementById("gesture-label"),
    statePill: document.getElementById("state-pill"),
    confBar: document.getElementById("conf-bar"),
    confText: document.getElementById("conf-text"),
    stabSegments: document.getElementById("stab-segments"),
    stabText: document.getElementById("stab-text"),
    recDot: document.getElementById("rec-dot"),
    recLabel: document.getElementById("rec-label"),

    // Audio & Speech (STT & TTS)
    sttBtn: document.getElementById("stt-btn"),
    sttBtnText: document.getElementById("stt-btn-text"),
    sttLangSelect: document.getElementById("stt-lang-select"),
    sttVisualizer: document.getElementById("stt-visualizer"),
    sttLivePreview: document.getElementById("stt-live-preview"),
    sttAppendCheckbox: document.getElementById("stt-append-checkbox"),
    sttIndicator: document.getElementById("stt-indicator"),
    sttIndicatorText: document.getElementById("stt-indicator-text"),
    ttsToggleBtn: document.getElementById("tts-toggle-btn"),
    ttsToggleText: document.getElementById("tts-toggle-text"),
    btnSpeakSentence: document.getElementById("btn-speak-sentence"),

    // Sentence Canvas & Dialogue Log
    tabCanvas: document.getElementById("tab-canvas"),
    tabDialogue: document.getElementById("tab-dialogue"),
    viewportCanvas: document.getElementById("viewport-canvas"),
    viewportDialogue: document.getElementById("viewport-dialogue"),
    dialogueFeed: document.getElementById("dialogue-feed"),
    sentenceBox: document.getElementById("sentence-box"),
    charCount: document.getElementById("char-count"),
    wordCount: document.getElementById("word-count"),
    btnCopy: document.getElementById("btn-copy"),
    btnDownload: document.getElementById("btn-download"),
    btnClearSentence: document.getElementById("btn-clear-sentence"),

    // Active Learning
    correctInput: document.getElementById("correct-input"),
    btnTeach: document.getElementById("btn-teach"),
    correctStatus: document.getElementById("correct-status"),

    // Primary Actions
    recToggleBtn: document.getElementById("rec-toggle-btn"),
    stopBtn: document.getElementById("stop-btn"),

    // Modals & Toasts
    welcomeModal: document.getElementById("welcome-modal"),
    gestureModal: document.getElementById("gesture-modal"),
    toastContainer: document.getElementById("toast-container"),
    gestureSearch: document.getElementById("gesture-search")
};

// ── State Management ────────────────────────────────────────────────────────
let isTTSEnabled = true;
let isRecordingVoice = false;
let isGestureRecordingActive = false;
let lastActiveState = false;
let lastSentenceText = "";
let selectedSTTLanguage = "en-US";
let audioCtx = null;
let micStream = null;
let micSource = null;
let analyserNode = null;
let scriptProcessorNode = null;
let audioPCMChunks = [];
let visualizerAnimId = null;
let pollTimer = null;
let lastBufMax = -1;
let recognition = null;
let speakTimeout = null;
let isMirrored = false;
let recentSigns = [];
let currentCameraIndex = 0;
let isSwitchingCamera = false;

// ── Camera Telemetry HUD & Recent Signs Trail ───────────────────────────────
function toggleCamTelemetryHud() {
    if (!dom.camTelemetryHud) return;
    const isCollapsed = dom.camTelemetryHud.classList.toggle("collapsed");
    const minIcon = dom.camHudToggleBtn ? dom.camHudToggleBtn.querySelector(".icon-minimize") : null;
    const expIcon = dom.camHudToggleBtn ? dom.camHudToggleBtn.querySelector(".icon-expand") : null;

    if (minIcon) minIcon.style.display = isCollapsed ? "none" : "block";
    if (expIcon) expIcon.style.display = isCollapsed ? "block" : "none";
    showToast(isCollapsed ? "Camera HUD minimized" : "Camera HUD expanded", "info", 1500);
}

function stripInternalSuffix(str) {
    if (!str || str === "—" || str === "none") return "—";
    return String(str).replace(/_(START|END)$/i, "").trim();
}

function addRecentSign(sign) {
    if (!sign || sign === "—" || sign === "NEUTRAL") return;
    const clean = stripInternalSuffix(sign).trim().toUpperCase();
    if (!clean || clean === "—") return;

    if (recentSigns.length > 0 && recentSigns[recentSigns.length - 1] === clean) {
        return;
    }

    recentSigns.push(clean);
    if (recentSigns.length > 8) {
        recentSigns.shift();
    }
    renderRecentSigns();
}

function renderRecentSigns() {
    if (!dom.camRecentTrail) return;
    if (recentSigns.length === 0) {
        dom.camRecentTrail.innerHTML = '<span class="cam-recent-placeholder">No recent signs</span>';
        return;
    }

    dom.camRecentTrail.innerHTML = recentSigns.map((s, idx) => {
        const isLatest = idx === recentSigns.length - 1;
        return `<span class="cam-recent-chip ${isLatest ? 'latest' : ''}">${escapeHTML(s)}</span>`;
    }).join("");

    dom.camRecentTrail.scrollLeft = dom.camRecentTrail.scrollWidth;
}

// ── Toast Notification System ───────────────────────────────────────────────
function showToast(message, type = "info", duration = 3500) {
    if (!dom.toastContainer) return;

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    // Icon mapping
    const icons = {
        info: "ℹ️",
        success: "✓",
        warning: "⚠️",
        error: "✕"
    };

    toast.innerHTML = `<span style="font-weight:700;margin-right:4px;">${icons[type] || "•"}</span> ${message}`;
    dom.toastContainer.appendChild(toast);

    // Trigger animate in
    requestAnimationFrame(() => toast.classList.add("show"));

    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ── Audio Engine Auto-Unlock (Web Audio Autoplay Policy) ───────────────────
function unlockAudio() {
    try {
        if (window.speechSynthesis) {
            const u = new SpeechSynthesisUtterance("");
            u.volume = 0;
            window.speechSynthesis.speak(u);
        }
    } catch (e) {
        console.warn("SpeechSynthesis auto-unlock:", e);
    }
}

// Auto-prime audio on first user click or keypress anywhere on page
window.addEventListener("click", () => {
    unlockAudio();
}, { once: true });
window.addEventListener("keydown", () => {
    unlockAudio();
}, { once: true });

let camRetryCount = 0;
let camAutoRetryTimer = null;

if (dom.cameraFeed) {
    dom.cameraFeed.onerror = () => {
        console.warn("[Camera] Video stream dropped or hiccup, auto-reconnecting...");
        if (camAutoRetryTimer) clearTimeout(camAutoRetryTimer);
        camAutoRetryTimer = setTimeout(() => {
            reconnectCamera(true);
        }, 1000);
    };

    dom.cameraFeed.onload = () => {
        camRetryCount = 0;
        dom.cameraFeed.style.display = "block";
        if (dom.camFallback) dom.camFallback.style.display = "none";
    };
}

function reconnectCamera(isSilent = false) {
    if (!dom.cameraFeed) return;
    dom.cameraFeed.src = "/video_feed?t=" + Date.now();
    dom.cameraFeed.style.display = "block";
    if (dom.camFallback) dom.camFallback.style.display = "none";
    if (!isSilent) showToast("Reconnecting video feed...", "info");
}

function toggleFullscreen() {
    const camPanel = document.querySelector(".cam-panel");
    if (!document.fullscreenElement) {
        if (camPanel.requestFullscreen) camPanel.requestFullscreen();
        else if (camPanel.webkitRequestFullscreen) camPanel.webkitRequestFullscreen();
    } else {
        if (document.exitFullscreen) document.exitFullscreen();
    }
}

function toggleCameraMirror() {
    isMirrored = !isMirrored;
    if (dom.cameraFeed) {
        dom.cameraFeed.style.transform = isMirrored ? "scaleX(-1)" : "scaleX(1)";
        showToast(isMirrored ? "Camera feed mirrored" : "Camera feed normal", "info");
    }
}

function switchCamera(targetIndex = null) {
    if (isSwitchingCamera) return;
    isSwitchingCamera = true;

    if (dom.btnSwitchCam) {
        dom.btnSwitchCam.classList.add("is-switching");
        dom.btnSwitchCam.disabled = true;
    }

    const payload = targetIndex !== null ? { index: targetIndex } : {};

    fetch("/api/camera/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.ok) {
            currentCameraIndex = data.camera_index;
            if (dom.hudCamName) {
                dom.hudCamName.textContent = `CAM ${data.camera_index} · LIVE FEED`;
            }
            if (dom.btnSwitchCam) {
                dom.btnSwitchCam.title = `Switch Camera (Current: CAM ${data.camera_index}, Shortcut: X)`;
            }
            // Reconnect feed after hardware initializes
            setTimeout(() => {
                reconnectCamera(true);
            }, 350);
            showToast(`Switched to Camera ${data.camera_index}`, "success", 2500);
        } else {
            showToast(data.error || "Failed to switch camera", "error");
        }
    })
    .catch(err => {
        console.error("Camera switch error:", err);
        showToast("Network error switching camera", "error");
    })
    .finally(() => {
        setTimeout(() => {
            isSwitchingCamera = false;
            if (dom.btnSwitchCam) {
                dom.btnSwitchCam.classList.remove("is-switching");
                dom.btnSwitchCam.disabled = false;
            }
        }, 600);
    });
}

// ── Speech-To-Text (STT) Dual-Engine Integration ───────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    try {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = selectedSTTLanguage;

        recognition.onresult = (event) => {
            let interimTranscript = "";
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    interimTranscript = event.results[i][0].transcript.trim();
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            if (interimTranscript && dom.sttLivePreview) {
                dom.sttLivePreview.textContent = `🎙️ "${interimTranscript}"`;
            }
        };

        recognition.onerror = (event) => {
            console.warn("Native SpeechRecognition event:", event.error);
        };
    } catch (e) {
        console.warn("Web Speech API init failed, using backend engine:", e);
        recognition = null;
    }
}

function changeSTTLanguage(lang) {
    selectedSTTLanguage = lang;
    if (recognition) {
        recognition.lang = lang;
    }
    const langNames = {
        "en-US": "English (US)",
        "vi-VN": "Vietnamese",
        "es-ES": "Spanish",
        "fr-FR": "French",
        "de-DE": "German",
        "ja-JP": "Japanese",
        "ko-KR": "Korean",
        "zh-CN": "Chinese"
    };
    showToast(`Voice language set to ${langNames[lang] || lang}`, "info");
}

// ── Pure Client-Side 16kHz PCM WAV Audio Encoder ─────────────────────────────
function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
    if (inputSampleRate === outputSampleRate || inputSampleRate <= 0) {
        return buffer;
    }
    const ratio = inputSampleRate / outputSampleRate;
    const newLength = Math.max(1, Math.round(buffer.length / ratio));
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
        const idx = Math.min(buffer.length - 1, Math.floor(i * ratio));
        result[i] = buffer[idx] || 0;
    }
    return result;
}

function encodeWAV(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF identifier
    writeString(view, 0, 'RIFF');
    // file length
    view.setUint32(4, 36 + samples.length * 2, true);
    // RIFF type
    writeString(view, 8, 'WAVE');
    // format chunk identifier
    writeString(view, 12, 'fmt ');
    // format chunk length
    view.setUint32(16, 16, true);
    // sample format (raw PCM = 1)
    view.setUint16(20, 1, true);
    // channel count (mono = 1)
    view.setUint16(22, 1, true);
    // sample rate
    view.setUint32(24, sampleRate, true);
    // byte rate (sampleRate * blockAlign)
    view.setUint32(28, sampleRate * 2, true);
    // block align (channels * bytes per sample = 2)
    view.setUint16(32, 2, true);
    // bits per sample = 16
    view.setUint16(34, 16, true);
    // data chunk identifier
    writeString(view, 36, 'data');
    // data chunk length
    view.setUint32(40, samples.length * 2, true);

    // Write PCM samples
    floatTo16BitPCM(view, 44, samples);
    return new Blob([view], { type: 'audio/wav' });
}

// ── Audio Waveform Visualizer ───────────────────────────────────────────────
function renderVisualizerWaveform() {
    if (!dom.sttVisualizer || !analyserNode) return;
    const canvas = dom.sttVisualizer;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
        if (!isRecordingVoice) {
            // Draw idle resting wave
            ctx.clearRect(0, 0, width, height);
            ctx.strokeStyle = "rgba(0, 240, 255, 0.25)";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();
            return;
        }

        visualizerAnimId = requestAnimationFrame(draw);
        analyserNode.getByteTimeDomainData(dataArray);

        ctx.fillStyle = "rgba(7, 9, 14, 0.4)";
        ctx.fillRect(0, 0, width, height);

        ctx.lineWidth = 2;
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, "#00f0ff");
        gradient.addColorStop(0.5, "#10b981");
        gradient.addColorStop(1, "#a855f7");
        ctx.strokeStyle = gradient;

        ctx.beginPath();
        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * height) / 2;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();
    };

    draw();
}

// ── Browser Microphone Recording (AudioContext + Backend Fallback) ──────────
async function startVoiceRecording() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showToast("Microphone recording not supported on this browser", "error");
            return;
        }

        audioPCMChunks = [];
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContextClass();

        micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        micSource = audioCtx.createMediaStreamSource(micStream);
        analyserNode = audioCtx.createAnalyser();
        analyserNode.fftSize = 512;

        // ScriptProcessorNode for pure PCM float32 extraction
        scriptProcessorNode = audioCtx.createScriptProcessor(4096, 1, 1);
        scriptProcessorNode.onaudioprocess = (e) => {
            if (!isRecordingVoice) return;
            const inputData = e.inputBuffer.getChannelData(0);
            audioPCMChunks.push(new Float32Array(inputData));
        };

        micSource.connect(analyserNode);
        analyserNode.connect(scriptProcessorNode);
        
        // Prevent microphone audio from playing back through user speakers (anti-feedback)
        const silentGain = audioCtx.createGain();
        silentGain.gain.value = 0;
        scriptProcessorNode.connect(silentGain);
        silentGain.connect(audioCtx.destination);

        isRecordingVoice = true;

        if (dom.sttBtn) dom.sttBtn.classList.add("recording");
        if (dom.sttBtnText) dom.sttBtnText.textContent = "Stop Voice";
        if (dom.sttIndicator) {
            dom.sttIndicator.innerHTML = `<span class="stt-dot active"></span> <span id="stt-indicator-text">Microphone listening...</span>`;
        }
        if (dom.sttLivePreview) {
            dom.sttLivePreview.textContent = "🎙️ Listening... Speak clearly now";
        }

        renderVisualizerWaveform();

        // Optional Web Speech API for low-latency live preview
        if (recognition) {
            try {
                recognition.start();
            } catch (e) {
                // Already started or ignored
            }
        }

        showToast("Microphone listening — speak now", "info");
    } catch (err) {
        console.error("Microphone access error:", err);
        showToast("Could not access microphone: " + (err.message || err.name), "error");
        stopVoiceRecordingLocally();
    }
}

async function stopVoiceRecording() {
    if (!isRecordingVoice) return;
    isRecordingVoice = false;

    if (dom.sttBtn) dom.sttBtn.classList.remove("recording");
    if (dom.sttBtnText) dom.sttBtnText.textContent = "Voice Input";
    if (dom.sttIndicator) {
        dom.sttIndicator.innerHTML = `<span class="stt-dot"></span> <span id="stt-indicator-text">Transcribing speech...</span>`;
    }
    if (dom.sttLivePreview) {
        dom.sttLivePreview.textContent = "⏳ Transcribing audio with ASLO Neural Engine...";
    }

    if (recognition) {
        try { recognition.stop(); } catch (e) {}
    }

    if (visualizerAnimId) {
        cancelAnimationFrame(visualizerAnimId);
        visualizerAnimId = null;
    }

    // Disconnect audio nodes
    if (scriptProcessorNode) scriptProcessorNode.disconnect();
    if (analyserNode) analyserNode.disconnect();
    if (micSource) micSource.disconnect();
    if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
    }

    const inputSampleRate = audioCtx ? audioCtx.sampleRate : 44100;
    if (audioCtx && audioCtx.state !== 'closed') {
        try { await audioCtx.close(); } catch (e) {}
    }

    // Process recorded audio PCM chunks
    if (audioPCMChunks.length === 0) {
        if (dom.sttLivePreview) dom.sttLivePreview.textContent = "No audio recorded.";
        stopVoiceRecordingLocally();
        return;
    }

    let totalLength = 0;
    for (const chunk of audioPCMChunks) {
        totalLength += chunk.length;
    }

    const mergedFloat32 = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of audioPCMChunks) {
        mergedFloat32.set(chunk, offset);
        offset += chunk.length;
    }

    // Downsample to 16kHz for speech recognition
    const downsampled16k = downsampleBuffer(mergedFloat32, inputSampleRate, 16000);
    const wavBlob = encodeWAV(downsampled16k, 16000);

    const isAppend = dom.sttAppendCheckbox ? dom.sttAppendCheckbox.checked : true;

    // Send audio to Python Flask backend
    const formData = new FormData();
    formData.append("audio", wavBlob, "voice_input.wav");
    formData.append("language", selectedSTTLanguage);
    formData.append("append", isAppend ? "true" : "false");

    fetch("/api/stt/transcribe", {
        method: "POST",
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.ok && data.text) {
            const transcript = data.text;
            if (dom.sttLivePreview) {
                dom.sttLivePreview.textContent = `Transcribed: "${transcript}"`;
            }
            addDialogueMessage("voice", transcript);
            showToast(`Voice transcribed: "${transcript}"`, "success");

            if (isAppend) {
                // Trigger live canvas counters refresh
                setTimeout(poll, 100);
            }
        } else {
            const err = data.error || "Could not transcribe audio.";
            if (dom.sttLivePreview) {
                dom.sttLivePreview.textContent = err;
            }
            showToast(err, "warning");
        }
        stopVoiceRecordingLocally();
    })
    .catch(err => {
        console.error("Transcription request failed:", err);
        if (dom.sttLivePreview) dom.sttLivePreview.textContent = "Transcription connection error.";
        showToast("Error communicating with transcription server", "error");
        stopVoiceRecordingLocally();
    });
}

function toggleVoiceRecording() {
    if (isRecordingVoice) {
        stopVoiceRecording();
    } else {
        startVoiceRecording();
    }
}

function stopVoiceRecordingLocally() {
    isRecordingVoice = false;
    if (dom.sttBtn) dom.sttBtn.classList.remove("recording");
    if (dom.sttBtnText) dom.sttBtnText.textContent = "Voice Input";
    if (dom.sttIndicator) {
        dom.sttIndicator.innerHTML = `<span class="stt-dot"></span> <span id="stt-indicator-text">Voice input standby</span>`;
    }
    renderVisualizerWaveform();
}




// ── View Switcher & Dialogue Log Helpers ─────────────────────────────────────
function switchCanvasTab(tab) {
    if (tab === "canvas") {
        if (dom.tabCanvas) dom.tabCanvas.classList.add("active");
        if (dom.tabDialogue) dom.tabDialogue.classList.remove("active");
        if (dom.viewportCanvas) dom.viewportCanvas.style.display = "block";
        if (dom.viewportDialogue) dom.viewportDialogue.style.display = "none";
    } else {
        if (dom.tabCanvas) dom.tabCanvas.classList.remove("active");
        if (dom.tabDialogue) dom.tabDialogue.classList.add("active");
        if (dom.viewportCanvas) dom.viewportCanvas.style.display = "none";
        if (dom.viewportDialogue) dom.viewportDialogue.style.display = "block";
    }
}

function addDialogueMessage(sender, text) {
    if (!dom.dialogueFeed || !text) return;
    const placeholder = dom.dialogueFeed.querySelector(".dialogue-placeholder");
    if (placeholder) placeholder.remove();

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const bubble = document.createElement("div");
    bubble.className = `dialogue-bubble ${sender}`;

    const icon = sender === "voice" ? "🗣️" : sender === "signer" ? "🤟" : "ℹ️";
    const label = sender === "voice" ? "Voice Input" : sender === "signer" ? "Sign Translation" : "System";

    bubble.innerHTML = `
        <div class="dialogue-meta">
            <span class="dialogue-sender">${icon} ${label}</span>
            <span class="dialogue-time">${timeStr}</span>
        </div>
        <div class="dialogue-body">${escapeHTML(text)}</div>
    `;

    dom.dialogueFeed.appendChild(bubble);
    dom.dialogueFeed.scrollTop = dom.dialogueFeed.scrollHeight;
}

function escapeHTML(str) {
    return String(str).replace(/[&<>'"]/g,
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

// ── Text-To-Speech (TTS) Integration ────────────────────────────────────────
function toggleTTS() {
    isTTSEnabled = !isTTSEnabled;
    if (dom.ttsToggleBtn) {
        if (isTTSEnabled) {
            dom.ttsToggleBtn.classList.add("active");
            dom.ttsToggleText.textContent = "TTS Enabled";
            showToast("Speech synthesis unmuted", "info");
        } else {
            dom.ttsToggleBtn.classList.remove("active");
            dom.ttsToggleText.textContent = "TTS Muted";
            window.speechSynthesis.cancel();
            showToast("Speech synthesis muted", "warning");
        }
    }
}

function speakText(text) {
    if (!isTTSEnabled || !text || !text.trim()) return;

    if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
        window.speechSynthesis.cancel();
    }

    if (speakTimeout) clearTimeout(speakTimeout);

    // Critical 50ms delay to prevent Chromium speech engine freeze bug
    speakTimeout = setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }, 50);
}

function speakWholeSentence() {
    const text = dom.sentenceBox ? dom.sentenceBox.textContent : "";
    if (text && text !== "(empty)") {
        const wasEnabled = isTTSEnabled;
        isTTSEnabled = true;
        speakText(text);
        isTTSEnabled = wasEnabled;
        showToast("Speaking sentence...", "info", 2000);
    } else {
        showToast("Sentence canvas is empty", "warning", 2000);
    }
}

let lastStopSpokenTime = 0;

function speakCompletedSentence(sentenceText) {
    const now = Date.now();
    if (now - lastStopSpokenTime < 1500) return;
    lastStopSpokenTime = now;

    if (!("speechSynthesis" in window)) return;
    if (speakTimeout) clearTimeout(speakTimeout);
    window.speechSynthesis.cancel();

    if (!isTTSEnabled) return;

    const raw = (sentenceText || "").trim();
    const clean = raw.replace(/_(START|END)/gi, "").trim();
    const hasSentence = clean && clean !== "(empty)" && clean !== "—" && !clean.startsWith("(");

    speakTimeout = setTimeout(() => {
        const stopUtt = new SpeechSynthesisUtterance("Recording stopped.");
        stopUtt.rate = 1.05;
        stopUtt.pitch = 1.0;
        window.speechSynthesis.speak(stopUtt);

        if (hasSentence) {
            const sentenceUtt = new SpeechSynthesisUtterance(clean);
            sentenceUtt.rate = 0.92;
            sentenceUtt.pitch = 1.0;
            window.speechSynthesis.speak(sentenceUtt);

            addDialogueMessage("signer", clean);
            showToast(`Sentence completed: "${clean}"`, "success", 3500);
        } else {
            showToast("Recording stopped — No signs in sentence", "info", 2000);
        }
    }, 60);
}


// ── Status Badge ────────────────────────────────────────────────────────────
function setStatusBadge(status, subtext = "") {
    if (!dom.statusBadge) return;
    dom.statusBadge.className = "badge " + status;
    const labels = {
        running: subtext ? `Online (${subtext})` : "Online",
        stopped: "Stopped",
        starting: "Connecting…",
        error: "Offline"
    };
    dom.statusBadge.textContent = labels[status] || status;
}

// ── Segmented Neural Buffer Equalizer ───────────────────────────────────────
function renderStability(bufFill, bufMax, agreeing) {
    if (!dom.stabSegments) return;

    if (bufMax !== lastBufMax) {
        dom.stabSegments.innerHTML = "";
        for (let i = 0; i < bufMax; i++) {
            const seg = document.createElement("div");
            seg.className = "seg";
            dom.stabSegments.appendChild(seg);
        }
        lastBufMax = bufMax;
    }

    const segs = dom.stabSegments.children;
    for (let i = 0; i < segs.length; i++) {
        segs[i].classList.remove("lit", "agree", "disagree");
        if (i < bufFill) {
            segs[i].classList.add("lit", agreeing ? "agree" : "disagree");
        }
    }

    if (dom.stabText) {
        dom.stabText.textContent = `${bufFill}/${bufMax}`;
    }
}

// ── Sentence Word & Character Counters ──────────────────────────────────────
function updateSentenceCounters(text) {
    if (!text || text === "(empty)") {
        if (dom.charCount) dom.charCount.textContent = "0 chars";
        if (dom.wordCount) dom.wordCount.textContent = "0 words";
        return;
    }
    const cleanText = text.trim();
    const chars = cleanText.length;
    const words = cleanText ? cleanText.split(/\s+/).length : 0;

    if (dom.charCount) dom.charCount.textContent = `${chars} char${chars === 1 ? '' : 's'}`;
    if (dom.wordCount) dom.wordCount.textContent = `${words} word${words === 1 ? '' : 's'}`;
}

// ── Sentence Operations (Copy, Download, Clear) ─────────────────────────────
function copySentence() {
    const text = dom.sentenceBox ? dom.sentenceBox.textContent : "";
    if (!text || text === "(empty)") {
        showToast("Sentence is empty to copy", "warning");
        return;
    }

    navigator.clipboard.writeText(text).then(() => {
        showToast("Copied to clipboard!", "success");
    }).catch(() => {
        showToast("Failed to copy to clipboard", "error");
    });
}

function downloadTranscript() {
    const text = dom.sentenceBox ? dom.sentenceBox.textContent : "";
    if (!text || text === "(empty)") {
        showToast("No transcript to download", "warning");
        return;
    }

    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aslo-translation-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Downloaded transcript file", "success");
}

function clearSentence() {
    fetch("/clear", { method: "POST" })
        .then(() => {
            if (dom.sentenceBox) {
                dom.sentenceBox.textContent = "(empty)";
                dom.sentenceBox.className = "empty";
            }
            updateSentenceCounters("");
            recentSigns = [];
            renderRecentSigns();
            showToast("Sentence cleared", "info", 2000);
        })
        .catch(e => console.error("Error clearing sentence:", e));
}

// ── Gesture Recording Toggle ────────────────────────────────────────────────
function toggleGestureRecording() {
    const nextState = !isGestureRecordingActive;
    fetch("/toggle_active", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active: nextState })
    })
    .then(r => r.json())
    .then(data => {
        if (data.ok) {
            updateGestureRecUI(nextState);
            if (nextState) {
                speakText("Recording start");
                showToast("Recording started — System translating signs", "success");
            } else {
                const currentText = dom.sentenceBox ? dom.sentenceBox.textContent : "";
                speakCompletedSentence(currentText);
            }
        }
    })
    .catch(e => console.error("Error toggling active state:", e));
}

function updateGestureRecUI(active) {
    isGestureRecordingActive = active;

    const camPanel = document.querySelector(".cam-panel");
    if (camPanel) {
        camPanel.classList.toggle("recording", active);
    }

    if (dom.recToggleBtn) {
        if (active) {
            dom.recToggleBtn.classList.add("active");
            dom.recToggleBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="6" y="4" width="4" height="16"></rect>
                    <rect x="14" y="4" width="4" height="16"></rect>
                </svg>
                Pause Recording (Sign STOP)
            `;
        } else {
            dom.recToggleBtn.classList.remove("active");
            dom.recToggleBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polygon points="10 8 16 12 10 16 10 8"></polygon>
                </svg>
                Start Recording (Sign START)
            `;
        }
    }

    if (dom.statePill) {
        if (active) {
            dom.statePill.className = "state-pill active";
            dom.statePill.textContent = "RECORDING";
        } else {
            dom.statePill.className = "state-pill";
            dom.statePill.textContent = "STANDBY";
        }
    }

    if (dom.hudRecTag) {
        if (active) {
            dom.hudRecTag.className = "hud-tag rec-tag active";
            if (dom.hudRecDot) dom.hudRecDot.className = "rec-dot-red";
            if (dom.hudRecStatus) dom.hudRecStatus.textContent = "REC";
        } else {
            dom.hudRecTag.className = "hud-tag rec-tag paused";
            if (dom.hudRecDot) dom.hudRecDot.className = "rec-dot-amber";
            if (dom.hudRecStatus) dom.hudRecStatus.textContent = "STANDBY";
        }
    }
}

// ── Active Learning & AI Teaching ───────────────────────────────────────────
function submitCorrection(customLabel = null) {
    const label = (customLabel || (dom.correctInput ? dom.correctInput.value : "")).trim().toUpperCase();
    if (!label) {
        showToast("Please specify a gesture label to teach", "warning");
        return;
    }

    if (dom.correctStatus) {
        dom.correctStatus.style.display = "block";
        dom.correctStatus.textContent = `Retraining model on "${label}" in background...`;
        dom.correctStatus.style.color = "var(--amber)";
    }

    showToast(`Teaching AI gesture: "${label}"`, "info");

    fetch("/correct_gesture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: label })
    })
    .then(r => r.json())
    .then(data => {
        if (data.ok) {
            if (dom.correctStatus) {
                dom.correctStatus.textContent = data.message;
                dom.correctStatus.style.color = "var(--emerald)";
            }
            if (dom.correctInput) dom.correctInput.value = "";
            showToast(data.message, "success", 4500);
        } else {
            if (dom.correctStatus) {
                dom.correctStatus.textContent = "Error: " + data.message;
                dom.correctStatus.style.color = "var(--ruby)";
            }
            showToast(data.message, "error", 4000);
        }
        setTimeout(() => {
            if (dom.correctStatus) dom.correctStatus.style.display = "none";
        }, 6000);
    })
    .catch(e => {
        if (dom.correctStatus) {
            dom.correctStatus.textContent = "Request failed.";
            dom.correctStatus.style.color = "var(--ruby)";
        }
        showToast("Failed to connect to training worker", "error");
    });
}

function quickTeach(label) {
    if (dom.correctInput) dom.correctInput.value = label;
    submitCorrection(label);
}

// ── Gesture Reference Modal ─────────────────────────────────────────────────
function openGestureModal() {
    if (dom.gestureModal) {
        dom.gestureModal.style.display = "flex";
        dom.gestureModal.style.opacity = "1";
    }
}

function closeGestureModal() {
    if (dom.gestureModal) {
        dom.gestureModal.style.display = "none";
    }
}

function filterGestures(query) {
    const q = query.toUpperCase().trim();
    const items = document.querySelectorAll(".gesture-badge-item");
    items.forEach(item => {
        const text = item.getAttribute("data-label") || item.textContent;
        if (!q || text.toUpperCase().includes(q)) {
            item.style.display = "flex";
        } else {
            item.style.display = "none";
        }
    });
}

// ── Dominant Hand Switching ─────────────────────────────────────────────────
function setDominantHand(hand) {
    fetch("/set_dominant_hand", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hand: hand })
    })
    .then(r => r.json())
    .then(data => {
        if (data.ok) {
            updateDominantHandUI(data.dominant_hand);
            const handNames = { right: "Right Hand (Default)", left: "Left Hand", auto: "Auto Detection" };
            showToast(`Dominant hand set to ${handNames[data.dominant_hand] || data.dominant_hand}`, "info");
        }
    })
    .catch(e => console.error("Error setting dominant hand:", e));
}

function updateDominantHandUI(hand) {
    const btnRight = document.getElementById("btn-hand-right");
    const btnAuto = document.getElementById("btn-hand-auto");
    const btnLeft = document.getElementById("btn-hand-left");

    if (btnRight) btnRight.classList.toggle("active", hand === "right");
    if (btnAuto) btnAuto.classList.toggle("active", hand === "auto");
    if (btnLeft) btnLeft.classList.toggle("active", hand === "left");
}

let consecutivePollFailures = 0;

// ── Main Telemetry Polling Loop ─────────────────────────────────────────────
function poll() {
    fetch("/gesture")
        .then(r => r.json())
        .then(data => {
            consecutivePollFailures = 0;
            const active = !!data.active;

            if (dom.cameraFeed && (dom.cameraFeed.style.display === "none" || (dom.camFallback && dom.camFallback.style.display === "flex"))) {
                dom.cameraFeed.style.display = "block";
                if (dom.camFallback) dom.camFallback.style.display = "none";
                reconnectCamera(true);
            }

            if (data.status === "error") {
                setStatusBadge("error");
            } else {
                setStatusBadge(data.status || "running", active ? "Recording" : "Standby");
            }

            // Sync dominant hand UI
            if (data.dominant_hand) {
                updateDominantHandUI(data.dominant_hand);
            }

            // Hand detection separation indicators
            const detectedHand = data.detected_hand || "none";
            const handMap = {
                right: { label: "✋ Right Hand", cls: "hand-right", hud: "✋ RIGHT HAND" },
                left:  { label: "🤚 Left Hand",  cls: "hand-left",  hud: "🤚 LEFT HAND" },
                both:  { label: "🙌 Both Hands", cls: "hand-both",  hud: "🙌 BOTH HANDS" },
                none:  { label: "Standby",       cls: "hand-none",  hud: "STANDBY" }
            };
            const hCfg = handMap[detectedHand] || handMap.none;
            const handBadge = document.getElementById("hand-badge");
            if (handBadge) {
                handBadge.textContent = hCfg.label;
                handBadge.className = "hand-badge " + hCfg.cls;
            }
            const gesture = stripInternalSuffix(data.live_gesture || "—");
            const rSign = stripInternalSuffix(data.right_gesture || "—");
            const rConf = Math.round((data.right_conf || 0) * 100);
            const lSign = stripInternalSuffix(data.left_gesture || "—");
            const lConf = Math.round((data.left_conf || 0) * 100);

            const hudHandText = document.getElementById("hud-hand-text");
            if (hudHandText) {
                if (detectedHand === "both") {
                    hudHandText.textContent = `🙌 BOTH: ${gesture}`;
                } else if (detectedHand === "right") {
                    hudHandText.textContent = `✋ RIGHT: ${rSign} (${rConf}%)`;
                } else if (detectedHand === "left") {
                    hudHandText.textContent = `🤚 LEFT: ${lSign} (${lConf}%)`;
                } else {
                    hudHandText.textContent = "STANDBY";
                }
            }

            // Camera index sync
            if (data.camera_index !== undefined && data.camera_index !== currentCameraIndex) {
                currentCameraIndex = data.camera_index;
                if (dom.hudCamName) {
                    dom.hudCamName.textContent = `CAM ${data.camera_index} · LIVE FEED`;
                }
                if (dom.btnSwitchCam) {
                    dom.btnSwitchCam.title = `Switch Camera (Current: CAM ${data.camera_index}, Shortcut: X)`;
                }
            }

            // Dual-Hand Independent Monitor Update
            const valR = document.getElementById("val-right");
            const meterR = document.getElementById("meter-right");
            const pctR = document.getElementById("pct-right");
            const badgeR = document.getElementById("badge-right");
            const cardR = document.getElementById("track-right");

            const valL = document.getElementById("val-left");
            const meterL = document.getElementById("meter-left");
            const pctL = document.getElementById("pct-left");
            const badgeL = document.getElementById("badge-left");
            const cardL = document.getElementById("track-left");

            if (valR) valR.textContent = rSign;
            if (meterR) meterR.style.width = rConf + "%";
            if (pctR) pctR.textContent = rConf + "%";

            if (valL) valL.textContent = lSign;
            if (meterL) meterL.style.width = lConf + "%";
            if (pctL) pctL.textContent = lConf + "%";

            const isRightActive = (detectedHand === "right" || detectedHand === "both");
            const isLeftActive = (detectedHand === "left" || detectedHand === "both");

            if (badgeR) {
                badgeR.textContent = isRightActive ? (detectedHand === "both" ? "DUAL" : "ACTIVE") : (rSign !== "—" ? "READY" : "IDLE");
                badgeR.className = "hand-track-badge " + (isRightActive ? "badge-active" : (rSign !== "—" ? "badge-ready" : "badge-idle"));
            }
            if (cardR) cardR.classList.toggle("active-driver", isRightActive);

            if (badgeL) {
                badgeL.textContent = isLeftActive ? (detectedHand === "both" ? "DUAL" : "ACTIVE") : (lSign !== "—" ? "READY" : "IDLE");
                badgeL.className = "hand-track-badge " + (isLeftActive ? "badge-active" : (lSign !== "—" ? "badge-ready" : "badge-idle"));
            }
            if (cardL) cardL.classList.toggle("active-driver", isLeftActive);

            // Live gesture label
            if (dom.gestureLabel) {
                dom.gestureLabel.textContent = gesture;
            }

            // Live confidence
            const confVal = data.live_conf || 0;
            const pct = Math.round(confVal * 100);
            if (dom.confBar) {
                dom.confBar.style.width = pct + "%";
                if (pct >= 70) {
                    dom.confBar.style.background = "linear-gradient(90deg, var(--cyan), var(--emerald))";
                } else if (pct >= 40) {
                    dom.confBar.style.background = "linear-gradient(90deg, var(--cyan), var(--amber))";
                } else {
                    dom.confBar.style.background = "linear-gradient(90deg, var(--amber), var(--ruby))";
                }
            }
            if (dom.confText) dom.confText.textContent = pct + "%";
            if (dom.hudConfidence) dom.hudConfidence.textContent = `${pct}% CONF`;

            // Segmented stability buffer
            renderStability(data.buffer_fill || 0, data.buffer_max || 15, !!data.agreeing);

            // Recording state
            if (dom.recDot && dom.recLabel) {
                if (active) {
                    dom.recDot.className = "rec-dot on";
                    dom.recLabel.textContent = "Recording — sign STOP to pause";
                } else {
                    dom.recDot.className = "rec-dot";
                    dom.recLabel.textContent = "Paused — sign START to record";
                }
            }
            updateGestureRecUI(active);

            // ── Dynamic Gesture Watchdog Countdown Timer ──────────────────
            const dynName = data.dynamic_gesture;
            const dynTimeLeft = typeof data.dynamic_time_left === "number" ? data.dynamic_time_left : 0;
            const maxDyn = data.max_dynamic_duration || 2.8;

            const dynBanner = document.getElementById("dynamic-timer-banner");
            const dynTimerName = document.getElementById("dyn-timer-name");
            const dynTimerVal = document.getElementById("dyn-timer-val");
            const dynBarFill = document.getElementById("dyn-timer-bar-fill");

            const dynTelemetryRow = document.getElementById("dynamic-timer-telemetry");
            const dynTelemetryTime = document.getElementById("dyn-telemetry-time");
            const dynTelemetryBar = document.getElementById("dyn-telemetry-bar");

            if (dynName && dynTimeLeft > 0) {
                const ratio = Math.max(0, Math.min(100, (dynTimeLeft / maxDyn) * 100));
                const cleanDynTitle = stripInternalSuffix(dynName);

                if (dynBanner) {
                    dynBanner.style.display = "flex";
                    if (dynTimerName) dynTimerName.textContent = cleanDynTitle;
                    if (dynTimerVal) dynTimerVal.textContent = dynTimeLeft.toFixed(1) + "s";
                    if (dynBarFill) dynBarFill.style.width = ratio + "%";
                }
                if (dynTelemetryRow) {
                    dynTelemetryRow.style.display = "block";
                    if (dynTelemetryTime) dynTelemetryTime.textContent = `${dynTimeLeft.toFixed(1)}s (${cleanDynTitle})`;
                    if (dynTelemetryBar) dynTelemetryBar.style.width = ratio + "%";
                }
                const statePill = document.getElementById("state-pill");
                if (statePill) {
                    statePill.textContent = `DYNAMIC: ${cleanDynTitle} (${dynTimeLeft.toFixed(1)}s)`;
                    statePill.className = "state-pill dynamic-active";
                }
            } else {
                if (dynBanner) dynBanner.style.display = "none";
                if (dynTelemetryRow) dynTelemetryRow.style.display = "none";
                const statePill = document.getElementById("state-pill");
                if (statePill && statePill.classList.contains("dynamic-active")) {
                    statePill.textContent = active ? "RECORDING" : "STANDBY";
                    statePill.className = "state-pill " + (active ? "active" : "standby");
                }
            }

            // Sentence builder
            const word = data.word || "";
            if (dom.sentenceBox) {
                if (word.trim() === "") {
                    dom.sentenceBox.textContent = "(empty)";
                    dom.sentenceBox.className = "empty";
                } else {
                    dom.sentenceBox.textContent = word;
                    dom.sentenceBox.className = "";
                }
            }
            updateSentenceCounters(word);

            // Update Recent Signs Trail on confirmed stability
            if (data.agreeing && gesture && gesture !== "—" && gesture !== "NEUTRAL") {
                addRecentSign(gesture);
            }

            // Voice feedback for system alerts (e.g. "Recording start", "Recording stop")
            if (data.speak_alert) {
                if (data.speak_alert === "Recording stop") {
                    speakCompletedSentence(word);
                } else {
                    speakText(data.speak_alert);
                    showToast(data.speak_alert, "info", 2000);
                }
            }

            // Audio feedback when recording transitions from paused to active
            if (!lastActiveState && active) {
                if (data.speak_alert !== "Recording start") {
                    speakText("Recording start");
                }
            }

            // TTS Real-Time Triggers for translated words (only while recording is active)
            if (active && word !== lastSentenceText) {
                if (word.length > lastSentenceText.length) {
                    const newPart = word.slice(lastSentenceText.length).trim();
                    if (newPart && newPart !== "(empty)") {
                        speakText(newPart);
                        addRecentSign(newPart);
                    }
                }
                lastSentenceText = word;
            } else if (!active) {
                lastSentenceText = word;
            }

            // Speak completed sentence when paused from active
            if (lastActiveState && !active) {
                speakCompletedSentence(word);
                if (isRecordingVoice) {
                    stopVoiceRecording();
                }
            }
            lastActiveState = active;
        })
        .catch(err => {
            consecutivePollFailures++;
            if (consecutivePollFailures >= 5) {
                setStatusBadge("error");
                if (dom.cameraFeed) dom.cameraFeed.style.display = "none";
                if (dom.camFallback) dom.camFallback.style.display = "flex";
            }
        });
}

// ── Stop Server Action ──────────────────────────────────────────────────────
function stopServer() {
    if (!confirm("Stop the ASLO server? This releases the camera hardware and shuts down the Python process.")) {
        return;
    }
    if (dom.stopBtn) {
        dom.stopBtn.disabled = true;
        dom.stopBtn.textContent = "Shutting down…";
    }

    fetch("/shutdown", { method: "POST" }).catch(() => {});

    if (pollTimer) clearInterval(pollTimer);
    if (dom.cameraFeed) dom.cameraFeed.src = "";
    setStatusBadge("stopped");
    showToast("Server shutdown initiated.", "warning");
}

// ── Keyboard Shortcuts ──────────────────────────────────────────────────────
window.addEventListener("keydown", (e) => {
    // If typing inside an input field, do not trigger shortcuts
    if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
        if (e.key === "Enter" && document.activeElement === dom.correctInput) {
            submitCorrection();
        }
        return;
    }

    if (e.code === "Space") {
        e.preventDefault();
        toggleGestureRecording();
    } else if (e.key === "c" || e.key === "C") {
        clearSentence();
    } else if (e.key === "h" || e.key === "H") {
        toggleCamTelemetryHud();
    } else if (e.key === "v" || e.key === "V") {
        toggleVoiceRecording();
    } else if (e.key === "m" || e.key === "M") {
        toggleTTS();
    } else if (e.key === "s" || e.key === "S") {
        speakWholeSentence();
    } else if (e.key === "x" || e.key === "X") {
        switchCamera();
    } else if (e.key === "?") {
        openGestureModal();
    } else if (e.key === "Escape") {
        closeGestureModal();
    }
});

// ── Initialize Polling ──────────────────────────────────────────────────────
pollTimer = setInterval(poll, 120);
poll();
