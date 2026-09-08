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
    btnFullscreen: document.getElementById("btn-fullscreen"),
    btnMirror: document.getElementById("btn-mirror"),

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

    // Audio & Speech
    sttBtn: document.getElementById("stt-btn"),
    sttBtnText: document.getElementById("stt-btn-text"),
    sttIndicator: document.getElementById("stt-indicator"),
    ttsToggleBtn: document.getElementById("tts-toggle-btn"),
    ttsToggleText: document.getElementById("tts-toggle-text"),
    btnSpeakSentence: document.getElementById("btn-speak-sentence"),

    // Sentence Canvas
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
let pollTimer = null;
let lastBufMax = -1;
let recognition = null;
let speakTimeout = null;
let isMirrored = false;

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

// ── Audio Engine Unlock (Web Audio Autoplay Policy) ──────────────────────────
function unlockAudio() {
    if (dom.welcomeModal) {
        dom.welcomeModal.style.opacity = "0";
        setTimeout(() => {
            dom.welcomeModal.style.display = "none";
        }, 300);
    }

    try {
        const u = new SpeechSynthesisUtterance("ASLO translation studio activated");
        u.volume = 0.6;
        u.rate = 1.05;
        window.speechSynthesis.speak(u);
        showToast("Audio engine online & speech synthesis unlocked", "success");
    } catch (e) {
        console.warn("SpeechSynthesis unlock error:", e);
    }
}

// ── Camera Feed Handlers & Viewfinder Controls ──────────────────────────────
if (dom.cameraFeed) {
    dom.cameraFeed.onerror = () => {
        dom.cameraFeed.style.display = "none";
        if (dom.camFallback) dom.camFallback.style.display = "flex";
        setStatusBadge("error");
        showToast("Video stream disconnected. Please verify webcam connection.", "error");
    };
}

function reconnectCamera() {
    if (!dom.cameraFeed) return;
    dom.cameraFeed.src = "/video_feed?t=" + Date.now();
    dom.cameraFeed.style.display = "block";
    if (dom.camFallback) dom.camFallback.style.display = "none";
    showToast("Reconnecting video feed...", "info");
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

// ── Speech-To-Text (STT) Integration ────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onstart = () => {
        isRecordingVoice = true;
        if (dom.sttBtn) dom.sttBtn.classList.add("recording");
        if (dom.sttBtnText) dom.sttBtnText.textContent = "Stop Voice";
        if (dom.sttIndicator) {
            dom.sttIndicator.innerHTML = `<span class="stt-dot active"></span> Listening for voice...`;
        }
        showToast("Microphone active — speak to transcribe", "info");
    };

    recognition.onresult = (event) => {
        const result = event.results[event.results.length - 1];
        if (result.isFinal) {
            const transcript = result[0].transcript.trim();
            if (transcript) {
                if (dom.sttIndicator) {
                    dom.sttIndicator.innerHTML = `<span class="stt-dot active"></span> Transcribed: "${transcript}"`;
                }
                appendVoiceText(transcript);
            }
        }
    };

    recognition.onerror = (event) => {
        console.error("Speech Recognition Error:", event.error);
        if (event.error === "not-allowed") {
            showToast("Microphone permission denied by browser", "error");
            stopVoiceRecordingLocally();
        } else {
            showToast(`Voice recognition error: ${event.error}`, "warning");
        }
    };

    recognition.onend = () => {
        if (isRecordingVoice) {
            try {
                recognition.start();
            } catch (e) {
                stopVoiceRecordingLocally();
            }
        } else {
            stopVoiceRecordingLocally();
        }
    };
} else {
    if (dom.sttBtn) {
        dom.sttBtn.disabled = true;
        dom.sttBtn.style.opacity = "0.5";
        dom.sttBtn.style.cursor = "not-allowed";
        dom.sttBtnText.textContent = "STT Unsupported";
    }
    if (dom.sttIndicator) {
        dom.sttIndicator.textContent = "Speech recognition unsupported on this browser";
    }
}

function toggleVoiceRecording() {
    if (!recognition) {
        showToast("Speech Recognition not supported on this browser engine", "warning");
        return;
    }
    if (isRecordingVoice) {
        isRecordingVoice = false;
        recognition.stop();
        stopVoiceRecordingLocally();
    } else {
        if (dom.sttIndicator) {
            dom.sttIndicator.innerHTML = `<span class="stt-dot active"></span> Requesting mic access...`;
        }
        recognition.start();
    }
}

function stopVoiceRecordingLocally() {
    isRecordingVoice = false;
    if (dom.sttBtn) dom.sttBtn.classList.remove("recording");
    if (dom.sttBtnText) dom.sttBtnText.textContent = "Voice Input";
    if (dom.sttIndicator) {
        dom.sttIndicator.innerHTML = `<span class="stt-dot"></span> Voice input standby`;
    }
}

function appendVoiceText(text) {
    fetch("/append_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(r => r.json())
    .then(data => {
        if (data.ok) {
            showToast(`Added: "${text}"`, "success", 2000);
        }
    })
    .catch(e => console.error("Error appending voice text:", e));
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

// ── Status Badge ────────────────────────────────────────────────────────────
function setStatusBadge(status) {
    if (!dom.statusBadge) return;
    dom.statusBadge.className = "badge " + status;
    const labels = {
        running: "Online",
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
            showToast(nextState ? "Gesture recording started" : "Gesture recording paused", nextState ? "success" : "info");
        }
    })
    .catch(e => console.error("Error toggling active state:", e));
}

function updateGestureRecUI(active) {
    isGestureRecordingActive = active;
    if (dom.recToggleBtn) {
        if (active) {
            dom.recToggleBtn.classList.add("active");
            dom.recToggleBtn.textContent = "Pause Recording";
        } else {
            dom.recToggleBtn.classList.remove("active");
            dom.recToggleBtn.textContent = "Start Recording";
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

// ── Main Telemetry Polling Loop ─────────────────────────────────────────────
function poll() {
    fetch("/gesture")
        .then(r => r.json())
        .then(data => {
            if (data.status === "error") {
                setStatusBadge("error");
            } else {
                setStatusBadge(data.status || "running");
            }

            // Live gesture label
            const gesture = data.live_gesture || "—";
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
            const active = !!data.active;
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

            // TTS Real-Time Triggers
            if (word !== lastSentenceText) {
                if (word.length > lastSentenceText.length) {
                    const newPart = word.slice(lastSentenceText.length).trim();
                    if (newPart && newPart !== "(empty)") {
                        speakText(newPart);
                    }
                }
                lastSentenceText = word;
            }

            // Speak completed sentence when paused from active
            if (lastActiveState && !active) {
                if (word && word !== "(empty)") {
                    speakText("Sentence completed: " + word);
                }
                if (isRecordingVoice) {
                    isRecordingVoice = false;
                    if (recognition) recognition.stop();
                }
            }
            lastActiveState = active;
        })
        .catch(err => {
            setStatusBadge("error");
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
    } else if (e.key === "v" || e.key === "V") {
        toggleVoiceRecording();
    } else if (e.key === "m" || e.key === "M") {
        toggleTTS();
    } else if (e.key === "s" || e.key === "S") {
        speakWholeSentence();
    } else if (e.key === "?") {
        openGestureModal();
    } else if (e.key === "Escape") {
        closeGestureModal();
    }
});

// ── Initialize Polling ──────────────────────────────────────────────────────
pollTimer = setInterval(poll, 120);
poll();
