/* ── ASLO Mobile Studio Controller (v2.0 Enhanced) ────────────────────────── */

const dom = {
    // Viewfinder
    camBox: document.getElementById("mobile-cam-box"),
    camImg: document.getElementById("mobile-cam-img"),
    camFallback: document.getElementById("m-cam-fallback"),
    recTag: document.getElementById("m-hud-rec-tag"),
    recTagText: document.getElementById("m-rec-status"),
    recToggleBtn: document.getElementById("btn-m-rec-toggle"),
    recBtnIcon: document.getElementById("m-rec-btn-icon"),
    recBtnText: document.getElementById("m-rec-btn-text"),
    
    // Dominant Hand Switcher
    handOptRight: document.getElementById("hand-opt-right"),
    handOptLeft: document.getElementById("hand-opt-left"),

    // HUD Telemetry
    motionPill: document.getElementById("m-hud-motion"),
    motionLabel: document.getElementById("m-motion-label"),
    stabDotsContainer: document.getElementById("m-stab-dots"),

    // Dynamic Watchdog Banner
    dynBanner: document.getElementById("m-dyn-banner"),
    dynTitle: document.getElementById("m-dyn-title"),
    dynTimer: document.getElementById("m-dyn-timer"),
    dynBarFill: document.getElementById("m-dyn-bar-fill"),

    // Current Sign Hero Card
    signVal: document.getElementById("m-sign-val"),
    signSub: document.getElementById("m-sign-sub"),
    confPill: document.getElementById("m-conf-pill"),
    handPill: document.getElementById("m-hand-pill"),

    // Transcript Box
    transcriptBox: document.getElementById("m-transcript-box"),
    charStats: document.getElementById("m-char-stats"),

    // Sound Toggle Icons
    soundBtn: document.getElementById("btn-sound-toggle"),
    soundIconOn: document.getElementById("icon-sound-on"),
    soundIconOff: document.getElementById("icon-sound-off"),

    // Modals & Toast
    toast: document.getElementById("m-toast"),
    dictModal: document.getElementById("m-dict-modal"),
    dictSearch: document.getElementById("m-dict-search")
};

let pollTimer = null;
let lastSentenceText = "";
let lastActiveState = false;
let camAutoRetryTimer = null;
let isSoundMuted = localStorage.getItem("aslo_mobile_sound_muted") === "true";
let isMirrored = false;
let stabilitySegmentsCount = 8;

// Initialize Stability Dots
function initStabilityDots() {
    if (!dom.stabDotsContainer) return;
    dom.stabDotsContainer.innerHTML = "";
    for (let i = 0; i < stabilitySegmentsCount; i++) {
        const dot = document.createElement("span");
        dot.className = "m-stab-dot";
        dom.stabDotsContainer.appendChild(dot);
    }
}
initStabilityDots();

// Update Sound Toggle UI
function updateSoundUI() {
    if (dom.soundBtn) {
        dom.soundBtn.classList.toggle("muted", isSoundMuted);
    }
    if (dom.soundIconOn && dom.soundIconOff) {
        dom.soundIconOn.style.display = isSoundMuted ? "none" : "block";
        dom.soundIconOff.style.display = isSoundMuted ? "block" : "none";
    }
}
updateSoundUI();

function toggleSound() {
    isSoundMuted = !isSoundMuted;
    localStorage.setItem("aslo_mobile_sound_muted", isSoundMuted);
    updateSoundUI();
    showToast(isSoundMuted ? "Voice readout muted" : "Voice readout enabled");
    triggerHaptic(20);
}

function triggerHaptic(duration = 30) {
    if ("vibrate" in navigator) {
        try {
            navigator.vibrate(duration);
        } catch (_) {}
    }
}

function stripInternalSuffix(str) {
    if (!str) return "—";
    return str.replace(/_(START|END)$/i, "").trim() || "—";
}

function showToast(msg, duration = 2400) {
    if (!dom.toast) return;
    dom.toast.textContent = msg;
    dom.toast.classList.add("show");
    setTimeout(() => {
        dom.toast.classList.remove("show");
    }, duration);
}

// ── Text-To-Speech Synthesis ────────────────────────────────────────────────
let lastStopSpokenTime = 0;
let speakTimeout = null;

function speakText(text) {
    if (isSoundMuted) return;
    if (!("speechSynthesis" in window) || !text || text === "(empty)" || text === "—") return;
    if (speakTimeout) clearTimeout(speakTimeout);
    window.speechSynthesis.cancel();
    speakTimeout = setTimeout(() => {
        const utt = new SpeechSynthesisUtterance(text);
        utt.rate = 1.05;
        utt.pitch = 1.0;
        window.speechSynthesis.speak(utt);
    }, 45);
}

function speakCompletedSentence(sentenceText) {
    if (isSoundMuted) return;
    const now = Date.now();
    if (now - lastStopSpokenTime < 1500) return;
    lastStopSpokenTime = now;

    if (!("speechSynthesis" in window)) return;
    if (speakTimeout) clearTimeout(speakTimeout);
    window.speechSynthesis.cancel();

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
            sentenceUtt.rate = 0.94;
            sentenceUtt.pitch = 1.0;
            window.speechSynthesis.speak(sentenceUtt);
            showToast(`Sentence: "${clean}"`, 3500);
        } else {
            showToast("Recording stopped");
        }
    }, 55);
}

// ── Real-Time Telemetry Polling ─────────────────────────────────────────────
function poll() {
    fetch("/gesture")
        .then(res => res.json())
        .then(data => {
            const active = !!data.active;
            const liveSign = stripInternalSuffix(data.live_gesture || "—");
            const confVal = data.live_conf || 0;
            const confPct = Math.round(confVal * 100);

            // 1. Camera Box & Recording HUD Status
            if (dom.camBox) {
                dom.camBox.classList.toggle("recording", active);
            }
            if (dom.recTagText) {
                dom.recTagText.textContent = active ? "REC" : "STANDBY";
            }
            if (dom.recToggleBtn) {
                dom.recToggleBtn.classList.toggle("recording", active);
            }
            if (dom.recBtnIcon && dom.recBtnText) {
                dom.recBtnIcon.textContent = active ? "⏸" : "⏺";
                dom.recBtnText.textContent = active ? "Stop Recording" : "Start Recording";
            }

            // 2. Dominant Hand Sync
            if (data.dominant_hand) {
                updateDominantHandUI(data.dominant_hand);
            }

            // 3. Motion Mode & Stability Gauge
            if (dom.motionPill && dom.motionLabel) {
                const isDyn = data.route_mode === "DYNAMIC";
                dom.motionPill.classList.toggle("dynamic", isDyn);
                const spd = typeof data.motion_speed === "number" ? ` (${data.motion_speed.toFixed(2)})` : "";
                dom.motionLabel.textContent = isDyn ? `⚡ DYNAMIC${spd}` : "STATIC";
            }

            if (dom.stabDotsContainer) {
                const bufferFill = typeof data.buffer_fill === "number" ? data.buffer_fill : 0;
                const bufferMax = typeof data.buffer_max === "number" ? data.buffer_max : 8;
                const activeDots = Math.min(stabilitySegmentsCount, Math.round((bufferFill / bufferMax) * stabilitySegmentsCount));
                
                const dots = dom.stabDotsContainer.children;
                for (let i = 0; i < dots.length; i++) {
                    dots[i].classList.toggle("filled", i < activeDots);
                }
            }

            // 4. Dynamic Gesture Watchdog Countdown Timer
            const dynName = data.dynamic_gesture;
            const dynTimeLeft = typeof data.dynamic_time_left === "number" ? data.dynamic_time_left : 0;
            const maxDyn = data.max_dynamic_duration || 2.8;

            if (dynName && dynTimeLeft > 0) {
                const ratio = Math.max(0, Math.min(100, (dynTimeLeft / maxDyn) * 100));
                const cleanDyn = stripInternalSuffix(dynName);

                if (dom.dynBanner) dom.dynBanner.style.display = "flex";
                if (dom.dynTitle) dom.dynTitle.textContent = cleanDyn;
                if (dom.dynTimer) dom.dynTimer.textContent = dynTimeLeft.toFixed(1) + "s";
                if (dom.dynBarFill) dom.dynBarFill.style.width = ratio + "%";
            } else {
                if (dom.dynBanner) dom.dynBanner.style.display = "none";
            }

            // 5. Current Sign Hero Card
            if (dom.signVal) {
                dom.signVal.textContent = liveSign;
            }
            if (dom.signSub) {
                dom.signSub.textContent = active ? "Live tracking" : "Paused";
            }

            if (dom.confPill) {
                dom.confPill.textContent = confPct + "%";
                dom.confPill.classList.remove("high", "mid", "low");
                if (confPct >= 70) {
                    dom.confPill.classList.add("high");
                } else if (confPct >= 40) {
                    dom.confPill.classList.add("mid");
                } else {
                    dom.confPill.classList.add("low");
                }
            }

            if (dom.handPill) {
                const handMap = {
                    right: "✋ Right Hand",
                    left: "🤚 Left Hand",
                    both: "🙌 Dual Hands",
                    none: "💤 Standby"
                };
                dom.handPill.textContent = handMap[data.detected_hand] || "💤 Standby";
            }

            // 6. Live Translated Sentence & Char Statistics
            const word = data.word || "";
            if (dom.transcriptBox) {
                const textElem = dom.transcriptBox.querySelector(".m-text-content");
                if (word.trim() === "") {
                    if (textElem) textElem.textContent = "(Waiting for signs… sign START to begin)";
                    dom.transcriptBox.classList.add("empty");
                } else {
                    if (textElem) textElem.textContent = word;
                    dom.transcriptBox.classList.remove("empty");
                }
            }

            if (dom.charStats) {
                const rawTrim = word.trim();
                const wordArr = rawTrim ? rawTrim.split(/\s+/) : [];
                const wordCount = wordArr.length;
                const charCount = word.length;
                dom.charStats.textContent = `${wordCount} words · ${charCount} chars`;
            }

            // 7. Backend Spoken Alerts (e.g. "Recording start", "Recording stop")
            if (data.speak_alert) {
                if (data.speak_alert === "Recording stop") {
                    speakCompletedSentence(word);
                } else {
                    speakText(data.speak_alert);
                }
                showToast(data.speak_alert);
            }

            // Audio feedback on recording start
            if (!lastActiveState && active) {
                if (data.speak_alert !== "Recording start") {
                    speakText("Recording start");
                }
            }

            // Speech triggers for new translated signs
            if (active && word !== lastSentenceText) {
                if (word.length > lastSentenceText.length) {
                    const newPart = word.slice(lastSentenceText.length).trim();
                    if (newPart && newPart !== "(empty)") {
                        speakText(newPart);
                    }
                }
                lastSentenceText = word;
            } else if (!active) {
                lastSentenceText = word;
            }

            // Speak completed sentence when recording stops
            if (lastActiveState && !active) {
                speakCompletedSentence(word);
            }

            lastActiveState = active;
        })
        .catch(err => {
            console.warn("[Mobile] Telemetry poll failed, retrying...", err);
        });
}

// ── Control Actions ─────────────────────────────────────────────────────────

// Toggle Recording
function toggleRecording() {
    triggerHaptic(40);
    fetch("/toggle_active", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    })
        .then(res => res.json())
        .then(data => {
            const isRec = !!data.active;
            showToast(isRec ? "Recording Started" : "Recording Paused");
            if (isRec) {
                speakText("Recording start");
            } else {
                const textElem = dom.transcriptBox ? dom.transcriptBox.querySelector(".m-text-content") : null;
                const currentText = textElem ? textElem.textContent : "";
                speakCompletedSentence(currentText);
            }
        })
        .catch(err => console.error(err));
}

// Clear Transcript
function clearTranscript() {
    triggerHaptic(30);
    fetch("/clear", { method: "POST" })
        .then(() => {
            showToast("Transcript cleared");
            if (dom.transcriptBox) {
                const textElem = dom.transcriptBox.querySelector(".m-text-content");
                if (textElem) textElem.textContent = "(Waiting for signs… sign START to begin)";
                dom.transcriptBox.classList.add("empty");
            }
            if (dom.charStats) {
                dom.charStats.textContent = "0 words · 0 chars";
            }
            lastSentenceText = "";
        })
        .catch(err => console.error(err));
}

// Backspace / Undo
function backspaceTranscript() {
    triggerHaptic(25);
    fetch("/backspace", { method: "POST" })
        .then(r => r.json())
        .then(data => {
            showToast("Removed last sign");
            const newWord = data.word || "";
            if (dom.transcriptBox) {
                const textElem = dom.transcriptBox.querySelector(".m-text-content");
                if (!newWord.trim()) {
                    if (textElem) textElem.textContent = "(Waiting for signs… sign START to begin)";
                    dom.transcriptBox.classList.add("empty");
                } else {
                    if (textElem) textElem.textContent = newWord;
                    dom.transcriptBox.classList.remove("empty");
                }
            }
            lastSentenceText = newWord;
        })
        .catch(err => console.error(err));
}

// Speak Transcript
function speakTranscript() {
    triggerHaptic(20);
    const textElem = dom.transcriptBox ? dom.transcriptBox.querySelector(".m-text-content") : null;
    const text = textElem ? textElem.textContent : "";
    if (!text || text.startsWith("(")) {
        showToast("No sentence to speak yet");
        return;
    }
    speakText(text);
    showToast("Speaking sentence…");
}

// Copy Transcript
function copyTranscript() {
    triggerHaptic(20);
    const textElem = dom.transcriptBox ? dom.transcriptBox.querySelector(".m-text-content") : null;
    const text = textElem ? textElem.textContent : "";
    if (!text || text.startsWith("(")) {
        showToast("Nothing to copy");
        return;
    }
    navigator.clipboard.writeText(text)
        .then(() => showToast("Copied to clipboard!"))
        .catch(() => showToast("Failed to copy"));
}

// Switch Camera Hardware
function switchCamera() {
    triggerHaptic(35);
    showToast("Switching camera…");
    fetch("/api/camera/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
    })
        .then(r => r.json())
        .then(data => {
            if (data.ok) {
                showToast(`Switched to Camera #${data.camera_index}`);
                reconnectCam();
            } else {
                showToast("Switch camera failed");
            }
        })
        .catch(() => showToast("Camera switch error"));
}

// Toggle Camera Mirror
function toggleCameraMirror() {
    triggerHaptic(20);
    isMirrored = !isMirrored;
    if (dom.camImg) {
        dom.camImg.style.transform = isMirrored ? "scaleX(-1)" : "none";
    }
    showToast(isMirrored ? "Feed mirrored" : "Feed normal");
}

// Dominant Hand Toggle
function setDominantHand(hand) {
    triggerHaptic(25);
    fetch("/set_dominant_hand", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hand: hand })
    })
        .then(r => r.json())
        .then(data => {
            if (data.ok) {
                updateDominantHandUI(data.dominant_hand);
                showToast(`Dominant hand: ${data.dominant_hand === "right" ? "Right Hand ✋" : "Left Hand 🤚"}`);
            }
        })
        .catch(err => console.error(err));
}

function updateDominantHandUI(hand) {
    if (dom.handOptRight && dom.handOptLeft) {
        dom.handOptRight.classList.toggle("active", hand === "right");
        dom.handOptLeft.classList.toggle("active", hand === "left");
    }
}

// Stream Reconnect & Recovery
function reconnectCam() {
    if (!dom.camImg) return;
    if (dom.camFallback) dom.camFallback.style.display = "none";
    dom.camImg.style.display = "block";
    dom.camImg.src = "/video_feed?t=" + Date.now();
}

if (dom.camImg) {
    dom.camImg.onerror = () => {
        console.warn("[Mobile Cam] Stream error, presenting fallback & retrying...");
        if (dom.camFallback) dom.camFallback.style.display = "flex";
        if (camAutoRetryTimer) clearTimeout(camAutoRetryTimer);
        camAutoRetryTimer = setTimeout(reconnectCam, 2000);
    };
    dom.camImg.onload = () => {
        if (dom.camFallback) dom.camFallback.style.display = "none";
        dom.camImg.style.display = "block";
    };
}

// ── Dictionary Bottom Sheet & Filter ────────────────────────────────────────
function openDictionary() {
    triggerHaptic(25);
    if (dom.dictModal) {
        dom.dictModal.classList.add("show");
        if (dom.dictSearch) {
            dom.dictSearch.value = "";
            filterDictionary("");
        }
    }
}

function closeDictionary() {
    if (dom.dictModal) dom.dictModal.classList.remove("show");
}

function filterDictionary(query) {
    const q = (query || "").toLowerCase().trim();
    const sections = document.querySelectorAll(".m-dict-section");

    sections.forEach(section => {
        const chips = section.querySelectorAll(".m-gesture-chip");
        let hasVisible = false;

        chips.forEach(chip => {
            const name = (chip.getAttribute("data-name") || chip.textContent).toLowerCase();
            const matches = !q || name.includes(q);
            chip.style.display = matches ? "block" : "none";
            if (matches) hasVisible = true;
        });

        section.style.display = hasVisible ? "block" : "none";
    });
}

// Attach tap feedback to gesture chips
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".m-gesture-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            triggerHaptic(20);
            const name = chip.getAttribute("data-name") || chip.textContent;
            showToast(`Sign: ${name.toUpperCase()}`);
        });
    });
});

// Initialize Polling
pollTimer = setInterval(poll, 120);
poll();
