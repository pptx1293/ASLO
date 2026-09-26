/* ── ASLO Mobile Studio Controller (v2.0 Enhanced) ────────────────────────── */

const dom = {
    // Viewfinder
    camBox: document.getElementById("mobile-cam-box"),
    webcam: document.getElementById("mobile-webcam"),
    landmarkCanvas: document.getElementById("mobile-landmark-canvas"),
    camImg: document.getElementById("mobile-webcam") || document.getElementById("mobile-cam-img"),
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
let isSoundMuted = localStorage.getItem("aslo_mobile_sound_muted") === "true";
let isMirrored = true;
let stabilitySegmentsCount = 8;
let webcamStream = null;
let captureCanvas = null;
let captureCtx = null;
let isPredicting = false;
let isWebcamActive = false;
let lastFrameTime = 0;
let currentFacingMode = "user";

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

// ── MediaPipe Hand Landmark Connections & Drawing ───────────────────────────
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
];

// Temporal smoothing cache for mobile
let mobileSmoothedHands = {};

function drawMobileHandLandmarks(landmarksList) {
    if (!dom.landmarkCanvas) return;
    const canvas = dom.landmarkCanvas;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
    }

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarksList || landmarksList.length === 0) {
        mobileSmoothedHands = {};
        return;
    }

    const currentKeys = new Set();

    landmarksList.forEach((hand, idx) => {
        const handKey = hand.label || `hand_${idx}`;
        currentKeys.add(handKey);
        const rawPts = hand.points;
        if (!rawPts || rawPts.length < 21) return;

        // Exponential smoothing
        if (!mobileSmoothedHands[handKey]) {
            mobileSmoothedHands[handKey] = rawPts.map(p => ({ x: p.x, y: p.y }));
        } else {
            const prev = mobileSmoothedHands[handKey];
            mobileSmoothedHands[handKey] = rawPts.map((p, i) => {
                const prevP = prev[i] || p;
                const dx = p.x - prevP.x;
                const dy = p.y - prevP.y;
                const distSq = dx * dx + dy * dy;
                const alpha = distSq > 0.002 ? 0.92 : 0.65;
                return { x: prevP.x + dx * alpha, y: prevP.y + dy * alpha };
            });
        }

        const pts = mobileSmoothedHands[handKey];

        // 1. Classic White Skeleton Connection Lines
        ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
        ctx.lineWidth = 2.4;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
        ctx.shadowBlur = 3;

        for (const [i, j] of HAND_CONNECTIONS) {
            const p1 = pts[i];
            const p2 = pts[j];
            if (!p1 || !p2) continue;
            ctx.beginPath();
            ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
            ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
            ctx.stroke();
        }

        // 2. Classic Red Landmark Points
        ctx.shadowBlur = 0;
        for (let i = 0; i < pts.length; i++) {
            const p = pts[i];
            const px = p.x * canvas.width;
            const py = p.y * canvas.height;
            const isFingertip = (i === 4 || i === 8 || i === 12 || i === 16 || i === 20);
            const radius = isFingertip ? 4.0 : 3.0;

            ctx.beginPath();
            ctx.arc(px, py, radius, 0, Math.PI * 2);
            ctx.fillStyle = "#FF0000";
            ctx.fill();

            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = 0.9;
            ctx.stroke();
        }
    });

    for (const k in mobileSmoothedHands) {
        if (!currentKeys.has(k)) {
            delete mobileSmoothedHands[k];
        }
    }
}

// ── Mobile Client-Side MediaPipe Hands Detector ─────────────────────────────
let mobileHands = null;
let mobileCamera = null;
let lastMobilePredictTime = 0;
const MOBILE_PREDICT_INTERVAL = 100; // ~10 req/sec

function initMobileMediaPipe() {
    if (mobileHands) return mobileHands;
    if (typeof Hands === "undefined") return null;

    try {
        mobileHands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        mobileHands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        mobileHands.onResults((results) => {
            const canvas = dom.landmarkCanvas;
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            const rect = canvas.getBoundingClientRect();
            if (canvas.width !== rect.width || canvas.height !== rect.height) {
                canvas.width = rect.width;
                canvas.height = rect.height;
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 1. Instant zero-latency skeleton rendering
            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
                for (const landmarks of results.multiHandLandmarks) {
                    if (typeof drawConnectors === "function" && typeof HAND_CONNECTIONS !== "undefined") {
                        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                            color: "#FFFFFF",
                            lineWidth: 2.4
                        });
                    }
                    if (typeof drawLandmarks === "function") {
                        drawLandmarks(ctx, landmarks, {
                            color: "#FF0000",
                            fillColor: "#FF0000",
                            lineWidth: 1.0,
                            radius: 3.5
                        });
                    }
                }
            }

            // 2. Throttled backend landmark prediction
            const now = performance.now();
            if (now - lastMobilePredictTime >= MOBILE_PREDICT_INTERVAL && !isPredicting) {
                lastMobilePredictTime = now;
                sendMobileLandmarks(results);
            }
        });

        return mobileHands;
    } catch (e) {
        console.error("Mobile MediaPipe init error:", e);
        return null;
    }
}

async function sendMobileLandmarks(results) {
    if (isPredicting) return;
    isPredicting = true;

    try {
        const hasHands = results && results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
        let payload = null;

        if (hasHands) {
            const primaryHand = results.multiHandLandmarks[0];
            const landmarks = primaryHand.map(pt => [
                Number(pt.x.toFixed(5)),
                Number(pt.y.toFixed(5)),
                Number((pt.z || 0).toFixed(5))
            ]);

            let handedness = "Right";
            if (results.multiHandedness && results.multiHandedness.length > 0) {
                handedness = results.multiHandedness[0].label || "Right";
            }

            const allHands = results.multiHandLandmarks.map((hand, idx) => {
                const label = (results.multiHandedness && results.multiHandedness[idx])
                    ? results.multiHandedness[idx].label
                    : (idx === 0 ? handedness : "Left");
                return {
                    label: label,
                    points: hand.map(pt => [
                        Number(pt.x.toFixed(5)),
                        Number(pt.y.toFixed(5)),
                        Number((pt.z || 0).toFixed(5))
                    ])
                };
            });

            payload = {
                landmarks: landmarks,
                handedness: handedness,
                all_hands: allHands,
                is_mirrored: isMirrored
            };
        } else {
            payload = {
                landmarks: [],
                all_hands: [],
                is_mirrored: isMirrored
            };
        }

        const response = await fetch("/predict_landmarks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const data = await response.json();
            applyMobileTelemetry(data);
        }
    } catch (err) {
        // Tolerated transient frame drops
    } finally {
        isPredicting = false;
    }
}

// ── Mobile Client Webcam Lifecycle ──────────────────────────────────────────
async function startMobileWebcam() {
    try {
        initMobileMediaPipe();

        if (mobileCamera) {
            try { await mobileCamera.stop(); } catch (e) {}
            mobileCamera = null;
        }
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
        }

        const constraints = {
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 480 },
                height: { ideal: 360 }
            },
            audio: false
        };

        if (typeof Camera !== "undefined" && mobileHands) {
            mobileCamera = new Camera(dom.webcam, {
                onFrame: async () => {
                    if (isWebcamActive && mobileHands) {
                        await mobileHands.send({ image: dom.webcam });
                    }
                },
                width: 480,
                height: 360
            });
            await mobileCamera.start();
        } else {
            webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
            if (dom.webcam) {
                dom.webcam.srcObject = webcamStream;
                dom.webcam.style.transform = isMirrored ? "scaleX(-1)" : "none";
                await dom.webcam.play();
                dom.webcam.style.display = "block";
            }
            const frameLoop = async () => {
                if (isWebcamActive && mobileHands) {
                    await mobileHands.send({ image: dom.webcam });
                    requestAnimationFrame(frameLoop);
                }
            };
            requestAnimationFrame(frameLoop);
        }

        isWebcamActive = true;
        if (dom.camFallback) dom.camFallback.style.display = "none";
        showToast("Mobile camera active");
    } catch (err) {
        console.error("[Mobile] Camera error:", err);
        isWebcamActive = false;
        if (dom.webcam) dom.webcam.style.display = "none";
        if (dom.camFallback) {
            dom.camFallback.style.display = "flex";
            const t = dom.camFallback.querySelector(".m-fallback-text");
            if (t) t.textContent = "Camera access denied or busy";
        }
        showToast("Camera error: " + (err.message || err.name));
    }
}

// ── Real-Time Telemetry State Processor ────────────────────────────────────
function applyMobileTelemetry(data) {
    if (!data) return;
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
}

// ── Fallback Polling (When webcam stream is idle) ───────────────────────────
function poll() {
    if (isWebcamActive) return;
    fetch("/gesture")
        .then(res => res.json())
        .then(data => {
            applyMobileTelemetry(data);
        })
        .catch(() => {});
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

// Switch Front / Back Camera
async function switchCamera() {
    triggerHaptic(35);
    currentFacingMode = (currentFacingMode === "user") ? "environment" : "user";
    isMirrored = (currentFacingMode === "user");
    showToast(`Switched to ${currentFacingMode === "user" ? "Front" : "Rear"} camera`);
    await startMobileWebcam();
}

// Toggle Camera Mirror
function toggleCameraMirror() {
    triggerHaptic(20);
    isMirrored = !isMirrored;
    if (dom.webcam) {
        dom.webcam.style.transform = isMirrored ? "scaleX(-1)" : "none";
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
    startMobileWebcam();
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

// Initialize Client Webcam & Fallback Polling
startMobileWebcam();
pollTimer = setInterval(poll, 2000);
