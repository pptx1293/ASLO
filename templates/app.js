/**
 * ASLO — American Sign Language Online Studio Engine
 * Orchestrates real-time telemetry, Web Speech API (STT & TTS),
 * Active Learning model correction, camera HUD, and interactive controls.
 */

// ── DOM References ──────────────────────────────────────────────────────────
const dom = {
    // Camera & HUD
    webcam: document.getElementById("webcam"),
    outputCanvas: document.getElementById("output_canvas") || document.getElementById("landmark-canvas"),
    landmarkCanvas: document.getElementById("landmark-canvas"),
    cameraFeed: document.getElementById("webcam") || document.getElementById("camera-feed"),
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
    gestureSearch: document.getElementById("gesture-search"),

    // Voice Speech Canvas
    voiceSentenceBox: document.getElementById("voice-sentence-box"),
    voiceCharCount: document.getElementById("voice-char-count"),
    voiceWordCount: document.getElementById("voice-word-count")
};

// ── State Management ────────────────────────────────────────────────────────
let isTTSEnabled = true;
let isRecordingVoice = false;
let isGestureRecordingActive = false;
let lastActiveState = false;
let lastSentenceText = "";

// Client-Side Session State & Prediction Smoothing
let clientSentence = "";
let voiceSentence = "";
let clientLastWord = "";
let clientStabilityBuffer = [];
let clientLastTriggerTime = 0;
let clientDynamicGesture = null;
let clientDynamicStartTime = 0;
let clientDynamicNeutralStart = 0;
const CLIENT_DYNAMIC_MAX_SEC = 2.8;
let clientSuppressYouUntil = 0;
let clientDynamicCooldownUntil = 0;
let dynamicReadyForStart = true;
let clientLastDynamicCommitted = { gesture: "", time: 0 };
const REPEAT_DELAY_MS = 1200;
const STABILITY_REQUIRED_COUNT = 3; // N_Frame = 3 (fast, snappy recognition)
const BUFFER_MAX_LEN = 5;
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
let isMirrored = true;
let recentSigns = [];
let currentCameraIndex = 0;
let isSwitchingCamera = false;
let webcamStream = null;
let captureCanvas = null;
let captureCtx = null;
let isPredicting = false;
let isWebcamActive = false;
let lastFrameTime = 0;
let videoDeviceIds = [];
let currentDeviceIndex = 0;

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

// ── MediaPipe Hand Landmark Connections & Temporal Stabilizer ───────────────
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],          // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],          // Index
    [5, 9], [9, 10], [10, 11], [11, 12],     // Middle
    [9, 13], [13, 14], [14, 15], [15, 16],   // Ring
    [13, 17], [17, 18], [18, 19], [19, 20],  // Pinky
    [0, 17]                                  // Palm base
];

/**
 * 1-Euro Filter Implementation (Casiez et al., CHI 2012)
 * Calibrated for rock-solid hand landmark tracking:
 * - minCutoff = 1.2 Hz dampens high-frequency jitter during stationary hand poses.
 * - beta = 1.2 dynamically scales cutoff with motion velocity to eliminate lag during gesture movement.
 * - Sudden position teleports (> 0.45 normalized units) snap cleanly to prevent rubber-band stretching.
 */
class OneEuroFilter {
    constructor(minCutoff = 1.2, beta = 1.2, dCutoff = 1.0) {
        this.minCutoff = minCutoff; // Minimum cutoff frequency (Hz)
        this.beta = beta;           // Speed coefficient for dynamic responsiveness
        this.dCutoff = dCutoff;     // Derivative cutoff frequency (Hz)
        this.xPrev = null;
        this.dxPrev = 0;
        this.tPrev = null;
    }

    alpha(cutoff, dt) {
        const tau = 1.0 / (2.0 * Math.PI * cutoff);
        return 1.0 / (1.0 + tau / dt);
    }

    filter(x, timestamp = performance.now()) {
        if (this.tPrev === null || this.xPrev === null || !Number.isFinite(this.xPrev)) {
            this.xPrev = x;
            this.dxPrev = 0;
            this.tPrev = timestamp;
            return x;
        }

        const dt = Math.max((timestamp - this.tPrev) / 1000.0, 0.001);
        this.tPrev = timestamp;

        const rawDiff = x - this.xPrev;
        // Tracking re-acquisition snap: if point jumped over 45% of screen width/height, re-anchor cleanly
        if (Math.abs(rawDiff) > 0.45) {
            this.xPrev = x;
            this.dxPrev = 0;
            return x;
        }

        // Derivative (velocity) estimation
        const dx = rawDiff / dt;
        const alphaD = this.alpha(this.dCutoff, dt);
        const edx = alphaD * dx + (1.0 - alphaD) * this.dxPrev;
        this.dxPrev = edx;

        // Adaptive cutoff based on motion speed (high responsiveness when moving, heavy smoothing when still)
        const cutoff = this.minCutoff + this.beta * Math.abs(edx);
        const alpha = Math.min(1.0, Math.max(0.04, this.alpha(cutoff, dt)));

        const xHat = alpha * x + (1.0 - alpha) * this.xPrev;
        this.xPrev = xHat;
        return xHat;
    }

    reset() {
        this.xPrev = null;
        this.dxPrev = 0;
        this.tPrev = null;
    }
}

/**
 * Persistent Spatial Hand Tracker
 * Resolves hands by spatial continuity (wrist Euclidean proximity) rather than noisy handedness classification.
 * Prevents hand identity flipping, filter thrashing, and rubber-band crossover artifacts.
 */
class SpatialHandTracker {
    constructor(minCutoff = 1.2, beta = 1.2, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.tracks = [
            { id: "track_0", prevWrist: null, lastTime: 0, filters: this.createFilters() },
            { id: "track_1", prevWrist: null, lastTime: 0, filters: this.createFilters() }
        ];
    }

    createFilters() {
        const filters = [];
        for (let i = 0; i < 21; i++) {
            filters.push({
                x: new OneEuroFilter(this.minCutoff, this.beta, this.dCutoff),
                y: new OneEuroFilter(this.minCutoff, this.beta, this.dCutoff),
                z: new OneEuroFilter(this.minCutoff, this.beta, this.dCutoff)
            });
        }
        return filters;
    }

    resetTrack(t) {
        t.prevWrist = null;
        t.lastTime = 0;
        for (let i = 0; i < 21; i++) {
            t.filters[i].x.reset();
            t.filters[i].y.reset();
            t.filters[i].z.reset();
        }
    }

    resetAll() {
        this.resetTrack(this.tracks[0]);
        this.resetTrack(this.tracks[1]);
    }

    resetHand(handKey) {
        // Discard unneeded track without thrashing active tracking
        if (handKey === "Left") {
            this.resetTrack(this.tracks[1]);
        } else if (handKey === "Right") {
            this.resetTrack(this.tracks[0]);
        }
    }

    filterLandmarks(track, rawLandmarks, timestamp) {
        return rawLandmarks.map((p, i) => {
            const rx = Number(p.x !== undefined ? p.x : p[0]);
            const ry = Number(p.y !== undefined ? p.y : p[1]);
            const rz = Number((p.z !== undefined ? p.z : p[2]) || 0);

            const fx = track.filters[i].x.filter(rx, timestamp);
            const fy = track.filters[i].y.filter(ry, timestamp);
            const fz = track.filters[i].z.filter(rz, timestamp);

            return { x: fx, y: fy, z: fz };
        });
    }

    filterHand(handKey, rawLandmarks, timestamp = performance.now()) {
        const track = (handKey === "Left") ? this.tracks[1] : this.tracks[0];
        return this.filterLandmarks(track, rawLandmarks, timestamp);
    }

    process(multiHandLandmarks, multiHandedness, timestamp = performance.now()) {
        if (!multiHandLandmarks || multiHandLandmarks.length === 0) {
            for (const t of this.tracks) {
                if (timestamp - t.lastTime > 300) {
                    this.resetTrack(t);
                }
            }
            return [];
        }

        const numHands = multiHandLandmarks.length;
        const incoming = multiHandLandmarks.map((lms, idx) => {
            const w = lms[0];
            return {
                index: idx,
                rawLandmarks: lms,
                wrist: { x: Number(w.x !== undefined ? w.x : w[0]), y: Number(w.y !== undefined ? w.y : w[1]) },
                handedness: multiHandedness && multiHandedness[idx]
            };
        });

        const results = [];

        if (numHands === 1) {
            const hand = incoming[0];
            const d0 = this.tracks[0].prevWrist ? Math.hypot(hand.wrist.x - this.tracks[0].prevWrist.x, hand.wrist.y - this.tracks[0].prevWrist.y) : 999;
            const d1 = this.tracks[1].prevWrist ? Math.hypot(hand.wrist.x - this.tracks[1].prevWrist.x, hand.wrist.y - this.tracks[1].prevWrist.y) : 999;

            let chosenTrack = this.tracks[0];
            let otherTrack = this.tracks[1];
            if (d1 < d0) {
                chosenTrack = this.tracks[1];
                otherTrack = this.tracks[0];
            }

            // Only reset inactive track after 350ms grace period to survive momentary 1-frame drops
            if (timestamp - otherTrack.lastTime > 350) {
                this.resetTrack(otherTrack);
            }

            const smoothed = this.filterLandmarks(chosenTrack, hand.rawLandmarks, timestamp);
            chosenTrack.prevWrist = { x: smoothed[0].x, y: smoothed[0].y };
            chosenTrack.lastTime = timestamp;

            const handLabel = getActualPhysicalHand(hand.handedness);
            results.push({
                trackId: chosenTrack.id,
                label: handLabel,
                landmarks: smoothed
            });
        } else if (numHands >= 2) {
            const h0 = incoming[0];
            const h1 = incoming[1];

            let pair0 = h0;
            let pair1 = h1;

            const t0HasPrev = !!this.tracks[0].prevWrist;
            const t1HasPrev = !!this.tracks[1].prevWrist;

            if (t0HasPrev && t1HasPrev) {
                // Optimal 2x2 bipartite matching
                const costNormal = Math.hypot(h0.wrist.x - this.tracks[0].prevWrist.x, h0.wrist.y - this.tracks[0].prevWrist.y) +
                                   Math.hypot(h1.wrist.x - this.tracks[1].prevWrist.x, h1.wrist.y - this.tracks[1].prevWrist.y);
                const costSwap = Math.hypot(h1.wrist.x - this.tracks[0].prevWrist.x, h1.wrist.y - this.tracks[0].prevWrist.y) +
                                 Math.hypot(h0.wrist.x - this.tracks[1].prevWrist.x, h0.wrist.y - this.tracks[1].prevWrist.y);
                if (costSwap < costNormal) {
                    pair0 = h1;
                    pair1 = h0;
                }
            } else if (t0HasPrev && !t1HasPrev) {
                // Associate closer incoming hand to existing track 0; new hand anchors to track 1
                const d0 = Math.hypot(h0.wrist.x - this.tracks[0].prevWrist.x, h0.wrist.y - this.tracks[0].prevWrist.y);
                const d1 = Math.hypot(h1.wrist.x - this.tracks[0].prevWrist.x, h1.wrist.y - this.tracks[0].prevWrist.y);
                if (d1 < d0) {
                    pair0 = h1;
                    pair1 = h0;
                }
            } else if (!t0HasPrev && t1HasPrev) {
                // Associate closer incoming hand to existing track 1; new hand anchors to track 0
                const d0 = Math.hypot(h0.wrist.x - this.tracks[1].prevWrist.x, h0.wrist.y - this.tracks[1].prevWrist.y);
                const d1 = Math.hypot(h1.wrist.x - this.tracks[1].prevWrist.x, h1.wrist.y - this.tracks[1].prevWrist.y);
                if (d0 < d1) {
                    pair0 = h1;
                    pair1 = h0;
                }
            }

            const smoothed0 = this.filterLandmarks(this.tracks[0], pair0.rawLandmarks, timestamp);
            this.tracks[0].prevWrist = { x: smoothed0[0].x, y: smoothed0[0].y };
            this.tracks[0].lastTime = timestamp;

            const smoothed1 = this.filterLandmarks(this.tracks[1], pair1.rawLandmarks, timestamp);
            this.tracks[1].prevWrist = { x: smoothed1[0].x, y: smoothed1[0].y };
            this.tracks[1].lastTime = timestamp;

            let label0 = getActualPhysicalHand(pair0.handedness);
            let label1 = getActualPhysicalHand(pair1.handedness);
            if (label0 === label1) {
                // Disambiguate by spatial position: in unmirrored view, Right hand has smaller x
                label0 = (smoothed0[0].x <= smoothed1[0].x) ? "Right" : "Left";
                label1 = (label0 === "Right") ? "Left" : "Right";
            }

            results.push({ trackId: this.tracks[0].id, label: label0, landmarks: smoothed0 });
            results.push({ trackId: this.tracks[1].id, label: label1, landmarks: smoothed1 });
        }

        return results;
    }
}

const landmarkStabilizer = new SpatialHandTracker(1.2, 1.2, 1.0);
let currentStabilizedHands = [];

function getTemporalSmoothedLandmarks(handKey, rawPts, timestamp = performance.now()) {
    return landmarkStabilizer.filterHand(handKey, rawPts, timestamp);
}

function toLandmarkArray(landmarks) {
    if (!landmarks) return [];
    return landmarks.map(p => {
        if (Array.isArray(p)) return p;
        return [Number(p.x || 0), Number(p.y || 0), Number(p.z || 0)];
    });
}

function drawStabilizedSkeleton(ctx, landmarks) {
    if (!ctx || !landmarks || landmarks.length < 21) return;
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;

    // 1. Draw Stabilized White Skeleton Connection Lines
    if (typeof drawConnectors === "function" && typeof HAND_CONNECTIONS !== "undefined") {
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
            color: "#FFFFFF",
            lineWidth: 2.8
        });
    } else if (typeof HAND_CONNECTIONS !== "undefined") {
        ctx.save();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2.8;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        for (const [i, j] of HAND_CONNECTIONS) {
            const p1 = landmarks[i];
            const p2 = landmarks[j];
            if (p1 && p2) {
                ctx.beginPath();
                ctx.moveTo(p1.x * w, p1.y * h);
                ctx.lineTo(p2.x * w, p2.y * h);
                ctx.stroke();
            }
        }
        ctx.restore();
    }

    // 2. Draw Stabilized Red Landmark Points with Crisp Edges
    if (typeof drawLandmarks === "function") {
        drawLandmarks(ctx, landmarks, {
            color: "#FF0000",
            fillColor: "#FF0000",
            lineWidth: 1.0,
            radius: 3.5
        });
    } else {
        ctx.save();
        ctx.fillStyle = "#FF0000";
        for (const pt of landmarks) {
            if (pt) {
                ctx.beginPath();
                ctx.arc(pt.x * w, pt.y * h, 3.5, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
        ctx.restore();
    }
}

// ── Client-Side Webcam Lifecycle & Enumerate Devices ──────────────────────
async function enumerateCameras() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        const devices = await navigator.mediaDevices.enumerateDevices();
        videoDeviceIds = devices.filter(d => d.kind === "videoinput").map(d => d.deviceId);
    } catch (e) {
        console.warn("Could not enumerate camera devices:", e);
    }
}

// ── Client-Side MediaPipe Hands Detector & Zero-Latency Tracker ─────────────
let mpHands = null;
let mpCamera = null;
let lastPredictTime = 0;
const PREDICT_INTERVAL_MS = 100; // ~10 requests/sec debounce for backend inference

function initClientMediaPipe() {
    if (mpHands) return mpHands;
    if (typeof Hands === "undefined") {
        console.warn("MediaPipe Hands library not loaded yet.");
        return null;
    }

    try {
        mpHands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        // Configure client-side Hands detector:
        // Match collect_data.py confidence parameters (0.5 detection / 0.5 tracking)
        // for continuous, uninterrupted tracking without frame-to-frame dropouts
        mpHands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.6
        });

        const outputCanvas = dom.outputCanvas || dom.landmarkCanvas;
        const canvasCtx = outputCanvas ? outputCanvas.getContext("2d") : null;

        mpHands.onResults((results) => {
            if (!outputCanvas || !canvasCtx) return;

            // 1. Maintain canvas internal pixel dimensions to match incoming video frame (native stream resolution)
            if (results.image) {
                const w = results.image.width || 1280;
                const h = results.image.height || 720;
                if (outputCanvas.width !== w || outputCanvas.height !== h) {
                    outputCanvas.width = w;
                    outputCanvas.height = h;
                }

                canvasCtx.save();
                canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
                canvasCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);

                // 2. Real-time stabilized 60 FPS skeleton rendering via persistent spatial tracking
                const now = performance.now();
                const stabilizedList = landmarkStabilizer.process(
                    results.multiHandLandmarks,
                    results.multiHandedness,
                    now
                );

                for (let i = 0; i < stabilizedList.length; i++) {
                    drawStabilizedSkeleton(canvasCtx, stabilizedList[i].landmarks);
                }

                canvasCtx.restore();
                currentStabilizedHands = stabilizedList;
            }

            // 3. Throttled backend inference (~8-10 requests/sec, 100ms debounce)
            const now = performance.now();
            if (now - lastPredictTime >= PREDICT_INTERVAL_MS && !isPredicting) {
                lastPredictTime = now;
                sendLandmarksInference(results, currentStabilizedHands);
            }
        });

        return mpHands;
    } catch (e) {
        console.error("Failed to initialize MediaPipe Hands:", e);
        return null;
    }
}

let lastLeftHandToastTime = 0;

/**
 * Resolves the user's actual physical hand from MediaPipe's handedness metadata.
 * Note: When passing unmirrored webcam feed into MediaPipe, MediaPipe's classification
 * model outputs inverted labels ("Left" for user's Right hand, "Right" for user's Left hand).
 */
function getActualPhysicalHand(handednessObj) {
    if (!handednessObj) return "Right";
    const raw = (handednessObj.label || (handednessObj.classification && handednessObj.classification[0] && handednessObj.classification[0].label) || "").trim();
    if (raw === "Left") return "Right";
    if (raw === "Right") return "Left";
    return raw || "Right";
}

function updateHandStatusUI(status) {
    const handBadge = document.getElementById("hand-badge");
    const hudHandText = document.getElementById("hud-hand-text");
    const statePill = document.getElementById("state-pill");

    if (status === "left_ignored") {
        if (handBadge) {
            handBadge.textContent = "🤚 Hand: LEFT (IGNORED)";
            handBadge.className = "hand-badge hand-warning";
        }
        if (hudHandText) hudHandText.textContent = "🤚 LEFT HAND (UNSUPPORTED)";
        if (statePill) {
            statePill.textContent = "LEFT HAND (UNSUPPORTED)";
            statePill.className = "state-pill standby";
        }
        if (dom.gestureLabel) dom.gestureLabel.textContent = "—";
        const curSign = document.getElementById("current-sign");
        if (curSign) curSign.textContent = "—";
        if (dom.confBar) dom.confBar.style.width = "0%";
        if (dom.confText) dom.confText.textContent = "0%";
        if (dom.hudConfidence) dom.hudConfidence.textContent = "0% CONF";
    } else if (status === "right") {
        if (handBadge) {
            handBadge.textContent = "✋ Hand: RIGHT";
            handBadge.className = "hand-badge hand-right";
        }
        if (hudHandText) hudHandText.textContent = "✋ RIGHT HAND";
    } else if (status === "both") {
        if (handBadge) {
            handBadge.textContent = "🙌 Hand: BOTH";
            handBadge.className = "hand-badge hand-both";
        }
        if (hudHandText) hudHandText.textContent = "🙌 BOTH HANDS";
    } else {
        if (handBadge) {
            handBadge.textContent = "Standby";
            handBadge.className = "hand-badge hand-none";
        }
        if (hudHandText) hudHandText.textContent = "STANDBY";
    }
}

// ── Lightweight Landmark JSON Transmission ──────────────────────────────────
async function sendLandmarksInference(results, stabilizedHands = []) {
    if (isPredicting) return;

    const numHands = (results && results.multiHandLandmarks) ? results.multiHandLandmarks.length : 0;

    // Case 0: 0 hands detected -> idle / standby (cleanly reset filter state)
    if (numHands === 0) {
        landmarkStabilizer.resetAll();
        updateHandStatusUI("none");
        return;
    }

    // Case 1: 1 hand detected -> check physical handedness
    if (numHands === 1) {
        const rawH = results.multiHandedness && results.multiHandedness[0];
        const physicalHand = getActualPhysicalHand(rawH);

        // If user is showing only their Left Hand: DO NOT send prediction requests!
        if (physicalHand === "Left") {
            landmarkStabilizer.resetHand("Left");
            updateHandStatusUI("left_ignored");
            const now = performance.now();
            if (now - lastLeftHandToastTime > 3500) {
                lastLeftHandToastTime = now;
                showToast("Please use your Right Hand for single-hand signs", "warning", 3500);
            }
            return;
        }

        // Single physical Right hand: Proceed and send tagged with hand_type: "right"
        updateHandStatusUI("right");
        isPredicting = true;
        try {
            const rawLandmarks = results.multiHandLandmarks[0];
            const smoothed = (stabilizedHands && stabilizedHands[0])
                ? stabilizedHands[0].landmarks
                : landmarkStabilizer.filterHand("Right", rawLandmarks);
            const landmarks = toLandmarkArray(smoothed);

            const payload = {
                hand_type: "right",
                handedness: "Right",
                landmarks: landmarks,
                is_mirrored: isMirrored
            };

            await sendInferenceRequest(payload);
        } catch (err) {
            // Suppress transient drops
        } finally {
            isPredicting = false;
        }
        return;
    }

    // Case 2: 2 hands detected -> verify one is Left and one is Right (or both present), send tagged with hand_type: "both"
    if (numHands >= 2) {
        isPredicting = true;
        try {
            let h0Smoothed, h1Smoothed, h0Hand, h1Hand;

            if (stabilizedHands && stabilizedHands.length >= 2) {
                h0Smoothed = stabilizedHands[0].landmarks;
                h1Smoothed = stabilizedHands[1].landmarks;
                h0Hand = stabilizedHands[0].label || "Right";
                h1Hand = stabilizedHands[1].label || "Left";
            } else {
                const hand0 = results.multiHandLandmarks[0];
                const hand1 = results.multiHandLandmarks[1];
                h0Hand = getActualPhysicalHand(results.multiHandedness && results.multiHandedness[0]);
                h1Hand = getActualPhysicalHand(results.multiHandedness && results.multiHandedness[1]);
                h0Smoothed = landmarkStabilizer.filterHand(h0Hand, hand0);
                h1Smoothed = landmarkStabilizer.filterHand(h1Hand, hand1);
            }

            // Phantom Duplicate Hand Guard:
            // Check Euclidean distance between detected wrists
            const w0 = h0Smoothed[0];
            const w1 = h1Smoothed[0];
            const x0 = w0.x !== undefined ? w0.x : w0[0];
            const y0 = w0.y !== undefined ? w0.y : w0[1];
            const x1 = w1.x !== undefined ? w1.x : w1[0];
            const y1 = w1.y !== undefined ? w1.y : w1[1];
            const wristDist = Math.hypot(x0 - x1, y0 - y1);

            // If wrists are in the same location (< 0.15), or same hand detected twice with dist < 0.22,
            // this is a single physical hand detected twice!
            if (wristDist < 0.15 || (h0Hand === h1Hand && wristDist < 0.22)) {
                const primaryHand = (h0Hand === "Right") ? h0Smoothed : ((h1Hand === "Right") ? h1Smoothed : h0Smoothed);
                const primaryLabel = (h0Hand === "Right" || h1Hand === "Right") ? "Right" : h0Hand;
                if (primaryLabel === "Left") {
                    landmarkStabilizer.resetHand("Left");
                    updateHandStatusUI("left_ignored");
                    return;
                }
                updateHandStatusUI("right");
                const landmarks = toLandmarkArray(primaryHand);
                const payload = {
                    hand_type: "right",
                    handedness: "Right",
                    landmarks: landmarks,
                    is_mirrored: isMirrored
                };
                await sendInferenceRequest(payload);
                return;
            }

            updateHandStatusUI("both");

            // Ensure one is Right and one is Left based on spatial coordinate
            if (h0Hand === h1Hand) {
                const avgX0 = h0Smoothed.reduce((acc, p) => acc + (p.x !== undefined ? p.x : p[0]), 0) / h0Smoothed.length;
                const avgX1 = h1Smoothed.reduce((acc, p) => acc + (p.x !== undefined ? p.x : p[0]), 0) / h1Smoothed.length;
                if (avgX0 <= avgX1) {
                    h0Hand = "Right";
                    h1Hand = "Left";
                } else {
                    h0Hand = "Left";
                    h1Hand = "Right";
                }
            }

            const h0Pts = toLandmarkArray(h0Smoothed);
            const h1Pts = toLandmarkArray(h1Smoothed);

            const handsList = [
                { label: h0Hand, points: h0Pts },
                { label: h1Hand, points: h1Pts }
            ];

            const payload = {
                hand_type: "both",
                handedness: "Both",
                landmarks: (h0Hand === "Right" ? h0Pts : h1Pts),
                hands: handsList,
                all_hands: handsList,
                is_mirrored: isMirrored
            };

            await sendInferenceRequest(payload);
        } catch (err) {
            // Suppress transient drops
        } finally {
            isPredicting = false;
        }
        return;
    }
}

async function sendInferenceRequest(payload) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    let response = null;
    try {
        response = await fetch("/predict_landmarks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: controller.signal
        });
        if (!response.ok && response.status === 404) {
            response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
        }
    } finally {
        clearTimeout(timeoutId);
    }

    if (response && response.ok) {
        const data = await response.json();
        applyTelemetry(data);
    }
}

async function startClientWebcam(deviceId = null) {
    try {
        initClientMediaPipe();

        const videoElement = dom.webcam;
        if (!videoElement) return;

        if (mpCamera) {
            try { await mpCamera.stop(); } catch (e) {}
            mpCamera = null;
        }
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
        }

        // Direct high-definition camera stream request
        // Preserves full native sensor resolution without lowering video dimensions
        const highResConstraints = {
            video: deviceId ? { deviceId: { exact: deviceId } } : {
                facingMode: "user",
                width: { ideal: 1920, min: 1280 },
                height: { ideal: 1080, min: 720 },
                frameRate: { ideal: 60, min: 30 }
            },
            audio: false
        };

        try {
            webcamStream = await navigator.mediaDevices.getUserMedia(highResConstraints);
        } catch (hdErr) {
            console.warn("HD constraints unavailable, falling back to standard userMedia constraints:", hdErr);
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" },
                audio: false
            });
        }

        videoElement.srcObject = webcamStream;
        await videoElement.play();

        // Mutex guard prevents concurrent mpHands.send calls from corrupting MediaPipe WASM memory
        let isProcessingFrame = false;
        const frameLoop = async () => {
            if (!isWebcamActive) return;
            if (mpHands && !isProcessingFrame && videoElement.readyState >= 2 && !videoElement.paused) {
                isProcessingFrame = true;
                try {
                    await mpHands.send({ image: videoElement });
                } catch (e) {
                    // Suppress transient frame drops
                } finally {
                    isProcessingFrame = false;
                }
            }
            if ("requestVideoFrameCallback" in videoElement) {
                videoElement.requestVideoFrameCallback(frameLoop);
            } else {
                requestAnimationFrame(frameLoop);
            }
        };

        if ("requestVideoFrameCallback" in videoElement) {
            videoElement.requestVideoFrameCallback(frameLoop);
        } else {
            requestAnimationFrame(frameLoop);
        }

        isWebcamActive = true;
        if (dom.camFallback) dom.camFallback.style.display = "none";
        setStatusBadge("running", isGestureRecordingActive ? "Recording" : "Standby");
        if (dom.hudCamName) dom.hudCamName.textContent = "CLIENT WEBCAM · 60 FPS";
        showToast("Webcam connected — Stable tracking active", "success", 2500);

        await enumerateCameras();
    } catch (err) {
        console.error("Camera access failed:", err);
        isWebcamActive = false;
        setStatusBadge("error");
        if (dom.camFallback) {
            dom.camFallback.style.display = "flex";
            const p = dom.camFallback.querySelector("p");
            if (p) p.textContent = "Camera access denied or busy: " + (err.message || err.name);
        }
        showToast("Camera access failed: " + (err.message || err.name), "error", 5000);
    }
}

function reconnectCamera(isSilent = false) {
    if (!isSilent) showToast("Restarting camera stream...", "info", 1500);
    startClientWebcam();
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
    const canvas = dom.outputCanvas || dom.landmarkCanvas;
    if (canvas) {
        canvas.style.transform = isMirrored ? "scaleX(-1)" : "scaleX(1)";
    }
    if (dom.webcam) {
        dom.webcam.style.transform = isMirrored ? "scaleX(-1)" : "scaleX(1)";
    }
    showToast(isMirrored ? "Camera feed mirrored" : "Camera feed normal", "info", 1500);
}

async function switchCamera() {
    if (isSwitchingCamera) return;
    isSwitchingCamera = true;

    if (dom.btnSwitchCam) {
        dom.btnSwitchCam.classList.add("is-switching");
        dom.btnSwitchCam.disabled = true;
    }

    try {
        await enumerateCameras();
        if (videoDeviceIds.length > 1) {
            currentDeviceIndex = (currentDeviceIndex + 1) % videoDeviceIds.length;
            const nextId = videoDeviceIds[currentDeviceIndex];
            await startClientWebcam(nextId);
            showToast(`Switched to Camera ${currentDeviceIndex + 1}`, "info", 2000);
        } else {
            showToast("Only 1 camera detected on this system", "info", 2000);
        }
    } catch (err) {
        console.error("Camera switch error:", err);
        showToast("Error switching camera", "error");
    } finally {
        setTimeout(() => {
            isSwitchingCamera = false;
            if (dom.btnSwitchCam) {
                dom.btnSwitchCam.classList.remove("is-switching");
                dom.btnSwitchCam.disabled = false;
            }
        }, 600);
    }
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

            appendVoiceTranscript(transcript);
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

function updateSentenceDOM() {
    if (dom.sentenceBox) {
        const trimmed = clientSentence.trim();
        if (!trimmed) {
            dom.sentenceBox.textContent = "(empty)";
            dom.sentenceBox.className = "empty";
        } else {
            dom.sentenceBox.textContent = clientSentence;
            dom.sentenceBox.className = "";
        }
    }
    updateSentenceCounters(clientSentence);
}

function appendSignToClientSentence(sign) {
    if (!sign || sign === "—" || sign.toUpperCase() === "NEUTRAL" || sign.toUpperCase() === "IDLE") return;
    const clean = stripInternalSuffix(sign).trim();
    if (!clean || clean.toUpperCase() === "NEUTRAL" || clean.toUpperCase() === "IDLE") return;

    const trimmedUpper = clientSentence.trim().toUpperCase();
    const cleanUpper = clean.toUpperCase();
    const now = Date.now();

    // Prevent duplicate consecutive multi-word phrases (e.g. "HOW ARE YOU HOW ARE YOU")
    const isDynamicPhrase = cleanUpper.includes(" ") || cleanUpper === "HOW ARE YOU" || cleanUpper === "NICE TO MEET YOU";
    if (isDynamicPhrase) {
        // Time-based lock: reject identical dynamic phrase if committed within last 4 seconds
        if (cleanUpper === clientLastDynamicCommitted.gesture && (now - clientLastDynamicCommitted.time < 4000)) {
            return;
        }
        // Sentence token check: reject if sentence already ends with this phrase
        const tokens = trimmedUpper.split(/\s+/).filter(Boolean);
        const cleanTokens = cleanUpper.split(/\s+/).filter(Boolean);
        if (tokens.length >= cleanTokens.length) {
            const lastNTokens = tokens.slice(-cleanTokens.length).join(" ");
            if (lastNTokens === cleanTokens.join(" ")) {
                return;
            }
        }
        clientLastDynamicCommitted = { gesture: cleanUpper, time: now };
    }

    if (clean === "SPACE") {
        if (clientSentence.length > 0 && !clientSentence.endsWith(" ")) {
            clientSentence += " ";
        }
    } else {
        if (clean.length > 1) {
            if (clientSentence.length > 0 && !clientSentence.endsWith(" ")) {
                clientSentence += " ";
            }
            clientSentence += clean + " ";
        } else {
            clientSentence += clean;
        }
    }
    updateSentenceDOM();
    addRecentSign(clean);
    speakText(clean);
}

function backspaceSentenceLocal() {
    if (!clientSentence) return;
    const trimmed = clientSentence.trimEnd();
    if (trimmed.length === 0) {
        clientSentence = "";
        updateSentenceDOM();
        return;
    }
    const lastSpace = trimmed.lastIndexOf(" ");
    if (lastSpace !== -1) {
        clientSentence = trimmed.substring(0, lastSpace + 1);
    } else {
        clientSentence = "";
    }
    updateSentenceDOM();
    showToast("Removed last sign", "info", 1500);
}

function clearSentence() {
    clientSentence = "";
    clientLastWord = "";
    clientStabilityBuffer = [];
    updateSentenceDOM();
    recentSigns = [];
    renderRecentSigns();
    showToast("Sentence cleared", "info", 2000);
}

// ── Voice Speech Canvas Helpers ─────────────────────────────────────────────
function updateVoiceCounters(text) {
    if (!text || text === '(Waiting for voice input… click "Voice Input" or press [V] to speak)') {
        if (dom.voiceCharCount) dom.voiceCharCount.textContent = "0 chars";
        if (dom.voiceWordCount) dom.voiceWordCount.textContent = "0 words";
        return;
    }
    const cleanText = text.trim();
    const chars = cleanText.length;
    const words = cleanText ? cleanText.split(/\s+/).length : 0;

    if (dom.voiceCharCount) dom.voiceCharCount.textContent = `${chars} char${chars === 1 ? '' : 's'}`;
    if (dom.voiceWordCount) dom.voiceWordCount.textContent = `${words} word${words === 1 ? '' : 's'}`;
}

function updateVoiceDOM() {
    if (dom.voiceSentenceBox) {
        const trimmed = voiceSentence.trim();
        if (!trimmed) {
            dom.voiceSentenceBox.textContent = '(Waiting for voice input… click "Voice Input" or press [V] to speak)';
            dom.voiceSentenceBox.className = "empty";
        } else {
            dom.voiceSentenceBox.textContent = voiceSentence;
            dom.voiceSentenceBox.className = "";
        }
    }
    updateVoiceCounters(voiceSentence);
}

function appendVoiceTranscript(transcript) {
    if (!transcript) return;
    const clean = transcript.trim();
    if (!clean) return;

    if (voiceSentence.length > 0 && !voiceSentence.endsWith(" ")) {
        voiceSentence += " ";
    }
    voiceSentence += clean + " ";
    updateVoiceDOM();
}

function clearVoiceSentence() {
    voiceSentence = "";
    updateVoiceDOM();
    showToast("Voice canvas cleared", "info", 2000);
}

function copyVoiceSentence() {
    const text = dom.voiceSentenceBox ? dom.voiceSentenceBox.textContent : "";
    if (!text || text.startsWith("(") || text === "(empty)") {
        showToast("Voice canvas is empty to copy", "warning");
        return;
    }

    navigator.clipboard.writeText(text).then(() => {
        showToast("Copied voice text to clipboard!", "success");
    }).catch(() => {
        showToast("Failed to copy voice text to clipboard", "error");
    });
}

function speakVoiceSentence() {
    const text = dom.voiceSentenceBox ? dom.voiceSentenceBox.textContent : "";
    if (text && !text.startsWith("(") && text !== "(empty)") {
        const wasEnabled = isTTSEnabled;
        isTTSEnabled = true;
        speakText(text);
        isTTSEnabled = wasEnabled;
        showToast("Speaking voice text...", "info", 2000);
    } else {
        showToast("Voice canvas is empty", "warning", 2000);
    }
}

// ── Gesture Recording Toggle ────────────────────────────────────────────────
function toggleGestureRecording() {
    isGestureRecordingActive = !isGestureRecordingActive;
    updateGestureRecUI(isGestureRecordingActive);
    if (isGestureRecordingActive) {
        speakText("Recording start");
        showToast("Recording started — System translating signs", "success");
    } else {
        speakCompletedSentence(clientSentence);
        showToast("Recording paused", "info");
    }
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

// ── Shared Telemetry & HUD State Processor ─────────────────────────────────
function applyTelemetry(data) {
    if (!data) return;
    consecutivePollFailures = 0;
    const active = !!data.active;

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
        right: { label: "✋ Hand: RIGHT", cls: "hand-right", hud: "✋ RIGHT HAND" },
        left:  { label: "🤚 Hand: LEFT (IGNORED)", cls: "hand-warning", hud: "🤚 LEFT HAND (UNSUPPORTED)" },
        left_ignored: { label: "🤚 Hand: LEFT (IGNORED)", cls: "hand-warning", hud: "🤚 LEFT HAND (UNSUPPORTED)" },
        both:  { label: "🙌 Hand: BOTH", cls: "hand-both", hud: "🙌 BOTH HANDS" },
        none:  { label: "Standby", cls: "hand-none", hud: "STANDBY" }
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

    // Toast feedback if server rejected an unsupported left-hand sign
    if (data.prediction === null && data.message && (detectedHand === "left" || detectedHand === "left_ignored")) {
        const now = performance.now();
        if (now - lastLeftHandToastTime > 3500) {
            lastLeftHandToastTime = now;
            showToast(data.message, "warning", 3500);
        }
    }

    const hudHandText = document.getElementById("hud-hand-text");
    if (hudHandText) {
        if (detectedHand === "both") {
            hudHandText.textContent = `🙌 BOTH: ${gesture}`;
        } else if (detectedHand === "right") {
            hudHandText.textContent = `✋ RIGHT: ${rSign} (${rConf}%)`;
        } else if (detectedHand === "left" || detectedHand === "left_ignored") {
            hudHandText.textContent = "🤚 LEFT HAND (UNSUPPORTED)";
        } else {
            hudHandText.textContent = "STANDBY";
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

    // Live gesture label (Control gestures must show in current sign box with ALL CAPITAL letters)
    const rawPredUpper = (data.raw_pred || data.prediction || "").toUpperCase().trim();
    const cleanPredUpper = stripInternalSuffix(data.prediction || "").toUpperCase().trim();
    const CONTROL_SET = new Set(["START", "STOP", "SPACE", "BACKSPACE", "BACK SPACE", "NEUTRAL", "IDLE"]);
    const DYNAMIC_GESTURES = new Set(["HOW ARE YOU", "NICE TO MEET YOU", "J", "Z"]);

    const TWO_HAND_REQUIRED = new Set(["START", "STOP"]);

    let displayGesture = gesture;
    if (CONTROL_SET.has(rawPredUpper) || CONTROL_SET.has(cleanPredUpper)) {
        const ctrl = CONTROL_SET.has(rawPredUpper) ? rawPredUpper : cleanPredUpper;
        const normCtrl = (ctrl === "BACK SPACE") ? "BACKSPACE" : ctrl;
        if (TWO_HAND_REQUIRED.has(normCtrl) && data.detected_hand !== "both") {
            displayGesture = "—";
        } else {
            displayGesture = normCtrl;
        }
    } else if (clientDynamicGesture) {
        displayGesture = `${clientDynamicGesture} (IN MOTION)`;
    } else if (rawPredUpper.endsWith("_END") || rawPredUpper.endsWith("_START") || DYNAMIC_GESTURES.has(cleanPredUpper)) {
        displayGesture = cleanPredUpper;
    }

    if (dom.gestureLabel) {
        dom.gestureLabel.textContent = displayGesture;
    }
    const currentSignEl = document.getElementById("current-sign");
    if (currentSignEl && currentSignEl !== dom.gestureLabel) {
        currentSignEl.textContent = displayGesture;
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
    const now = Date.now();
    const rawPred = (data.raw_pred || data.prediction || "").toUpperCase();
    const cleanPred = stripInternalSuffix(data.prediction).trim();

    const dynBanner = document.getElementById("dynamic-timer-banner");
    const dynTimerName = document.getElementById("dyn-timer-name");
    const dynTimerVal = document.getElementById("dyn-timer-val");
    const dynBarFill = document.getElementById("dyn-timer-bar-fill");
    const dynTelemetryRow = document.getElementById("dynamic-timer-telemetry");
    const dynTelemetryTime = document.getElementById("dyn-telemetry-time");
    const dynTelemetryBar = document.getElementById("dyn-telemetry-bar");
    const statePill = document.getElementById("state-pill");

    // Dynamic gesture in progress countdown
    if (clientDynamicGesture) {
        const elapsedSec = (now - clientDynamicStartTime) / 1000.0;
        const dynamicMaxSec = (clientDynamicGesture === "NICE TO MEET YOU") ? 4.2 : 3.0;
        const remainingSec = Math.max(0, dynamicMaxSec - elapsedSec);
        const ratio = Math.max(0, Math.min(100, (remainingSec / dynamicMaxSec) * 100));

        if (dynBanner) {
            dynBanner.style.display = "flex";
            if (dynTimerName) dynTimerName.textContent = clientDynamicGesture;
            if (dynTimerVal) dynTimerVal.textContent = remainingSec.toFixed(1) + "s";
            if (dynBarFill) dynBarFill.style.width = ratio + "%";
        }
        if (dynTelemetryRow) {
            dynTelemetryRow.style.display = "block";
            if (dynTelemetryTime) dynTelemetryTime.textContent = `${remainingSec.toFixed(1)}s (${clientDynamicGesture})`;
            if (dynTelemetryBar) dynTelemetryBar.style.width = ratio + "%";
        }
        if (statePill) {
            statePill.textContent = `DYNAMIC: ${clientDynamicGesture} (${remainingSec.toFixed(1)}s)`;
            statePill.className = "state-pill dynamic-active";
        }

        // Only abort dynamic gesture if user drops to neutral for more than 1.2s continuously
        const isMomentaryNeutral = (rawPred === "NEUTRAL" || data.is_neutral || !data.prediction || data.prediction === "—");
        if (isMomentaryNeutral) {
            if (!clientDynamicNeutralStart) {
                clientDynamicNeutralStart = now;
            } else if (now - clientDynamicNeutralStart > 1200) {
                clientDynamicGesture = null;
                clientDynamicNeutralStart = 0;
                if (dynBanner) dynBanner.style.display = "none";
                if (dynTelemetryRow) dynTelemetryRow.style.display = "none";
                if (statePill) {
                    statePill.textContent = isGestureRecordingActive ? "RECORDING" : "STANDBY";
                    statePill.className = "state-pill " + (isGestureRecordingActive ? "active" : "standby");
                }
                clientStabilityBuffer = [];
                renderStability(0, STABILITY_REQUIRED_COUNT, false);
                return;
            }
        } else {
            clientDynamicNeutralStart = 0;
        }

        // Check completion trigger: requires minimum elapsed movement time and true final pose
        let isEndSignal = false;
        if (clientDynamicGesture === "NICE TO MEET YOU") {
            // "Nice to meet you" has 3 parts:
            // 1. NICE (sliding palms, ~0.0s - 1.2s)
            // 2. MEET (hands together in middle, ~1.2s - 1.5s) -> Must NOT complete early here!
            // 3. True LAST FRAME (held at >= 1.5s): user performs end pose (Nice to meet you_END or YOU)
            const isPointingYou = (rawPredUpper === "YOU" || cleanPredUpper === "YOU" || (data.right_gesture && String(data.right_gesture).toUpperCase() === "YOU"));
            const isNiceEndPose = (rawPredUpper === "NICE TO MEET YOU_END" || rawPredUpper.endsWith("_END"));
            if ((elapsedSec >= 1.5 && isNiceEndPose) || (elapsedSec >= 1.2 && isPointingYou)) {
                isEndSignal = true;
            }
        } else if (clientDynamicGesture === "HOW ARE YOU") {
            const isPointingYou = (rawPredUpper === "YOU" || cleanPredUpper === "YOU" || (data.right_gesture && String(data.right_gesture).toUpperCase() === "YOU"));
            const isHowEndPose = (rawPredUpper === "HOW ARE YOU_END" || rawPredUpper.endsWith("_END"));
            if (elapsedSec >= 0.65 && (isPointingYou || isHowEndPose)) {
                isEndSignal = true;
            }
        } else {
            // Other dynamic gestures (e.g. J, Z)
            if (rawPredUpper.endsWith("_END") && elapsedSec >= 0.4) {
                isEndSignal = true;
            }
        }
        if (isEndSignal) {
            const completedGesture = clientDynamicGesture;
            clientDynamicGesture = null;
            clientDynamicNeutralStart = 0;
            clientDynamicCooldownUntil = now + REPEAT_DELAY_MS;
            clientSuppressYouUntil = now + REPEAT_DELAY_MS;
            clientLastWord = completedGesture;
            clientLastTriggerTime = now;
            if (dynBanner) dynBanner.style.display = "none";
            if (dynTelemetryRow) dynTelemetryRow.style.display = "none";
            if (statePill) {
                statePill.textContent = isGestureRecordingActive ? "RECORDING" : "STANDBY";
                statePill.className = "state-pill " + (isGestureRecordingActive ? "active" : "standby");
            }
            if (isGestureRecordingActive) {
                appendSignToClientSentence(completedGesture);
                showToast(`Dynamic Sign: ${completedGesture}`, "success");
            }
            clientStabilityBuffer = [];
            renderStability(0, STABILITY_REQUIRED_COUNT, false);
            return;
        }

        // Cancel if timeout reached without achieving the completion pose
        if (remainingSec <= 0) {
            clientDynamicGesture = null;
            clientDynamicNeutralStart = 0;
            if (dynBanner) dynBanner.style.display = "none";
            if (dynTelemetryRow) dynTelemetryRow.style.display = "none";
            if (statePill) {
                statePill.textContent = isGestureRecordingActive ? "RECORDING" : "STANDBY";
                statePill.className = "state-pill " + (isGestureRecordingActive ? "active" : "standby");
            }
            clientStabilityBuffer = [];
            renderStability(0, STABILITY_REQUIRED_COUNT, false);
            return;
        }
        return;
    } else {
        if (dynBanner) dynBanner.style.display = "none";
        if (dynTelemetryRow) dynTelemetryRow.style.display = "none";
        if (statePill && statePill.classList.contains("dynamic-active")) {
            statePill.textContent = isGestureRecordingActive ? "RECORDING" : "STANDBY";
            statePill.className = "state-pill " + (isGestureRecordingActive ? "active" : "standby");
        }
    }

    // Initiate dynamic gesture ONLY on a genuine _START frame with solid confidence
    // (Never on bare names or _END frames, preventing accidental initiation)
    const isStartCandidate = (now >= clientDynamicCooldownUntil) && (
        rawPredUpper.endsWith("_START")
    );
    if (isStartCandidate && confVal >= 0.40) {
        const candidate = cleanPredUpper;
        clientDynamicGesture = candidate;
        clientDynamicStartTime = now;
        clientDynamicNeutralStart = 0;
        const candidateMaxSec = (candidate === "NICE TO MEET YOU") ? 4.5 : 3.0;
        if (dynBanner) {
            dynBanner.style.display = "flex";
            if (dynTimerName) dynTimerName.textContent = candidate;
            if (dynTimerVal) dynTimerVal.textContent = candidateMaxSec.toFixed(1) + "s";
            if (dynBarFill) dynBarFill.style.width = "100%";
        }
        clientStabilityBuffer = [];
        return;
    }

    // Dynamic Frame Guard:
    // Any uninitiated _END or _START frame is an incomplete fragment that shouldn't be added as static
    if (rawPredUpper.endsWith("_END") || rawPredUpper.endsWith("_START") || DYNAMIC_GESTURES.has(cleanPredUpper)) {
        clientStabilityBuffer = [];
        renderStability(0, STABILITY_REQUIRED_COUNT, false);
        return;
    }

    // Neutral / Rest Pose Gate: Do not accumulate in stability buffer or commit to sentence
    const isNeutralOrIdle = (
        !data.prediction ||
        data.prediction === "—" ||
        data.is_neutral ||
        data.detected_hand === "none" ||
        (data.prediction && data.prediction.toUpperCase() === "NEUTRAL") ||
        (data.raw_pred && data.raw_pred.toUpperCase() === "NEUTRAL")
    );

    if (isNeutralOrIdle) {
        if (dom.gestureLabel) dom.gestureLabel.textContent = "NEUTRAL";
        if (dom.statePill) {
            dom.statePill.textContent = isGestureRecordingActive ? "RECORDING (RESTING)" : "STANDBY (RESTING)";
            dom.statePill.className = "state-pill standby";
        }
        if (clientDynamicGesture) {
            clientDynamicGesture = null;
            if (dynBanner) dynBanner.style.display = "none";
            if (dynTelemetryRow) dynTelemetryRow.style.display = "none";
        }
        clientStabilityBuffer = [];
        renderStability(0, STABILITY_REQUIRED_COUNT, false);
        return;
    }

    // Stability Buffer Smoothing
    clientStabilityBuffer.push(cleanPred);
    if (clientStabilityBuffer.length > BUFFER_MAX_LEN) clientStabilityBuffer.shift();

    const matchCount = clientStabilityBuffer.filter(p => p === cleanPred).length;
    const isStable = matchCount >= STABILITY_REQUIRED_COUNT;
    renderStability(matchCount, STABILITY_REQUIRED_COUNT, isStable);

    if (isStable && confVal >= 0.40) {
        const upper = cleanPred.toUpperCase();
        if (upper === "NEUTRAL" || upper === "IDLE" || upper === "—") {
            clientStabilityBuffer = [];
            return;
        }
        if (DYNAMIC_GESTURES.has(upper)) {
            clientStabilityBuffer = [];
            return;
        }
        if (upper === "START" || upper === "STOP") {
            if (data.detected_hand !== "both" || confVal < 0.55) {
                clientStabilityBuffer = [];
                return;
            }
            if (upper === "START") {
                if (!isGestureRecordingActive) {
                    isGestureRecordingActive = true;
                    updateGestureRecUI(true);
                    speakText("Recording start");
                    showToast("Recording started — START gesture detected", "success");
                }
                clientStabilityBuffer = [];
                return;
            }
            if (upper === "STOP") {
                if (isGestureRecordingActive) {
                    isGestureRecordingActive = false;
                    updateGestureRecUI(false);
                    speakCompletedSentence(clientSentence);
                    showToast("Recording paused — STOP gesture detected", "info");
                }
                clientStabilityBuffer = [];
                return;
            }
        }
        if (upper === "SPACE") {
            if (now - clientLastTriggerTime > REPEAT_DELAY_MS) {
                if (isGestureRecordingActive) appendSignToClientSentence("SPACE");
                clientLastTriggerTime = now;
            }
            clientStabilityBuffer = [];
            return;
        }
        if (upper === "BACKSPACE" || upper === "BACK SPACE") {
            if (now - clientLastTriggerTime > REPEAT_DELAY_MS) {
                backspaceSentenceLocal();
                clientLastTriggerTime = now;
            }
            clientStabilityBuffer = [];
            return;
        }

        if (upper === "YOU" && clientDynamicGesture) {
            const completedGesture = clientDynamicGesture;
            clientDynamicGesture = null;
            clientDynamicNeutralStart = 0;
            clientDynamicCooldownUntil = now + REPEAT_DELAY_MS;
            clientSuppressYouUntil = now + REPEAT_DELAY_MS;
            clientLastWord = completedGesture;
            clientLastTriggerTime = now;
            if (dynBanner) dynBanner.style.display = "none";
            if (dynTelemetryRow) dynTelemetryRow.style.display = "none";
            if (isGestureRecordingActive) {
                appendSignToClientSentence(completedGesture);
                showToast(`Dynamic Sign: ${completedGesture}`, "success");
            }
            clientStabilityBuffer = [];
            renderStability(0, STABILITY_REQUIRED_COUNT, false);
            return;
        }

        if (upper === "YOU" && now < clientSuppressYouUntil) {
            clientStabilityBuffer = [];
            return;
        }

        if (cleanPred !== clientLastWord || (now - clientLastTriggerTime > REPEAT_DELAY_MS)) {
            if (isGestureRecordingActive) {
                appendSignToClientSentence(cleanPred);
                showToast(`Sign: ${cleanPred}`, "info", 1500);
            }
            clientLastWord = cleanPred;
            clientLastTriggerTime = now;
            clientStabilityBuffer = [];
        }
    }
}

// ── Fallback Polling Loop (Only when webcam not active) ─────────────────────
function poll() {
    if (isWebcamActive) return;
    fetch("/gesture")
        .then(r => r.json())
        .then(data => {
            applyTelemetry(data);
        })
        .catch(err => {
            consecutivePollFailures++;
            if (consecutivePollFailures >= 5) {
                setStatusBadge("error");
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

// ── Initialize Client Webcam & Isolated UI ──────────────────────────────────
updateSentenceDOM();
updateVoiceDOM();
startClientWebcam();
