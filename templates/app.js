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
let clientDisplayHistory = [];
let clientSmoothedDisplayGesture = "—";
let clientSmoothedConf = 0;

// ── Dual-Host API Resolver & Engine Auto-Negotiation ────────────────────────
const API_BASE_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? '' 
    : window.location.origin;

let currentInferenceUrl = API_BASE_URL;
let serverMode = localStorage.getItem("aslo_server_mode") || "auto";
let isLocalAlive = false;

async function checkLocalServerAlive() {
    try {
        const controller = new AbortController();
        const tid = setTimeout(() => controller.abort(), 600);
        const res = await fetch("http://127.0.0.1:5000/ping", {
            method: "GET",
            mode: "cors",
            signal: controller.signal
        });
        clearTimeout(tid);
        if (res.ok) {
            const data = await res.json();
            return Boolean(data && data.status === "online");
        }
    } catch (_) {}
    return false;
}

async function updateServerNegotiation() {
    isLocalAlive = await checkLocalServerAlive();
    const select = document.getElementById("server-mode-select");
    const pill = document.getElementById("server-status-pill");
    const downloadLink = document.getElementById("local-engine-download-link");

    if (select && select.value !== serverMode) {
        select.value = serverMode;
    }

    if (serverMode === "local") {
        if (isLocalAlive) {
            currentInferenceUrl = "http://127.0.0.1:5000";
            if (pill) {
                pill.textContent = "Local (0ms Active)";
                pill.style.background = "#10b981";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "none";
        } else {
            currentInferenceUrl = API_BASE_URL;
            if (pill) {
                pill.textContent = "Local Offline (Cloud Fallback)";
                pill.style.background = "#ef4444";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "inline";
        }
    } else if (serverMode === "cloud") {
        currentInferenceUrl = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? '' : window.location.origin;
        if (pill) {
            pill.textContent = "Cloud Active";
            pill.style.background = "#0284c7";
            pill.style.color = "#fff";
        }
        if (downloadLink) downloadLink.style.display = "none";
    } else {
        // "auto" mode
        if (isLocalAlive) {
            currentInferenceUrl = "http://127.0.0.1:5000";
            if (pill) {
                pill.textContent = "Auto: Local Engine (0ms)";
                pill.style.background = "#10b981";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "none";
        } else {
            currentInferenceUrl = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? '' : window.location.origin;
            if (pill) {
                pill.textContent = "Auto: Cloud Active";
                pill.style.background = "#6366f1";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "none";
        }
    }
}

function initServerNegotiator() {
    const select = document.getElementById("server-mode-select");
    if (select) {
        select.value = serverMode;
        select.addEventListener("change", (e) => {
            serverMode = e.target.value;
            localStorage.setItem("aslo_server_mode", serverMode);
            updateServerNegotiation();
        });
    }
    updateServerNegotiation();
    setInterval(updateServerNegotiation, 10000);
}

// ── State Hysteresis & Latching Constants & State ──────────────────────────
let activeSign = null;
let candidateSign = null;
let candidateCount = 0;
const COMMIT_THRESHOLD = 3; // Must win >= 3 out of 4 consecutive frames
const CONFIRM_BUFFER_SIZE = 4;
const CONFIRM_WIN_COUNT = 3;
let confirmationBuffer = [];
const HYSTERESIS_LOCK_MS = 350; // 350ms hysteresis lock against adjacent sign overriding
let lockUntil = 0; // Cooldown timestamp

// Optional Local Edge Inference Fallback State
let edgeModel = null;
let isEdgeModelReady = false;

const N_FRAME = 5;
const N_FRAMES = 5;
const PREDICTION_WINDOW_MAX = 5;
const MAJORITY_VOTE_RATIO = 0.60;
const GESTURE_LOCK_HOLD_MS = 120;

let predictionWindow = [];
let lastConfirmedGesture = "—";
let confirmedGestureTime = 0;
let hasPassedThroughNeutral = false;
let currentPipelineState = "Detecting...";

function getSmoothedDisplayGesture(newDisplay, conf = 0) {
    if (newDisplay && newDisplay.includes("(IN MOTION)")) {
        return newDisplay;
    }

    const now = Date.now();
    if (now < lockUntil && activeSign !== null) {
        return activeSign;
    }

    if (activeSign !== null) {
        return activeSign;
    }

    if (currentPipelineState === "Stable" && lastConfirmedGesture && lastConfirmedGesture !== "—") {
        return lastConfirmedGesture;
    }

    // Maintain the last stable translated word on screen during transitions instead of flickering or disappearing
    if (lastConfirmedGesture && lastConfirmedGesture !== "—") {
        return lastConfirmedGesture;
    }

    return (newDisplay && newDisplay !== "NEUTRAL") ? newDisplay : "—";
}

let clientDynamicGesture = null;
let clientDynamicStartTime = 0;
let clientDynamicNeutralStart = 0;
const CLIENT_DYNAMIC_MAX_SEC = 2.8;
let clientSuppressYouUntil = 0;
let clientDynamicCooldownUntil = 0;
let dynamicReadyForStart = true;
let clientLastDynamicCommitted = { gesture: "", time: 0 };
const REPEAT_DELAY_MS = 140; // Sub-150ms repeat cooldown
const STABILITY_REQUIRED_COUNT = 3; // 3 out of 5 frames
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
let isRequestPending = false;
let isPredicting = false;
let isWebcamActive = false;
let lastFrameTime = 0;
let videoDeviceIds = [];
let currentDeviceIndex = 0;

/**
 * Single Closest-Hand Proximity Filter & Primary User Isolation
 * 1. Calculate scale = dist(wrist_0, middle_mcp_9).
 * 2. If multiple hands exist, discard all hands whose scale is < 70% of largest hand.
 * 3. Only process the hand closest to the center (0.2 < x < 0.8).
 * 4. Verify physical Right hand before dispatching single-hand signs.
 */
function isolatePrimaryUserHands(multiHandLandmarks, multiHandedness) {
    if (!multiHandLandmarks || multiHandLandmarks.length === 0) {
        return { hands: [], status: "none" };
    }

    const candidates = multiHandLandmarks.map((lms, idx) => {
        const w0 = lms[0];
        const m9 = lms[9];
        const wx = Number(w0.x !== undefined ? w0.x : w0[0]);
        const wy = Number(w0.y !== undefined ? w0.y : w0[1]);
        const mx = Number(m9.x !== undefined ? m9.x : m9[0]);
        const my = Number(m9.y !== undefined ? m9.y : m9[1]);

        // scale = dist(wrist_0, middle_mcp_9)
        const scale = Math.hypot(wx - mx, wy - my);
        const rawH = multiHandedness && multiHandedness[idx];
        const physicalHand = getActualPhysicalHand(rawH);
        const distFromCenter = Math.abs(wx - 0.5);
        const inCenterRegion = (wx >= 0.20 && wx <= 0.80);

        return {
            index: idx,
            rawLandmarks: lms,
            handedness: rawH,
            physicalHand: physicalHand,
            scale: scale,
            wrist: { x: wx, y: wy },
            distFromCenter: distFromCenter,
            inCenterRegion: inCenterRegion
        };
    });

    let maxScale = 0;
    for (const c of candidates) {
        if (c.scale > maxScale) maxScale = c.scale;
    }

    // Discard all hands whose scale is < 65% of the largest hand
    const valid = candidates.filter(c => c.scale >= 0.65 * maxScale);
    if (valid.length === 0) {
        return { hands: [], status: "aligning" };
    }

    // Keep only hands in the central region (0.2 < x < 0.8) if available
    const centerHands = valid.filter(c => c.inCenterRegion);
    const pool = centerHands.length > 0 ? centerHands : valid;

    // Pick hand closest to center
    pool.sort((a, b) => a.distFromCenter - b.distFromCenter);

    const primary = pool[0];

    // Single hand passed filter:
    if (pool.length === 1 || valid.length === 1) {
        return {
            hands: [primary],
            status: primary.inCenterRegion ? "isolated" : "aligning"
        };
    }

    // Two or more hands: search for a valid pair belonging to the SAME primary user
    let pairedSecond = null;
    for (let j = 1; j < pool.length; j++) {
        const cand = pool[j];

        const dy = Math.abs(primary.wrist.y - cand.wrist.y);
        const dx = Math.abs(primary.wrist.x - cand.wrist.x);
        const wristDist = Math.hypot(primary.wrist.x - cand.wrist.x, primary.wrist.y - cand.wrist.y);

        if (dy >= 0.40 || dx >= 0.65) continue;
        if ((primary.wrist.x < 0.15 && cand.wrist.x > 0.85) || (cand.wrist.x < 0.15 && primary.wrist.x > 0.85)) continue;
        if (wristDist < 0.04) continue; // Only discard exact identical duplicate detections

        // Handedness pairing: opposite or spatial assignment if MediaPipe misclassified identical handedness
        let isOpposite = (
            (primary.physicalHand === "Right" && cand.physicalHand === "Left") ||
            (primary.physicalHand === "Left" && cand.physicalHand === "Right")
        );
        if (!isOpposite) {
            // Assign opposite based on horizontal position so touching/overlapping hands in START/STOP pair reliably
            if (cand.wrist.x < primary.wrist.x) {
                cand.physicalHand = isMirrored ? "Right" : "Left";
                primary.physicalHand = isMirrored ? "Left" : "Right";
            } else {
                cand.physicalHand = isMirrored ? "Left" : "Right";
                primary.physicalHand = isMirrored ? "Right" : "Left";
            }
        }

        pairedSecond = cand;
        break;
    }

    if (pairedSecond) {
        const pair = primary.physicalHand === "Right" ? [primary, pairedSecond] : [pairedSecond, primary];
        return { hands: pair, status: "isolated" };
    }

    return { hands: [primary], status: primary.inCenterRegion ? "isolated" : "aligning" };
}

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
 * Pre-Inference Landmark Exponential Moving Average (EMA: alpha ~ 0.65)
 * Suppresses frame-level tracking jitter before landmark feature extraction.
 */
class LandmarkEMASmoother {
    constructor(alpha = 0.65) {
        this.alpha = alpha;
        this.history = { Right: null, Left: null };
        this.lastTime = { Right: 0, Left: 0 };
    }

    smooth(handKey, landmarks, timestamp = performance.now()) {
        if (!landmarks || landmarks.length === 0) return landmarks;
        const key = (handKey === "Left") ? "Left" : "Right";
        const prev = this.history[key];
        const dt = timestamp - (this.lastTime[key] || 0);

        if (!prev || prev.length !== landmarks.length || dt > 350) {
            this.history[key] = landmarks.map(p => ({
                x: Number(p.x !== undefined ? p.x : p[0]),
                y: Number(p.y !== undefined ? p.y : p[1]),
                z: Number((p.z !== undefined ? p.z : p[2]) || 0)
            }));
            this.lastTime[key] = timestamp;
            return this.history[key];
        }

        const a = this.alpha;
        const smoothed = landmarks.map((p, i) => {
            const rx = Number(p.x !== undefined ? p.x : p[0]);
            const ry = Number(p.y !== undefined ? p.y : p[1]);
            const rz = Number((p.z !== undefined ? p.z : p[2]) || 0);
            return {
                x: a * rx + (1.0 - a) * prev[i].x,
                y: a * ry + (1.0 - a) * prev[i].y,
                z: a * rz + (1.0 - a) * prev[i].z
            };
        });

        this.history[key] = smoothed;
        this.lastTime[key] = timestamp;
        return smoothed;
    }

    reset(handKey) {
        if (handKey) {
            this.history[handKey] = null;
            this.lastTime[handKey] = 0;
        } else {
            this.history.Right = null;
            this.history.Left = null;
            this.lastTime.Right = 0;
            this.lastTime.Left = 0;
        }
    }
}

const landmarkEMASmoother = new LandmarkEMASmoother(0.65);
let predictSequenceId = 0;
let lastHandledSequenceId = 0;

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
        t.prevLandmarks = null;
        t.velocities = null;
        t.coastFrames = 0;
        t.lastLabel = null;
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
        const dt = track.lastTime > 0 ? (timestamp - track.lastTime) : 16;
        const smoothed = rawLandmarks.map((p, i) => {
            const rx = Number(p.x !== undefined ? p.x : p[0]);
            const ry = Number(p.y !== undefined ? p.y : p[1]);
            const rz = Number((p.z !== undefined ? p.z : p[2]) || 0);

            const fx = track.filters[i].x.filter(rx, timestamp);
            const fy = track.filters[i].y.filter(ry, timestamp);
            const fz = track.filters[i].z.filter(rz, timestamp);

            return { x: fx, y: fy, z: fz };
        });

        if (track.prevLandmarks && dt > 4) {
            const dtSec = Math.max(0.005, dt / 1000.0);
            track.velocities = smoothed.map((p, i) => ({
                vx: (p.x - track.prevLandmarks[i].x) / dtSec,
                vy: (p.y - track.prevLandmarks[i].y) / dtSec,
                vz: (p.z - track.prevLandmarks[i].z) / dtSec
            }));
        }
        track.prevLandmarks = smoothed;
        track.coastFrames = 0;
        return smoothed;
    }

    filterHand(handKey, rawLandmarks, timestamp = performance.now()) {
        const track = (handKey === "Left") ? this.tracks[1] : this.tracks[0];
        return this.filterLandmarks(track, rawLandmarks, timestamp);
    }

    process(multiHandLandmarks, multiHandedness, timestamp = performance.now()) {
        if (!multiHandLandmarks || multiHandLandmarks.length === 0) {
            const coastedResults = [];
            for (const t of this.tracks) {
                const elapsed = timestamp - t.lastTime;
                // Temporal coasting for 1-2 dropped frames within 100ms
                if (t.prevLandmarks && t.velocities && t.coastFrames < 2 && elapsed <= 100) {
                    t.coastFrames++;
                    const dtSec = Math.max(0.016, elapsed / 1000.0);
                    const projected = t.prevLandmarks.map((p, i) => ({
                        x: p.x + (t.velocities[i].vx || 0) * dtSec * 0.7,
                        y: p.y + (t.velocities[i].vy || 0) * dtSec * 0.7,
                        z: p.z + (t.velocities[i].vz || 0) * dtSec * 0.7
                    }));
                    t.prevLandmarks = projected;
                    coastedResults.push({
                        trackId: t.id,
                        label: t.lastLabel || "Right",
                        landmarks: projected,
                        isCoasted: true
                    });
                } else if (elapsed > 100) {
                    this.resetTrack(t);
                }
            }
            if (coastedResults.length > 0) {
                return coastedResults;
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
            chosenTrack.lastLabel = handLabel;
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

            this.tracks[0].lastLabel = label0;
            this.tracks[1].lastLabel = label1;

            results.push({ trackId: this.tracks[0].id, label: label0, landmarks: smoothed0 });
            results.push({ trackId: this.tracks[1].id, label: label1, landmarks: smoothed1 });
        }

        return results;
    }
}

/**
 * Trajectory-Based State Machine for Dynamic "J"
 * Tracks pinky tip (ID 20) over a sliding buffer (15-25 frames).
 * 1. Trigger Phase: starting pose is "I" and downward vertical velocity vy > 0.
 * 2. Active Motion Tracking: suppresses static single-frame predictions ("Y", "O", "You", "I") from updating UI.
 * 3. Path Verification: downward descent (dy >= 0.045) + upward recovery hook (dy >= 0.02) within 350ms-1000ms.
 * 4. Commit: commits "J" with high confidence (1.0).
 */
class JTrajectoryTracker {
    constructor() {
        this.buffer = [];
        this.state = "IDLE"; // "IDLE" | "SWOOPING"
        this.startTime = 0;
        this.lastTriggerTime = 0;
        this.isMoving = false;
    }

    isStartingIPose(landmarks) {
        if (!landmarks || landmarks.length < 21) return false;
        const p20 = landmarks[20];
        const p18 = landmarks[18];
        const p17 = landmarks[17];
        const w0 = landmarks[0];
        const m9 = landmarks[9];
        const scale = Math.hypot(w0.x - m9.x, w0.y - m9.y) + 1e-6;

        const pinkyUp = (p20.y < p18.y) && (p20.y < p17.y - 0.03 * scale);
        const fingersCurled = (landmarks[8].y > landmarks[6].y - 0.03) &&
                              (landmarks[12].y > landmarks[10].y - 0.03) &&
                              (landmarks[16].y > landmarks[14].y - 0.03);

        const d45 = Math.hypot(landmarks[4].x - landmarks[5].x, landmarks[4].y - landmarks[5].y);
        const flareRatio = d45 / scale;

        return pinkyUp && (flareRatio < 0.38 || fingersCurled);
    }

    update(landmarks, timestamp = performance.now()) {
        if (!landmarks || landmarks.length < 21) {
            if (this.state === "SWOOPING" && timestamp - this.startTime > 1000) {
                this.reset();
            }
            return { state: this.state, committed: false, suppressStatic: (this.state === "SWOOPING") };
        }

        const p20 = landmarks[20];
        const isI = this.isStartingIPose(landmarks);

        this.buffer.push({
            x: p20.x,
            y: p20.y,
            z: p20.z || 0,
            t: timestamp,
            isI: isI
        });

        if (this.buffer.length > 25) {
            this.buffer.shift();
        }

        if (this.buffer.length < 5) {
            return { state: this.state, committed: false, suppressStatic: (this.state === "SWOOPING") };
        }

        const now = timestamp;
        const cur = this.buffer[this.buffer.length - 1];
        const prev5 = this.buffer[Math.max(0, this.buffer.length - 5)];
        const dtSec = Math.max(0.01, (cur.t - prev5.t) / 1000.0);
        const vy = (cur.y - prev5.y) / dtSec;

        const recentI = this.buffer.slice(-6).some(b => b.isI);
        if (this.state === "IDLE") {
            if (recentI && vy > 0.12 && (now - this.lastTriggerTime > 1200)) {
                this.state = "SWOOPING";
                this.startTime = now;
                this.isMoving = true;
            }
        }

        if (this.state === "SWOOPING") {
            const duration = now - this.startTime;
            if (duration > 1100) {
                this.reset();
                return { state: "IDLE", committed: false, suppressStatic: false };
            }

            const motionFrames = this.buffer.filter(b => b.t >= this.startTime - 120);
            if (motionFrames.length >= 7 && duration >= 300) {
                const minY = Math.min(...motionFrames.map(b => b.y));
                const maxY = Math.max(...motionFrames.map(b => b.y));
                const lastY = motionFrames[motionFrames.length - 1].y;

                const downwardTravel = maxY - minY;
                const upwardRecovery = maxY - lastY;

                if (downwardTravel >= 0.045 && upwardRecovery >= 0.02) {
                    this.lastTriggerTime = now;
                    this.reset();
                    return { state: "COMMITTED", committed: true, suppressStatic: true, gesture: "J", confidence: 1.0 };
                }
            }

            return { state: "SWOOPING", committed: false, suppressStatic: true };
        }

        return { state: "IDLE", committed: false, suppressStatic: false };
    }

    reset() {
        this.state = "IDLE";
        this.startTime = 0;
        this.isMoving = false;
        this.buffer = [];
    }
}

const jTrajectoryTracker = new JTrajectoryTracker();

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
const PREDICT_INTERVAL_MS = 70; // 12-15 Hz (every ~70 ms)
let currentMinDetectionConfidence = 0.55;
let droppedHandFramesCount = 0;

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

        // Decoupled Dual-Threshold Confidence:
        // Detection 0.55 / Tracking 0.65 for adverse angle & low-light stability
        mpHands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.55,
            minTrackingConfidence: 0.65
        });

        const outputCanvas = dom.outputCanvas || dom.landmarkCanvas;
        const canvasCtx = outputCanvas ? outputCanvas.getContext("2d") : null;

        mpHands.onResults((results) => {
            if (!outputCanvas || !canvasCtx) return;

            // Tracking Fallback: if drops for > 4 frames, drop detection confidence to 0.45 for rapid re-acquisition
            const hasRawHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
            if (!hasRawHands) {
                droppedHandFramesCount++;
                if (droppedHandFramesCount > 4 && currentMinDetectionConfidence !== 0.45) {
                    currentMinDetectionConfidence = 0.45;
                    mpHands.setOptions({ minDetectionConfidence: 0.45, minTrackingConfidence: 0.65 });
                }
            } else {
                droppedHandFramesCount = 0;
                if (currentMinDetectionConfidence !== 0.55) {
                    currentMinDetectionConfidence = 0.55;
                    mpHands.setOptions({ minDetectionConfidence: 0.55, minTrackingConfidence: 0.65 });
                }
            }

            // 1. Single-person hand isolation before skeleton rendering and inference
            const isolation = isolatePrimaryUserHands(results.multiHandLandmarks, results.multiHandedness);

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

                // 2. Real-time stabilized 60 FPS skeleton rendering for isolated user hands only
                const now = performance.now();
                const isolatedLandmarks = isolation.hands.map(h => h.rawLandmarks);
                const isolatedHandedness = isolation.hands.map(h => h.handedness);
                const stabilizedList = landmarkStabilizer.process(
                    isolatedLandmarks,
                    isolatedHandedness,
                    now
                );

                for (let i = 0; i < stabilizedList.length; i++) {
                    drawStabilizedSkeleton(canvasCtx, stabilizedList[i].landmarks);
                }

                canvasCtx.restore();
                currentStabilizedHands = stabilizedList;
            }

            // Trajectory-Based State Machine for Dynamic "J"
            const now = performance.now();
            if (currentStabilizedHands.length === 1 && isolation.hands.length === 1 && isolation.hands[0].physicalHand === "Right") {
                const jResult = jTrajectoryTracker.update(currentStabilizedHands[0].landmarks, now);
                if (jResult.committed) {
                    activeSign = "J";
                    lastConfirmedGesture = "J";
                    lockUntil = Date.now() + HYSTERESIS_LOCK_MS;
                    updatePipelineStatusUI("Stable");
                    if (isGestureRecordingActive) {
                        appendSignToClientSentence("J");
                    }
                    addRecentSign("J");
                    showToast("Dynamic Sign: J", "success");
                    applyTelemetry({
                        ok: true,
                        prediction: "J",
                        live_gesture: "J",
                        live_conf: 1.0,
                        confidence: 1.0,
                        detected_hand: "right",
                        status: "running"
                    });
                    return;
                }
                if (jResult.suppressStatic) {
                    updatePipelineStatusUI("DYNAMIC: J (IN MOTION)");
                    return; // Suppress static single-frame inference updates during J swoop
                }
            } else if (currentStabilizedHands.length === 0) {
                jTrajectoryTracker.reset();
            }

            // 3. Throttled backend inference (~14 Hz, non-blocking lock)
            if (now - lastPredictTime >= PREDICT_INTERVAL_MS) {
                if (isRequestPending || isPredicting) return;
                lastPredictTime = now;
                sendLandmarksInference(results, currentStabilizedHands, isolation);
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

function updatePipelineStatusUI(status) {
    currentPipelineState = status;
    const statePill = document.getElementById("state-pill");
    if (!statePill) return;

    statePill.textContent = status;
    statePill.className = "state-pill";

    if (status === "Stable") {
        statePill.classList.add("stable");
    } else if (status === "Ambiguous") {
        statePill.classList.add("ambiguous");
    } else if (status === "Aligning User...") {
        statePill.classList.add("aligning");
    } else if (status === "Detecting...") {
        statePill.classList.add("detecting");
    } else if (status.startsWith("DYNAMIC")) {
        statePill.classList.add("dynamic-active");
    } else {
        statePill.classList.add("standby");
    }
}

function updateHandStatusUI(status) {
    const handBadge = document.getElementById("hand-badge");
    const hudHandText = document.getElementById("hud-hand-text");

    if (status === "left_ignored") {
        if (handBadge) {
            handBadge.textContent = "🤚 Hand: LEFT (IGNORED)";
            handBadge.className = "hand-badge hand-warning";
        }
        if (hudHandText) hudHandText.textContent = "🤚 LEFT HAND (UNSUPPORTED)";
        updatePipelineStatusUI("Aligning User...");
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
async function sendLandmarksInference(results, stabilizedHands = [], precomputedIsolation = null) {
    if (isRequestPending || isPredicting) return;

    const isolation = precomputedIsolation || isolatePrimaryUserHands(
        results && results.multiHandLandmarks,
        results && results.multiHandedness
    );

    const numHands = isolation.hands.length;

    // Case 0: 0 hands detected -> idle / standby (cleanly reset filter state)
    if (numHands === 0) {
        landmarkStabilizer.resetAll();
        landmarkEMASmoother.reset();
        updateHandStatusUI("none");
        updatePipelineStatusUI("Detecting...");
        return;
    }

    if (isolation.status === "aligning") {
        updatePipelineStatusUI("Aligning User...");
    }

    // Case 1: 1 hand isolated -> check physical handedness
    if (numHands === 1) {
        const primary = isolation.hands[0];
        const physicalHand = primary.physicalHand;

        // If user is showing only their Left Hand: DO NOT send prediction requests!
        if (physicalHand === "Left") {
            landmarkStabilizer.resetHand("Left");
            landmarkEMASmoother.reset("Left");
            updateHandStatusUI("left_ignored");
            updatePipelineStatusUI("Aligning User...");
            const now = performance.now();
            if (now - lastLeftHandToastTime > 3500) {
                lastLeftHandToastTime = now;
                showToast("Please use your Right Hand for single-hand signs", "warning", 3500);
            }
            return;
        }

        // Single physical Right hand: Smooth with EMA (alpha ~ 0.65) and send
        updateHandStatusUI("right");
        isRequestPending = true;
        isPredicting = true;
        const currentSeq = ++predictSequenceId;
        try {
            const rawLandmarks = primary.rawLandmarks;
            const smoothedOneEuro = (stabilizedHands && stabilizedHands[0])
                ? stabilizedHands[0].landmarks
                : landmarkStabilizer.filterHand("Right", rawLandmarks);
            const emaSmoothed = landmarkEMASmoother.smooth("Right", smoothedOneEuro);
            const landmarks = toLandmarkArray(emaSmoothed);

            const payload = {
                hand_type: "right",
                handedness: "Right",
                landmarks: landmarks,
                is_mirrored: isMirrored,
                seq_id: currentSeq
            };

            await sendInferenceRequest(payload, currentSeq);
        } catch (err) {
            // Suppress transient drops
        } finally {
            isRequestPending = false;
            isPredicting = false;
        }
        return;
    }

    // Case 2: 2 hands isolated -> verify strict opposite handedness (Right & Left)
    if (numHands >= 2) {
        const h0 = isolation.hands[0];
        const h1 = isolation.hands[1];

        updateHandStatusUI("both");
        isRequestPending = true;
        isPredicting = true;
        const currentSeq = ++predictSequenceId;
        try {
            const rHandObj = (h0.physicalHand === "Right") ? h0 : h1;
            const lHandObj = (h0.physicalHand === "Left") ? h0 : h1;

            const rSmoothedOneEuro = landmarkStabilizer.filterHand("Right", rHandObj.rawLandmarks);
            const lSmoothedOneEuro = landmarkStabilizer.filterHand("Left", lHandObj.rawLandmarks);

            const rEma = landmarkEMASmoother.smooth("Right", rSmoothedOneEuro);
            const lEma = landmarkEMASmoother.smooth("Left", lSmoothedOneEuro);

            const rPts = toLandmarkArray(rEma);
            const lPts = toLandmarkArray(lEma);

            const handsList = [
                { label: "Right", points: rPts },
                { label: "Left", points: lPts }
            ];

            const payload = {
                hand_type: "both",
                handedness: "Both",
                landmarks: rPts,
                hands: handsList,
                all_hands: handsList,
                is_mirrored: isMirrored,
                seq_id: currentSeq
            };

            await sendInferenceRequest(payload, currentSeq);
        } catch (err) {
            // Suppress transient drops
        } finally {
            isRequestPending = false;
            isPredicting = false;
        }
        return;
    }
}

async function sendInferenceRequest(payload, seqId) {
    // Optional Local Edge Inference Fallback:
    if (isEdgeModelReady && edgeModel && !navigator.onLine) {
        try {
            const edgeResult = await runEdgeInference(payload);
            if (edgeResult && seqId > lastHandledSequenceId) {
                lastHandledSequenceId = seqId;
                applyTelemetry(edgeResult);
            }
            return;
        } catch (edgeErr) {
            console.warn("Edge inference fallback error, reverting to server fetch:", edgeErr);
        }
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    let response = null;
    try {
        response = await fetch(`${currentInferenceUrl}/predict_landmarks`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: controller.signal
        });
        if (!response.ok && response.status === 404) {
            response = await fetch(`${currentInferenceUrl}/predict`, {
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
        // Drop any responses that arrived out-of-order or are older than the last handled request
        if (seqId > lastHandledSequenceId) {
            lastHandledSequenceId = seqId;
            applyTelemetry(data);
        }
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
        // Request continuous auto exposure for rapid lighting adaptation
        const highResConstraints = {
            video: deviceId ? { deviceId: { exact: deviceId } } : {
                facingMode: "user",
                width: { ideal: 1920, min: 1280 },
                height: { ideal: 1080, min: 720 },
                frameRate: { ideal: 60, min: 30 },
                advanced: [{ exposureMode: "continuous" }]
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

        // Low-light luminance dynamic range preprocessor
        let lumSamplingCanvas = null;
        let lumSamplingCtx = null;
        let lastLumCheckTime = 0;

        function evaluateLowLightAndEnhance(videoEl) {
            if (!videoEl || videoEl.videoWidth === 0) return;
            const now = performance.now();
            if (now - lastLumCheckTime < 250) return;
            lastLumCheckTime = now;

            if (!lumSamplingCanvas) {
                lumSamplingCanvas = document.createElement("canvas");
                lumSamplingCanvas.width = 40;
                lumSamplingCanvas.height = 30;
                lumSamplingCtx = lumSamplingCanvas.getContext("2d", { willReadFrequently: true });
            }

            try {
                lumSamplingCtx.drawImage(videoEl, 0, 0, 40, 30);
                const data = lumSamplingCtx.getImageData(0, 0, 40, 30).data;
                let sumY = 0;
                const total = 40 * 30;
                for (let i = 0; i < data.length; i += 4) {
                    sumY += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                }
                const avgLuminance = sumY / total;
                if (avgLuminance < 45) {
                    videoEl.style.filter = "brightness(1.25) contrast(1.15)";
                } else {
                    videoEl.style.filter = "none";
                }
            } catch (e) {}
        }

        // Mutex guard prevents concurrent mpHands.send calls from corrupting MediaPipe WASM memory
        let isProcessingFrame = false;
        const frameLoop = async () => {
            if (!isWebcamActive) return;
            evaluateLowLightAndEnhance(videoElement);
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

    fetch(`${API_BASE_URL}/api/stt/transcribe`, {
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

    fetch(`${API_BASE_URL}/correct_gesture`, {
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

// ── Interactive Sign Dictionary & Stick Figure Animation Visualizer ─────────
let dictionaryData = [];
let dictionaryMap = {};
let currentDictSign = null;
let dictAnimRunning = true;
let dictAnimSpeed = 1.0;
let dictCurrentKeyframeIndex = 0;
let dictLastFrameTimestamp = 0;
let dictAnimReqId = null;
let currentDictCategory = 'all';

async function loadDictionaryData() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/dictionary`);
        if (res.ok) {
            const data = await res.json();
            if (data.ok && data.dictionary) {
                dictionaryData = data.dictionary;
                dictionaryMap = {};
                for (const item of dictionaryData) {
                    dictionaryMap[item.id] = item;
                }
                if (!currentDictSign) {
                    selectDictionarySign('A');
                }
            }
        }
    } catch (e) {
        console.warn("Failed to load sign dictionary data:", e);
    }
}

function openGestureModal() {
    if (dom.gestureModal) {
        dom.gestureModal.style.display = "flex";
        dom.gestureModal.style.opacity = "1";
    }
    if (dictionaryData.length === 0) {
        loadDictionaryData();
    } else if (!currentDictSign) {
        selectDictionarySign('A');
    }
    startDictAnimationLoop();
}

function closeGestureModal() {
    if (dom.gestureModal) {
        dom.gestureModal.style.display = "none";
    }
    if (dictAnimReqId) {
        cancelAnimationFrame(dictAnimReqId);
        dictAnimReqId = null;
    }
}

function setDictCategory(cat) {
    currentDictCategory = cat;
    const pills = document.querySelectorAll('.dict-tab-pill');
    pills.forEach(p => {
        const text = p.textContent.toLowerCase();
        if (cat === 'all' ? text.includes('all') : text.includes(cat)) {
            p.classList.add('active');
        } else {
            p.classList.remove('active');
        }
    });

    const searchInput = document.getElementById('gesture-search');
    filterDictionaryList(searchInput ? searchInput.value : '');
}

function filterDictionaryList(query) {
    const q = (query || "").toUpperCase().trim();
    const items = document.querySelectorAll('.gesture-badge-item');

    items.forEach(item => {
        const itemCat = item.getAttribute('data-category') || 'static';
        const label = (item.getAttribute('data-label') || item.textContent).toUpperCase().trim();
        const matchesQuery = !q || label.includes(q);
        const matchesCat = (currentDictCategory === 'all') || (itemCat === currentDictCategory);

        if (matchesQuery && matchesCat) {
            item.style.display = 'flex';
        } else {
            item.style.display = 'none';
        }
    });
}

const FINGER_PLACEMENT_BADGES = {
    "A": "Thumb: Lateral Side (Upright)",
    "B": "Fingers: 4 Flat Upright",
    "C": "Hand: Curved 'C' Silhouette",
    "D": "Index: Pointing Up, Loop Closed",
    "E": "Fingertips: Curled into Palm Base",
    "F": "Index + Thumb: Circle, 3 Up",
    "G": "Index + Thumb: Pointing Sideways",
    "H": "Index + Middle: Pointing Sideways",
    "I": "Pinky: Upright, Fist Closed",
    "J": "Pinky: Downward Swoop Hook",
    "K": "Index: Up, Middle Forward",
    "L": "Index + Thumb: 90° 'L' Shape",
    "M": "Thumb: Under 3 Fingers",
    "N": "Thumb: Under 2 Fingers",
    "O": "All Tips: Closed Oval Loop",
    "P": "K-Shape: Pointing Downward",
    "Q": "G-Shape: Pointing Downward",
    "R": "Index + Middle: Crossed",
    "S": "Thumb: Across Front Knuckles",
    "T": "Thumb: Tucked Between Index/Mid",
    "U": "Index + Middle: Touching Up",
    "V": "Index + Middle: 'V' Spread",
    "W": "3 Fingers: 'W' Spread Up",
    "X": "Index: Bent Hook",
    "Y": "Thumb + Pinky: Spread Wide",
    "Z": "Index: 'Z' Zigzag In Air",
    "HELLO": "Flat Hand: Temple Wave",
    "THANK YOU": "Flat Hand: Chin to Receiver",
    "I LOVE YOU": "Thumb + Index + Pinky Up",
    "LOVE": "Both Arms: Crossed Over Chest",
    "FINE": "5-Hand: Thumbs on Chest",
    "ME": "Index: Pointing to Chest",
    "YOU": "Index: Pointing Forward",
    "HOW ARE YOU": "Cupped Hands: Flip Palms Up",
    "NICE TO MEET YOU": "Slide Palm + Meet Index",
    "START": "Both Hands: Open Facing Cam",
    "STOP": "Both Hands: Vertical Halt",
    "SPACE": "Both Hands: Move Outward",
    "BACKSPACE": "Both Hands: Flick Backward",
    "NEUTRAL": "Both Hands: Relaxed at Rest"
};

function selectDictionarySign(signId) {
    let sign = dictionaryMap[signId];
    if (!sign) {
        // Case-insensitive lookup
        const upper = String(signId).toUpperCase().trim();
        for (const k in dictionaryMap) {
            if (k.toUpperCase().trim() === upper) {
                sign = dictionaryMap[k];
                break;
            }
        }
    }
    if (!sign) return;

    currentDictSign = sign;
    dictCurrentKeyframeIndex = 0;

    // Highlight active item in list
    document.querySelectorAll('.gesture-badge-item').forEach(el => {
        const lbl = (el.getAttribute('data-label') || el.textContent).toUpperCase().trim();
        if (lbl === sign.id.toUpperCase().trim()) {
            el.classList.add('active-dict-item');
        } else {
            el.classList.remove('active-dict-item');
        }
    });

    // Update details pane
    const titleEl = document.getElementById('dict-sign-title');
    if (titleEl) titleEl.textContent = `Sign "${sign.name}"`;

    const catBadge = document.getElementById('dict-sign-badge');
    if (catBadge) {
        const isDyn = (sign.category === 'dynamic');
        catBadge.textContent = isDyn ? 'DYNAMIC' : 'STATIC';
        catBadge.className = `dict-badge ${isDyn ? 'dynamic' : 'static'}`;
    }

    const handBadge = document.getElementById('dict-hand-badge');
    if (handBadge) {
        const isBoth = (sign.handedness === 'both');
        handBadge.textContent = isBoth ? '🙌 BOTH HANDS' : '✋ RIGHT HAND';
    }

    const placementBadge = document.getElementById('dict-placement-badge');
    if (placementBadge) {
        const desc = FINGER_PLACEMENT_BADGES[sign.id.toUpperCase()] || "Finger alignment verified";
        placementBadge.textContent = desc;
    }

    const descEl = document.getElementById('dict-sign-desc');
    if (descEl) descEl.textContent = sign.description || "Canonical gesture posture.";

    const tipsEl = document.getElementById('dict-sign-tips');
    if (tipsEl) tipsEl.textContent = sign.tips || "Maintain steady finger alignment.";

    renderDictionaryCanvas();
}

function toggleDictAnimation() {
    dictAnimRunning = !dictAnimRunning;
    const icon = document.getElementById('dict-anim-icon');
    const text = document.getElementById('dict-anim-text');
    if (icon) icon.textContent = dictAnimRunning ? '⏸' : '▶';
    if (text) text.textContent = dictAnimRunning ? 'Pause' : 'Play';
}

function toggleDictSpeed() {
    dictAnimSpeed = (dictAnimSpeed === 1.0) ? 0.5 : 1.0;
    const btn = document.getElementById('dict-anim-speed');
    if (btn) btn.textContent = `⚡ Speed: ${dictAnimSpeed}x`;
}

function restartDictAnimation() {
    dictCurrentKeyframeIndex = 0;
    renderDictionaryCanvas();
}

function startDictAnimationLoop() {
    if (dictAnimReqId) cancelAnimationFrame(dictAnimReqId);

    const loop = (timestamp) => {
        if (!dictLastFrameTimestamp) dictLastFrameTimestamp = timestamp;
        const elapsed = timestamp - dictLastFrameTimestamp;

        // 12-15 FPS smooth keyframe loop (~75ms per frame)
        const frameInterval = 75 / dictAnimSpeed;
        if (dictAnimRunning && elapsed >= frameInterval) {
            dictLastFrameTimestamp = timestamp;
            if (currentDictSign && currentDictSign.keyframes && currentDictSign.keyframes.length > 0) {
                dictCurrentKeyframeIndex = (dictCurrentKeyframeIndex + 1) % currentDictSign.keyframes.length;
            }
            renderDictionaryCanvas();
        } else if (!dictAnimRunning) {
            renderDictionaryCanvas();
        }

        dictAnimReqId = requestAnimationFrame(loop);
    };

    dictAnimReqId = requestAnimationFrame(loop);
}

/**
 * Dual-View Skeleton Visualizer (#dictionary-canvas)
 * Left Panel (Upper Body Context): Demonstrates head, shoulders, torso, and arm angle posture
 * Right Panel (Zoomed Hand Detail): Enlarges 21-point hand skeleton with distinct joint colors:
 *   Thumb: Orange (#FF9800), Index: Cyan (#00E5FF), Middle: Green (#00E676),
 *   Ring: Yellow (#FFEA00), Pinky: Pink (#FF4081), Wrist: White (#FFFFFF).
 */
function renderDictionaryCanvas() {
    const canvas = document.getElementById('dictionary-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // 1. Dark Futuristic Cybernetic Background
    ctx.clearRect(0, 0, W, H);
    const bgGrad = ctx.createRadialGradient(W / 2, H / 2, 20, W / 2, H / 2, W * 0.7);
    bgGrad.addColorStop(0, '#0d1527');
    bgGrad.addColorStop(1, '#050811');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    // Subtle Sci-Fi Grid Backdrop
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 20; x < W; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
    }
    for (let y = 20; y < H; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
    }

    const splitX = Math.floor(W * 0.38);

    // ── LEFT PANEL: UPPER BODY POSTURE CONTEXT ─────────────────────────────
    ctx.save();
    // Panel Header Tag
    ctx.font = '600 10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(0, 240, 255, 0.8)';
    ctx.textAlign = 'center';
    ctx.fillText('UPPER BODY CONTEXT', splitX / 2, 18);

    const cx = splitX / 2;
    const headY = 44;
    const neckY = 64;
    const shoulderY = 76;
    const spineBottomY = 158;

    const shoulderLX = cx - 36;
    const shoulderRX = cx + 36;

    // Head Visor & Outline
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.2;
    ctx.fillStyle = 'rgba(14, 165, 233, 0.18)';
    ctx.beginPath();
    ctx.arc(cx, headY, 15, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Cyan Visor
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 8, headY);
    ctx.lineTo(cx + 8, headY);
    ctx.stroke();

    // Neck
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(cx, headY + 15);
    ctx.lineTo(cx, neckY);
    ctx.stroke();

    // Shoulder Crossbar
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    ctx.moveTo(shoulderLX, shoulderY);
    ctx.lineTo(shoulderRX, shoulderY);
    ctx.stroke();

    // Spine & Torso
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    ctx.moveTo(cx, neckY);
    ctx.lineTo(cx, spineBottomY);
    ctx.stroke();

    // Hips Bar
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(cx - 22, spineBottomY);
    ctx.lineTo(cx + 22, spineBottomY);
    ctx.stroke();

    const isBoth = (currentDictSign && (currentDictSign.handedness === 'both' || (currentDictSign.landmarks_secondary && currentDictSign.landmarks_secondary.length > 0)));

    // Right Arm: Dominant Signing Hand
    let rElbowX = shoulderRX + 20;
    let rElbowY = shoulderY + 38;
    let rWristX = shoulderRX + 10;
    let rWristY = shoulderY + 76;

    if (currentDictSign && currentDictSign.category === 'dynamic') {
        const dynOffset = Math.sin((dictCurrentKeyframeIndex / 12) * Math.PI * 2) * 6;
        rWristX += dynOffset;
    }

    ctx.strokeStyle = '#00E5FF';
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    ctx.moveTo(shoulderRX, shoulderY);
    ctx.lineTo(rElbowX, rElbowY);
    ctx.lineTo(rWristX, rWristY);
    ctx.stroke();

    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(shoulderRX, shoulderY, 3.5, 0, Math.PI * 2);
    ctx.arc(rElbowX, rElbowY, 3, 0, Math.PI * 2);
    ctx.arc(rWristX, rWristY, 4, 0, Math.PI * 2);
    ctx.fill();

    // Left Arm
    if (isBoth) {
        let lElbowX = shoulderLX - 20;
        let lElbowY = shoulderY + 38;
        let lWristX = shoulderLX - 10;
        let lWristY = shoulderY + 76;

        ctx.strokeStyle = '#c026d3';
        ctx.lineWidth = 2.8;
        ctx.beginPath();
        ctx.moveTo(shoulderLX, shoulderY);
        ctx.lineTo(lElbowX, lElbowY);
        ctx.lineTo(lWristX, lWristY);
        ctx.stroke();

        ctx.fillStyle = '#e879f9';
        ctx.beginPath();
        ctx.arc(shoulderLX, shoulderY, 3.5, 0, Math.PI * 2);
        ctx.arc(lElbowX, lElbowY, 3, 0, Math.PI * 2);
        ctx.arc(lWristX, lWristY, 4, 0, Math.PI * 2);
        ctx.fill();
    } else {
        const lElbowX = shoulderLX - 8;
        const lElbowY = shoulderY + 38;
        const lHandX = shoulderLX - 6;
        const lHandY = shoulderY + 78;

        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(shoulderLX, shoulderY);
        ctx.lineTo(lElbowX, lElbowY);
        ctx.lineTo(lHandX, lHandY);
        ctx.stroke();

        ctx.fillStyle = '#475569';
        ctx.beginPath();
        ctx.arc(lElbowX, lElbowY, 2.5, 0, Math.PI * 2);
        ctx.arc(lHandX, lHandY, 3, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.restore();

    // ── TWO-PANEL VERTICAL DIVIDER ─────────────────────────────────────────
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(splitX, 12);
    ctx.lineTo(splitX, H - 12);
    ctx.stroke();
    ctx.restore();

    // ── RIGHT PANEL: ZOOMED HAND SKELETON (21 JOINTS) ─────────────────────
    ctx.save();
    const rightCenterX = splitX + (W - splitX) / 2;
    ctx.font = '600 10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.textAlign = 'center';
    ctx.fillText('ZOOMED HAND SKELETON (21 JOINTS)', rightCenterX, 18);

    // Extract current landmarks
    let rHandPoints = null;
    if (currentDictSign) {
        if (currentDictSign.category === 'dynamic' && currentDictSign.keyframes && currentDictSign.keyframes.length > 0) {
            rHandPoints = currentDictSign.keyframes[dictCurrentKeyframeIndex] || currentDictSign.landmarks;
        } else {
            rHandPoints = currentDictSign.landmarks;
        }
    }

    if (rHandPoints && rHandPoints.length >= 21) {
        renderEnlargedHandSkeleton(ctx, splitX + 15, 30, W - 20, H - 35, rHandPoints);
    } else {
        ctx.font = '12px "Plus Jakarta Sans", sans-serif';
        ctx.fillStyle = '#64748b';
        ctx.textAlign = 'center';
        ctx.fillText('No hand landmark data available', rightCenterX, H / 2);
    }

    // Dynamic Keyframe Telemetry Bar
    if (currentDictSign && currentDictSign.category === 'dynamic') {
        const totalKf = (currentDictSign.keyframes && currentDictSign.keyframes.length) || 12;
        ctx.font = '500 10px "JetBrains Mono", monospace';
        ctx.fillStyle = '#00E5FF';
        ctx.textAlign = 'center';
        ctx.fillText(`KEYFRAME ${dictCurrentKeyframeIndex + 1} / ${totalKf}`, rightCenterX, H - 12);
    } else {
        // Finger Color Legend beneath the hand
        ctx.font = '500 9px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.65)';
        ctx.textAlign = 'center';
        ctx.fillText('Thumb: 🟠  Index: 🔷  Mid: 🟢  Ring: 🟡  Pinky: 🌸', rightCenterX, H - 12);
    }
    ctx.restore();
}

/**
 * Enlarges the 21-point hand skeleton inside the specified bounding box
 * with distinct joint colors:
 * Thumb: Orange (#FF9800), Index: Cyan (#00E5FF), Middle: Green (#00E676),
 * Ring: Yellow (#FFEA00), Pinky: Pink (#FF4081), Wrist: White (#FFFFFF).
 */
function renderEnlargedHandSkeleton(ctx, boxLeft, boxTop, boxRight, boxBottom, landmarks) {
    if (!landmarks || landmarks.length < 21) return;

    // Joint color mapping
    const JOINT_COLORS = {
        0: '#FFFFFF', // Wrist: White
        1: '#FF9800', 2: '#FF9800', 3: '#FF9800', 4: '#FF9800', // Thumb: Orange
        5: '#00E5FF', 6: '#00E5FF', 7: '#00E5FF', 8: '#00E5FF', // Index: Cyan
        9: '#00E676', 10: '#00E676', 11: '#00E676', 12: '#00E676', // Middle: Green
        13: '#FFEA00', 14: '#FFEA00', 15: '#FFEA00', 16: '#FFEA00', // Ring: Yellow
        17: '#FF4081', 18: '#FF4081', 19: '#FF4081', 20: '#FF4081'  // Pinky: Pink
    };

    // Calculate bounding box of hand landmarks
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (let i = 0; i < 21; i++) {
        const p = landmarks[i];
        const px = Number(p.x !== undefined ? p.x : p[0]);
        const py = Number(p.y !== undefined ? p.y : p[1]);
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
    }

    const spanX = Math.max(0.01, maxX - minX);
    const spanY = Math.max(0.01, maxY - minY);

    const availW = (boxRight - boxLeft);
    const availH = (boxBottom - boxTop);

    // Maintain aspect ratio while zooming to fill panel
    const scale = Math.min(availW / spanX, availH / spanY) * 0.82;
    const centerNormX = (minX + maxX) / 2;
    const centerNormY = (minY + maxY) / 2;

    const targetCenterX = boxLeft + availW / 2;
    const targetCenterY = boxTop + availH / 2;

    const screenPoints = landmarks.map(p => {
        const px = Number(p.x !== undefined ? p.x : p[0]);
        const py = Number(p.y !== undefined ? p.y : p[1]);
        return {
            x: targetCenterX + (px - centerNormX) * scale,
            y: targetCenterY + (py - centerNormY) * scale
        };
    });

    // 1. Draw Bones
    ctx.lineWidth = 2.4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const [i, j] of HAND_CONNECTIONS) {
        const p0 = screenPoints[i];
        const p1 = screenPoints[j];
        if (!p0 || !p1) continue;

        // Bone color matching finger
        let boneColor = 'rgba(255, 255, 255, 0.45)';
        if (i >= 1 && j <= 4) boneColor = 'rgba(255, 152, 0, 0.75)';
        else if (i >= 5 && j <= 8) boneColor = 'rgba(0, 229, 255, 0.75)';
        else if (i >= 9 && j <= 12) boneColor = 'rgba(0, 230, 118, 0.75)';
        else if (i >= 13 && j <= 16) boneColor = 'rgba(255, 234, 0, 0.75)';
        else if (i >= 17 && j <= 20) boneColor = 'rgba(255, 64, 129, 0.75)';

        ctx.strokeStyle = boneColor;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
    }

    // 2. Draw 21 Joints with Distinct Specific Colors
    const fingertips = new Set([4, 8, 12, 16, 20]);
    for (let k = 0; k < 21; k++) {
        const sp = screenPoints[k];
        if (!sp) continue;

        const color = JOINT_COLORS[k] || '#FFFFFF';

        if (fingertips.has(k)) {
            // Enlarged Fingertip Node with Outer Halo Glow
            ctx.beginPath();
            ctx.arc(sp.x, sp.y, 6.2, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            ctx.lineWidth = 1.8;
            ctx.strokeStyle = '#FFFFFF';
            ctx.stroke();
        } else if (k === 0) {
            // Wrist Node: Pure White with Glowing Core
            ctx.beginPath();
            ctx.arc(sp.x, sp.y, 4.8, 0, Math.PI * 2);
            ctx.fillStyle = '#FFFFFF';
            ctx.fill();
            ctx.lineWidth = 1.2;
            ctx.strokeStyle = 'rgba(0, 229, 255, 0.8)';
            ctx.stroke();
        } else {
            // Knuckle / Joint Node
            ctx.beginPath();
            ctx.arc(sp.x, sp.y, 3.4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.lineWidth = 1.0;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.stroke();
        }
    }
}

// ── Dominant Hand Switching ─────────────────────────────────────────────────
function setDominantHand(hand) {
    fetch(`${API_BASE_URL}/set_dominant_hand`, {
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

    // Live confidence
    const confVal = data.live_conf || 0;
    const pct = Math.round(confVal * 100);

    const smoothedDisplay = getSmoothedDisplayGesture(displayGesture, confVal);
    if (dom.gestureLabel) {
        dom.gestureLabel.textContent = smoothedDisplay;
    }
    const currentSignEl = document.getElementById("current-sign");
    if (currentSignEl && currentSignEl !== dom.gestureLabel) {
        currentSignEl.textContent = smoothedDisplay;
    }
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
            if (elapsedSec >= 1.5 && (isNiceEndPose || isPointingYou)) {
                isEndSignal = true;
            }
        } else if (clientDynamicGesture === "HOW ARE YOU") {
            const isPointingYou = (rawPredUpper === "YOU" || cleanPredUpper === "YOU" || (data.right_gesture && String(data.right_gesture).toUpperCase() === "YOU"));
            const isHowEndPose = (rawPredUpper === "HOW ARE YOU_END" || rawPredUpper.endsWith("_END"));
            if (elapsedSec >= 0.8 && (isPointingYou || isHowEndPose)) {
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
    if (isStartCandidate && confVal >= 0.40 && DYNAMIC_GESTURES.has(cleanPredUpper)) {
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
        predictionWindow.push("FRAGMENT");
        if (predictionWindow.length > PREDICTION_WINDOW_MAX) predictionWindow.shift();
        renderStability(0, Math.ceil(predictionWindow.length * MAJORITY_VOTE_RATIO), false);
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
        hasPassedThroughNeutral = true;
        if (now >= lockUntil) {
            activeSign = null;
            candidateSign = null;
            candidateCount = 0;
            confirmationBuffer = [];
            renderStability(0, CONFIRM_WIN_COUNT, false);
        }
        if (clientDynamicGesture) {
            clientDynamicGesture = null;
            if (dynBanner) dynBanner.style.display = "none";
            if (dynTelemetryRow) dynTelemetryRow.style.display = "none";
        }

        const nowHold = performance.now();
        const inHold = (nowHold - confirmedGestureTime < GESTURE_LOCK_HOLD_MS) || (now < lockUntil);
        if (!inHold) {
            updatePipelineStatusUI(isGestureRecordingActive ? "RECORDING (RESTING)" : "STANDBY (RESTING)");
        }
        return;
    }

    // ── State Hysteresis & Latching Prediction Handler ──
    const cleanPred = displayGesture || cleanPredUpper || stripInternalSuffix(data.prediction || "");

    // Backend Margin Gate handling:
    // If backend marked the prediction status as "ambiguous" (e.g. fist cluster margin < 0.15)
    if (data.status === "ambiguous") {
        updatePipelineStatusUI("Ambiguous (Stabilizing...)");
        if (activeSign !== null) {
            // Drop rapid override swapping and hold prior confirmed state
            if (dom.gestureLabel) dom.gestureLabel.textContent = activeSign;
            const currentSignEl = document.getElementById("current-sign");
            if (currentSignEl && currentSignEl !== dom.gestureLabel) {
                currentSignEl.textContent = activeSign;
            }
        }
        return;
    }

    const FIST_SIGNS = new Set(["A", "S", "T", "N", "M", "E"]);
    const candUpper = (cleanPred || "").toUpperCase().trim();
    const predUpper = (data.prediction || "").toUpperCase().trim();
    const isFistDisambiguationSwap = (
        FIST_SIGNS.has(activeSign) && (FIST_SIGNS.has(candUpper) || FIST_SIGNS.has(predUpper))
    );

    if (now < lockUntil && activeSign !== null && !isFistDisambiguationSwap) {
        // Cooldown lockout: maintain current sign without volatile interim swapping
        if (dom.gestureLabel) dom.gestureLabel.textContent = activeSign;
        const currentSignEl = document.getElementById("current-sign");
        if (currentSignEl && currentSignEl !== dom.gestureLabel) {
            currentSignEl.textContent = activeSign;
        }
        updatePipelineStatusUI("Stable");
    } else {
        handlePrediction(cleanPred, confVal, data.status);
        const displayWord = (activeSign !== null)
            ? activeSign
            : ((lastConfirmedGesture && lastConfirmedGesture !== "—") ? lastConfirmedGesture : (cleanPred || "—"));
        if (dom.gestureLabel) dom.gestureLabel.textContent = displayWord;
        const currentSignEl = document.getElementById("current-sign");
        if (currentSignEl && currentSignEl !== dom.gestureLabel) {
            currentSignEl.textContent = displayWord;
        }
    }
}

function commitSignToUI(sign) {
    if (!sign || sign === "—") return;
    const now = Date.now();
    const upper = sign.toUpperCase().trim();

    if (dom.gestureLabel) dom.gestureLabel.textContent = sign;
    const currentSignEl = document.getElementById("current-sign");
    if (currentSignEl && currentSignEl !== dom.gestureLabel) {
        currentSignEl.textContent = sign;
    }

    lastConfirmedGesture = sign;
    confirmedGestureTime = performance.now();
    updatePipelineStatusUI("Stable");
    renderStability(CONFIRM_WIN_COUNT, CONFIRM_WIN_COUNT, true);

    const DYNAMIC_GESTURES = new Set(["HOW ARE YOU", "NICE TO MEET YOU", "J", "Z"]);
    if (upper === "NEUTRAL" || upper === "IDLE") {
        hasPassedThroughNeutral = true;
        return;
    }
    if (DYNAMIC_GESTURES.has(upper)) {
        return;
    }

    // Execute dialogue/translation commit
    if (upper === "START" || upper === "STOP") {
        if (upper === "START" && !isGestureRecordingActive) {
            isGestureRecordingActive = true;
            updateGestureRecUI(true);
            speakText("Recording start");
            showToast("Recording started — START gesture detected", "success");
        } else if (upper === "STOP" && isGestureRecordingActive) {
            isGestureRecordingActive = false;
            updateGestureRecUI(false);
            speakCompletedSentence(clientSentence);
            showToast("Recording paused — STOP gesture detected", "info");
        }
    } else if (upper === "SPACE") {
        if (now - clientLastTriggerTime > REPEAT_DELAY_MS) {
            if (isGestureRecordingActive) appendSignToClientSentence("SPACE");
            clientLastTriggerTime = now;
        }
    } else if (upper === "BACKSPACE" || upper === "BACK SPACE") {
        if (now - clientLastTriggerTime > REPEAT_DELAY_MS) {
            backspaceSentenceLocal();
            clientLastTriggerTime = now;
        }
    } else if (upper === "YOU" && clientDynamicGesture) {
        const completedGesture = clientDynamicGesture;
        clientDynamicGesture = null;
        clientDynamicNeutralStart = 0;
        clientDynamicCooldownUntil = now + REPEAT_DELAY_MS;
        clientSuppressYouUntil = now + REPEAT_DELAY_MS;
        clientLastWord = completedGesture;
        clientLastTriggerTime = now;
        const dynBanner = document.getElementById("dynamic-timer-banner");
        const dynTelemetryRow = document.getElementById("dynamic-timer-telemetry");
        if (dynBanner) dynBanner.style.display = "none";
        if (dynTelemetryRow) dynTelemetryRow.style.display = "none";
        if (isGestureRecordingActive) {
            appendSignToClientSentence(completedGesture);
            showToast(`Dynamic Sign: ${completedGesture}`, "success");
        }
    } else if (!(upper === "YOU" && now < clientSuppressYouUntil)) {
        if (sign !== clientLastWord || (now - clientLastTriggerTime > REPEAT_DELAY_MS)) {
            if (isGestureRecordingActive) {
                appendSignToClientSentence(sign);
                showToast(`Sign: ${sign}`, "info", 1500);
            }
            clientLastWord = sign;
            clientLastTriggerTime = now;
        }
    }
}

function handlePrediction(predictedLabel, confidence, status = "ok") {
    const now = Date.now();
    if (!predictedLabel || confidence < 0.50 || status === "ambiguous") return;

    // 4-frame confirmation buffer:
    // Push new inference label into the sliding window of size 4
    confirmationBuffer.push(predictedLabel);
    if (confirmationBuffer.length > CONFIRM_BUFFER_SIZE) {
        confirmationBuffer.shift();
    }

    // A gesture must win at least 3 out of 4 consecutive inference frames to become the active candidate
    const counts = {};
    let winningCandidate = null;
    let maxWins = 0;
    for (const label of confirmationBuffer) {
        counts[label] = (counts[label] || 0) + 1;
        if (counts[label] > maxWins) {
            maxWins = counts[label];
        }
        if (counts[label] >= CONFIRM_WIN_COUNT) {
            winningCandidate = label;
        }
    }

    renderStability(maxWins, CONFIRM_WIN_COUNT, !!winningCandidate);

    if (!winningCandidate) {
        return; // Did not meet 3-of-4 confirmation threshold yet
    }

    // 350ms hysteresis lock:
    // Once a gesture commits to the UI, lock it from being replaced by an adjacent similar sign
    // unless the new sign is held consistently past the lock duration (or it's a disambiguation tie-breaker)
    const FIST_SIGNS = new Set(["A", "S", "T", "N", "M", "E"]);
    const isFistDisambiguationSwap = (
        FIST_SIGNS.has(activeSign) && FIST_SIGNS.has(winningCandidate)
    );

    if (activeSign !== null) {
        if (winningCandidate === activeSign) {
            return;
        }
        if (now < lockUntil && !isFistDisambiguationSwap) {
            // Locked: suppress rapid swap
            return;
        }
    }

    // Commit confirmed winning candidate
    activeSign = winningCandidate;
    commitSignToUI(activeSign);
    lockUntil = now + HYSTERESIS_LOCK_MS; // 350ms hysteresis lock
}

// ── Fallback Polling Loop (Only when webcam not active) ─────────────────────
function poll() {
    if (isWebcamActive) return;
    fetch(`${API_BASE_URL}/gesture`)
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

    fetch(`${API_BASE_URL}/shutdown`, { method: "POST" }).catch(() => {});

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
initServerNegotiator();
startClientWebcam();
