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

    // Transcript Box (Sign Language)
    transcriptBox: document.getElementById("m-transcript-box"),
    charStats: document.getElementById("m-char-stats"),

    // Voice Speech Transcript Box
    voiceTranscriptBox: document.getElementById("m-voice-transcript-box"),
    voiceCharStats: document.getElementById("m-voice-char-stats"),

    // Sound Toggle Icons
    soundBtn: document.getElementById("btn-sound-toggle"),
    soundIconOn: document.getElementById("icon-sound-on"),
    soundIconOff: document.getElementById("icon-sound-off"),

    // Modals & Toast
    toast: document.getElementById("m-toast"),
    dictModal: document.getElementById("m-dict-modal"),
    dictSearch: document.getElementById("m-dict-search"),

    // Speech-To-Text elements
    voiceBtn: document.getElementById("btn-m-voice"),
    voiceBtnText: document.getElementById("btn-m-voice-text"),
    dockVoiceBtn: document.getElementById("btn-dock-voice"),
    sttPreview: document.getElementById("m-stt-preview"),
    sttPreviewText: document.getElementById("m-stt-preview-text")
};

// Client-Side Session State & Camera Tracking
let currentFacingMode = "user"; // 'user' (front) or 'environment' (rear)
let isMirrored = true;
let isGestureRecordingActive = false;
let clientSentence = "";
let mobileVoiceSentence = "";
let clientLastWord = "";
let clientStabilityBuffer = [];
let clientLastTriggerTime = 0;
let mobileDisplayHistory = [];
let mobileSmoothedDisplaySign = "—";
let mobileSmoothedConf = 0;

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

async function updateMobileServerNegotiation() {
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
                pill.textContent = "Local (0ms)";
                pill.style.background = "#10b981";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "none";
        } else {
            currentInferenceUrl = API_BASE_URL;
            if (pill) {
                pill.textContent = "Offline (Cloud)";
                pill.style.background = "#ef4444";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "inline";
        }
    } else if (serverMode === "cloud") {
        currentInferenceUrl = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? '' : window.location.origin;
        if (pill) {
            pill.textContent = "Cloud";
            pill.style.background = "#0284c7";
            pill.style.color = "#fff";
        }
        if (downloadLink) downloadLink.style.display = "none";
    } else {
        // "auto" mode
        if (isLocalAlive) {
            currentInferenceUrl = "http://127.0.0.1:5000";
            if (pill) {
                pill.textContent = "Auto: Local (0ms)";
                pill.style.background = "#10b981";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "none";
        } else {
            currentInferenceUrl = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? '' : window.location.origin;
            if (pill) {
                pill.textContent = "Auto: Cloud";
                pill.style.background = "#6366f1";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "none";
        }
    }
}

function initMobileServerNegotiator() {
    const select = document.getElementById("server-mode-select");
    if (select) {
        select.value = serverMode;
        select.addEventListener("change", (e) => {
            serverMode = e.target.value;
            localStorage.setItem("aslo_server_mode", serverMode);
            updateMobileServerNegotiation();
        });
    }
    updateMobileServerNegotiation();
    setInterval(updateMobileServerNegotiation, 10000);
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

// ── Sliding Window & Hysteresis Hold Constants & State ─────────────────────
const N_FRAME = 5;
const N_FRAMES = 5;
const PREDICTION_WINDOW_MAX = 5;
const MAJORITY_VOTE_RATIO = 0.60;
const GESTURE_LOCK_HOLD_MS = 120;

let mobilePredictionWindow = [];
let mobileLastConfirmedGesture = "—";
let mobileConfirmedGestureTime = 0;
let mobileHasPassedThroughNeutral = false;
let mobileCurrentPipelineState = "Detecting...";
let mobilePredictSequenceId = 0;
let mobileLastHandledSequenceId = 0;

function getSmoothedDisplaySign(newDisplay, conf = 0) {
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

    if (mobileCurrentPipelineState === "Stable" && mobileLastConfirmedGesture && mobileLastConfirmedGesture !== "—") {
        return mobileLastConfirmedGesture;
    }

    // Maintain the last stable translated word on screen during transitions instead of flickering or disappearing
    if (mobileLastConfirmedGesture && mobileLastConfirmedGesture !== "—") {
        return mobileLastConfirmedGesture;
    }

    return (newDisplay && newDisplay !== "NEUTRAL") ? newDisplay : "—";
}

let clientDynamicGesture = null;
let clientDynamicStartTime = 0;
const CLIENT_DYNAMIC_MAX_SEC = 2.8;
let clientDynamicNeutralStart = 0;
let clientSuppressYouUntil = 0;
let clientDynamicCooldownUntil = 0;
let dynamicReadyForStart = true;
let clientLastDynamicCommitted = { gesture: "", time: 0 };
let lastLeftHandToastTime = 0;
const REPEAT_DELAY_MS = 140; // Sub-150ms repeat cooldown
const STABILITY_REQUIRED_COUNT = 3; // 3 out of 5 frames
const BUFFER_MAX_LEN = 5;

let isSoundMuted = localStorage.getItem("aslo_mobile_sound_muted") === "true";
let stabilitySegmentsCount = 8;
let webcamStream = null;
let isRequestPending = false;
let isPredicting = false;
let isWebcamActive = false;

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

/**
 * 1-Euro Filter Implementation for Mobile (Casiez et al., CHI 2012)
 * Adaptive low-pass filter specifically designed for human-computer interaction.
 * Heavily dampens high-frequency jitter during slow or stationary hand poses,
 * while dynamically increasing cutoff frequency during fast motion to eliminate lag.
 */
class MobileOneEuroFilter {
    constructor(minCutoff = 1.2, beta = 1.2, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
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
        // Tracking re-acquisition snap (> 0.45 normalized units)
        if (Math.abs(rawDiff) > 0.45) {
            this.xPrev = x;
            this.dxPrev = 0;
            return x;
        }

        const dx = rawDiff / dt;
        const alphaD = this.alpha(this.dCutoff, dt);
        const edx = alphaD * dx + (1.0 - alphaD) * this.dxPrev;
        this.dxPrev = edx;

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
 * Resolves the physical hand considering camera mirroring.
 * On front selfie camera (isMirrored === true), MediaPipe's non-mirrored model
 * classifies the user's physical Right hand as "Left".
 */
function getActualPhysicalHand(handednessObj) {
    if (!handednessObj) return "Right";
    const raw = (handednessObj.label || (handednessObj.classification && handednessObj.classification[0] && handednessObj.classification[0].label) || "").trim();
    if (isMirrored) {
        if (raw === "Left") return "Right";
        if (raw === "Right") return "Left";
    }
    return raw || "Right";
}

/**
 * ── Single-Person Hand Isolation & Filtering ────────────────────────────────
 * Locks onto the primary user closest to the camera in the central region:
 * 1. Hand scale metric: ||lm[0] - lm[9]||. Rejects hands < 65% of max detected scale.
 * 2. Center proximity: prioritizes wrists in central band 0.20 <= x <= 0.80.
 * 3. Spatial clustering: verifies pairs have |dy| < 0.25 and |dx| < 0.55, not on far opposite edges.
 * 4. Strict handedness pairing: at most ONE Right hand and ONE Left hand.
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

/**
 * Pre-Inference Landmark Exponential Moving Average (EMA: alpha ~ 0.65)
 * Suppresses frame-level tracking jitter before landmark feature extraction.
 */
class MobileLandmarkEMASmoother {
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

const mobileLandmarkEMASmoother = new MobileLandmarkEMASmoother(0.65);

function updateMobilePipelineStatusUI(status) {
    mobileCurrentPipelineState = status;
    if (dom.signSub) {
        dom.signSub.textContent = status;
        dom.signSub.className = "m-sign-sub";
        if (status === "Stable") {
            dom.signSub.classList.add("stable");
        } else if (status === "Ambiguous") {
            dom.signSub.classList.add("ambiguous");
        } else if (status === "Aligning User...") {
            dom.signSub.classList.add("aligning");
        } else if (status === "Detecting...") {
            dom.signSub.classList.add("detecting");
        }
    }
}

/**
 * Mobile Persistent Spatial Hand Tracker
 * Resolves hands by spatial continuity (wrist proximity) rather than noisy handedness classification.
 */
class MobileSpatialHandTracker {
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
                x: new MobileOneEuroFilter(this.minCutoff, this.beta, this.dCutoff),
                y: new MobileOneEuroFilter(this.minCutoff, this.beta, this.dCutoff),
                z: new MobileOneEuroFilter(this.minCutoff, this.beta, this.dCutoff)
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

            // Only reset inactive track after 350ms grace period
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
                const costNormal = Math.hypot(h0.wrist.x - this.tracks[0].prevWrist.x, h0.wrist.y - this.tracks[0].prevWrist.y) +
                                   Math.hypot(h1.wrist.x - this.tracks[1].prevWrist.x, h1.wrist.y - this.tracks[1].prevWrist.y);
                const costSwap = Math.hypot(h1.wrist.x - this.tracks[0].prevWrist.x, h1.wrist.y - this.tracks[0].prevWrist.y) +
                                 Math.hypot(h0.wrist.x - this.tracks[1].prevWrist.x, h0.wrist.y - this.tracks[1].prevWrist.y);
                if (costSwap < costNormal) {
                    pair0 = h1;
                    pair1 = h0;
                }
            } else if (t0HasPrev && !t1HasPrev) {
                const d0 = Math.hypot(h0.wrist.x - this.tracks[0].prevWrist.x, h0.wrist.y - this.tracks[0].prevWrist.y);
                const d1 = Math.hypot(h1.wrist.x - this.tracks[0].prevWrist.x, h1.wrist.y - this.tracks[0].prevWrist.y);
                if (d1 < d0) {
                    pair0 = h1;
                    pair1 = h0;
                }
            } else if (!t0HasPrev && t1HasPrev) {
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
 * Mobile Trajectory-Based State Machine for Dynamic "J"
 */
class MobileJTrajectoryTracker {
    constructor() {
        this.buffer = [];
        this.state = "IDLE";
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

const mobileJTrajectoryTracker = new MobileJTrajectoryTracker();

const mobileStabilizer = new MobileSpatialHandTracker(1.0, 0.05, 1.0);
let currentMobileStabilizedHands = [];

function getMobileTemporalSmoothedLandmarks(handKey, rawPts, timestamp = performance.now()) {
    return mobileStabilizer.filterHand(handKey, rawPts, timestamp);
}

function toMobileLandmarkArray(landmarks) {
    if (!landmarks) return [];
    return landmarks.map(p => {
        if (Array.isArray(p)) return p;
        return [Number(p.x || 0), Number(p.y || 0), Number(p.z || 0)];
    });
}



// ── Mobile Client-Side MediaPipe Hands Detector ─────────────────────────────
let mobileHands = null;
let mobileCamera = null;
let lastMobilePredictTime = 0;
const MOBILE_PREDICT_INTERVAL = 70; // 12-15 Hz (every ~70 ms)
let currentMobileMinDetectionConfidence = 0.55;
let mobileDroppedFramesCount = 0;

function initMobileMediaPipe() {
    if (mobileHands) return mobileHands;
    if (typeof Hands === "undefined") return null;

    try {
        mobileHands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        // Decoupled Dual-Threshold Confidence
        mobileHands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.55,
            minTrackingConfidence: 0.65
        });

        mobileHands.onResults((results) => {
            const canvas = dom.landmarkCanvas;
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            // Tracking fallback: if tracking drops > 4 frames, drop detection confidence to 0.45
            const hasRawHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
            if (!hasRawHands) {
                mobileDroppedFramesCount++;
                if (mobileDroppedFramesCount > 4 && currentMobileMinDetectionConfidence !== 0.45) {
                    currentMobileMinDetectionConfidence = 0.45;
                    mobileHands.setOptions({ minDetectionConfidence: 0.45, minTrackingConfidence: 0.65 });
                }
            } else {
                mobileDroppedFramesCount = 0;
                if (currentMobileMinDetectionConfidence !== 0.55) {
                    currentMobileMinDetectionConfidence = 0.55;
                    mobileHands.setOptions({ minDetectionConfidence: 0.55, minTrackingConfidence: 0.65 });
                }
            }

            const rect = canvas.getBoundingClientRect();
            if (canvas.width !== rect.width || canvas.height !== rect.height) {
                canvas.width = rect.width;
                canvas.height = rect.height;
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 1. Single-person hand isolation before skeleton rendering and inference
            const isolation = isolatePrimaryUserHands(results.multiHandLandmarks, results.multiHandedness);

            const now = performance.now();
            const isolatedLandmarks = isolation.hands.map(h => h.rawLandmarks);
            const isolatedHandedness = isolation.hands.map(h => h.handedness);
            const stabilizedList = mobileStabilizer.process(
                isolatedLandmarks,
                isolatedHandedness,
                now
            );

            for (let i = 0; i < stabilizedList.length; i++) {
                const smoothed = stabilizedList[i].landmarks;
                if (typeof drawConnectors === "function" && typeof HAND_CONNECTIONS !== "undefined") {
                    drawConnectors(ctx, smoothed, HAND_CONNECTIONS, {
                        color: "#FFFFFF",
                        lineWidth: 2.6
                    });
                } else if (typeof HAND_CONNECTIONS !== "undefined") {
                    ctx.save();
                    ctx.strokeStyle = "#FFFFFF";
                    ctx.lineWidth = 2.6;
                    ctx.lineCap = "round";
                    ctx.lineJoin = "round";
                    for (const [ci, cj] of HAND_CONNECTIONS) {
                        const p1 = smoothed[ci];
                        const p2 = smoothed[cj];
                        if (p1 && p2) {
                            ctx.beginPath();
                            ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
                            ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
                            ctx.stroke();
                        }
                    }
                    ctx.restore();
                }

                if (typeof drawLandmarks === "function") {
                    drawLandmarks(ctx, smoothed, {
                        color: "#00E5FF",
                        fillColor: "#00E5FF",
                        lineWidth: 1.0,
                        radius: 3.5
                    });
                } else {
                    ctx.save();
                    ctx.fillStyle = "#00E5FF";
                    for (const pt of smoothed) {
                        if (pt) {
                            ctx.beginPath();
                            ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3.5, 0, 2 * Math.PI);
                            ctx.fill();
                        }
                    }
                    ctx.restore();
                }
            }

            currentMobileStabilizedHands = stabilizedList;

            // Trajectory-Based State Machine for Dynamic "J" on Mobile
            if (currentMobileStabilizedHands.length === 1 && isolation.hands.length === 1 && isolation.hands[0].physicalHand === "Right") {
                const jResult = mobileJTrajectoryTracker.update(currentMobileStabilizedHands[0].landmarks, now);
                if (jResult.committed) {
                    activeSign = "J";
                    lastConfirmedGesture = "J";
                    lockUntil = Date.now() + HYSTERESIS_LOCK_MS;
                    if (isRecording) {
                        appendMobileSignToSentence("J");
                    }
                    showToast("Dynamic Sign: J");
                    applyMobileTelemetry({
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
                    updateMobilePipelineStatusUI("DYNAMIC: J (IN MOTION)");
                    return; // Suppress static single-frame updates
                }
            } else if (currentMobileStabilizedHands.length === 0) {
                mobileJTrajectoryTracker.reset();
            }

            // 2. Throttled backend landmark prediction (non-blocking lock)
            const nowTime = performance.now();
            if (nowTime - lastMobilePredictTime >= MOBILE_PREDICT_INTERVAL) {
                if (isRequestPending || isPredicting) return;
                lastMobilePredictTime = nowTime;
                sendMobileLandmarks(results, currentMobileStabilizedHands, isolation);
            }
        });

        return mobileHands;
    } catch (e) {
        console.error("Mobile MediaPipe init error:", e);
        return null;
    }
}

// ── Camera Mirror & Display Synchronizer ──────────────────────────────────
function updateCameraMirrorDisplay() {
    if (dom.webcam) {
        dom.webcam.style.transform = isMirrored ? "scaleX(-1)" : "scaleX(1)";
    }
    if (dom.landmarkCanvas) {
        dom.landmarkCanvas.style.transform = isMirrored ? "scaleX(-1)" : "scaleX(1)";
    }
}

// ── Transcript & Sentence State Management (Stateless Isolation) ───────────
function updateTranscriptDOM() {
    if (dom.transcriptBox) {
        const textElem = dom.transcriptBox.querySelector(".m-text-content");
        const trimmed = clientSentence.trim();
        if (!trimmed) {
            if (textElem) textElem.textContent = "(Waiting for signs… sign START to begin)";
            dom.transcriptBox.classList.add("empty");
        } else {
            if (textElem) textElem.textContent = clientSentence;
            dom.transcriptBox.classList.remove("empty");
        }
    }

    if (dom.charStats) {
        const trimmed = clientSentence.trim();
        const wordArr = trimmed ? trimmed.split(/\s+/) : [];
        dom.charStats.textContent = `${wordArr.length} words · ${clientSentence.length} chars`;
    }
}

function appendSignToTranscript(sign) {
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
    updateTranscriptDOM();
    speakText(clean);
}

function backspaceTranscriptLocal() {
    if (!clientSentence) return;
    const trimmed = clientSentence.trimEnd();
    if (trimmed.length === 0) {
        clientSentence = "";
        updateTranscriptDOM();
        return;
    }
    const lastSpace = trimmed.lastIndexOf(" ");
    if (lastSpace !== -1) {
        clientSentence = trimmed.substring(0, lastSpace + 1);
    } else {
        clientSentence = "";
    }
    updateTranscriptDOM();
    showToast("Removed last sign");
}

function updateStabilityDots(activeCount) {
    if (!dom.stabDotsContainer) return;
    const dots = dom.stabDotsContainer.children;
    const scaledCount = Math.min(stabilitySegmentsCount, Math.round((activeCount / STABILITY_REQUIRED_COUNT) * stabilitySegmentsCount));
    for (let i = 0; i < dots.length; i++) {
        dots[i].classList.toggle("filled", i < scaledCount);
    }
}

function updateRecordingUI() {
    if (dom.camBox) dom.camBox.classList.toggle("recording", isGestureRecordingActive);
    if (dom.recTagText) dom.recTagText.textContent = isGestureRecordingActive ? "REC" : "STANDBY";
    if (dom.recToggleBtn) dom.recToggleBtn.classList.toggle("recording", isGestureRecordingActive);
    if (dom.recBtnIcon && dom.recBtnText) {
        dom.recBtnIcon.textContent = isGestureRecordingActive ? "⏸" : "⏺";
        dom.recBtnText.textContent = isGestureRecordingActive ? "Stop Recording" : "Start Recording";
    }
    if (dom.signSub) dom.signSub.textContent = isGestureRecordingActive ? "Live tracking" : "Paused";
}

async function sendMobileLandmarks(results, stabilizedHands = [], precomputedIsolation = null) {
    if (isRequestPending || isPredicting) return;

    const isolation = precomputedIsolation || isolatePrimaryUserHands(
        results && results.multiHandLandmarks,
        results && results.multiHandedness
    );

    const numHands = isolation.hands.length;

    // Case 0: 0 hands detected -> idle / standby (cleanly reset filter state)
    if (numHands === 0) {
        mobileStabilizer.resetAll();
        mobileLandmarkEMASmoother.reset();
        updateMobilePipelineStatusUI("Detecting...");
        applyMobileTelemetry({
            ok: true,
            prediction: "—",
            live_gesture: "—",
            live_conf: 0.0,
            detected_hand: "none",
            status: "none"
        });
        return;
    }

    if (isolation.status === "aligning") {
        updateMobilePipelineStatusUI("Aligning User...");
    }

    // Case 1: 1 hand isolated -> check physical handedness
    if (numHands === 1) {
        const primary = isolation.hands[0];
        const physicalHand = primary.physicalHand;

        // If user is showing only their Left Hand: DO NOT send prediction requests!
        if (physicalHand === "Left") {
            mobileStabilizer.resetHand("Left");
            mobileLandmarkEMASmoother.reset("Left");
            updateMobilePipelineStatusUI("Aligning User...");
            applyMobileTelemetry({
                ok: true,
                prediction: null,
                message: "Please use your Right Hand for single-hand signs",
                detected_hand: "left_ignored",
                live_gesture: "—",
                live_conf: 0.0,
                status: "aligning"
            });
            const now = performance.now();
            if (now - lastLeftHandToastTime > 3500) {
                lastLeftHandToastTime = now;
                showToast("Please use your Right Hand for single-hand signs", 3500);
            }
            return;
        }

        // Single physical Right hand: Smooth with EMA (alpha ~ 0.65) and send
        isRequestPending = true;
        isPredicting = true;
        const currentSeq = ++mobilePredictSequenceId;
        try {
            const rawLandmarks = primary.rawLandmarks;
            const smoothedOneEuro = (stabilizedHands && stabilizedHands[0])
                ? stabilizedHands[0].landmarks
                : mobileStabilizer.filterHand("Right", rawLandmarks);
            const emaSmoothed = mobileLandmarkEMASmoother.smooth("Right", smoothedOneEuro);
            const landmarks = toMobileLandmarkArray(emaSmoothed);

            const payload = {
                landmarks: landmarks,
                handedness: "Right",
                hand_type: "right",
                is_mirrored: isMirrored,
                seq_id: currentSeq
            };

            await sendMobileInferenceRequest(payload, currentSeq);
        } catch (err) {
            // Tolerated frame drop
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

        isRequestPending = true;
        isPredicting = true;
        const currentSeq = ++mobilePredictSequenceId;
        try {
            const rHandObj = (h0.physicalHand === "Right") ? h0 : h1;
            const lHandObj = (h0.physicalHand === "Left") ? h0 : h1;

            const rSmoothedOneEuro = mobileStabilizer.filterHand("Right", rHandObj.rawLandmarks);
            const lSmoothedOneEuro = mobileStabilizer.filterHand("Left", lHandObj.rawLandmarks);

            const rEma = mobileLandmarkEMASmoother.smooth("Right", rSmoothedOneEuro);
            const lEma = mobileLandmarkEMASmoother.smooth("Left", lSmoothedOneEuro);

            const rPts = toMobileLandmarkArray(rEma);
            const lPts = toMobileLandmarkArray(lEma);

            const allHands = [
                { label: "Right", points: rPts },
                { label: "Left", points: lPts }
            ];

            const payload = {
                landmarks: rPts,
                all_hands: allHands,
                hands: allHands,
                handedness: "Both",
                hand_type: "both",
                is_mirrored: isMirrored,
                seq_id: currentSeq
            };

            await sendMobileInferenceRequest(payload, currentSeq);
        } catch (err) {
            // Tolerated frame drop
        } finally {
            isRequestPending = false;
            isPredicting = false;
        }
        return;
    }
}

async function sendMobileInferenceRequest(payload, seqId) {
    // Optional Local Edge Inference Fallback:
    if (isEdgeModelReady && edgeModel && !navigator.onLine) {
        try {
            const edgeResult = await runEdgeInference(payload);
            if (edgeResult && seqId > mobileLastHandledSequenceId) {
                mobileLastHandledSequenceId = seqId;
                applyMobileTelemetry(edgeResult);
            }
            return;
        } catch (edgeErr) {
            console.warn("Mobile edge inference error, reverting to server fetch:", edgeErr);
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
        if (seqId > mobileLastHandledSequenceId) {
            mobileLastHandledSequenceId = seqId;
            applyMobileTelemetry(data);
        }
    }
}

// ── Mobile Client Webcam Lifecycle ──────────────────────────────────────────
let mobileFrameLoopId = null;

async function startMobileWebcam() {
    try {
        initMobileMediaPipe();

        // 1. Cancel previous frame loop
        if (mobileFrameLoopId) {
            cancelAnimationFrame(mobileFrameLoopId);
            mobileFrameLoopId = null;
        }

        // 2. Stop previous MediaPipe Camera helper if any
        if (mobileCamera) {
            try { await mobileCamera.stop(); } catch (e) {}
            mobileCamera = null;
        }

        // 3. Stop previous stream tracks cleanly to release sensor
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => {
                try { t.stop(); } catch (_) {}
            });
            webcamStream = null;
        }

        if (dom.webcam) {
            try { dom.webcam.pause(); } catch (_) {}
            dom.webcam.srcObject = null;
        }

        // 4. Request camera using exact facingMode on mobile, with ideal fallback
        // Native sensor resolution is preserved without lowering or constraining video dimensions
        // Request continuous exposure mode for real-time sensor lighting compensation
        let stream = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { exact: currentFacingMode },
                    advanced: [{ exposureMode: "continuous" }]
                },
                audio: false
            });
        } catch (exactErr) {
            console.warn("[Mobile] Exact facingMode failed, falling back to ideal:", exactErr);
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: currentFacingMode },
                    advanced: [{ exposureMode: "continuous" }]
                },
                audio: false
            });
        }

        webcamStream = stream;
        if (dom.webcam) {
            dom.webcam.srcObject = webcamStream;
            updateCameraMirrorDisplay();
            await dom.webcam.play();
            dom.webcam.style.display = "block";
        }

        isWebcamActive = true;
        updateCameraMirrorDisplay();
        if (dom.camFallback) dom.camFallback.style.display = "none";

        // Mobile low-light luminance preprocessor
        let lumMobileCanvas = null;
        let lumMobileCtx = null;
        let lastMobileLumTime = 0;

        function evaluateMobileLowLight(videoEl) {
            if (!videoEl || videoEl.videoWidth === 0) return;
            const now = performance.now();
            if (now - lastMobileLumTime < 300) return;
            lastMobileLumTime = now;

            if (!lumMobileCanvas) {
                lumMobileCanvas = document.createElement("canvas");
                lumMobileCanvas.width = 40;
                lumMobileCanvas.height = 30;
                lumMobileCtx = lumMobileCanvas.getContext("2d", { willReadFrequently: true });
            }

            try {
                lumMobileCtx.drawImage(videoEl, 0, 0, 40, 30);
                const data = lumMobileCtx.getImageData(0, 0, 40, 30).data;
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

        // 5. Continuous frame processing loop directly into MediaPipe Hands with WASM mutex lock
        let isProcessingMobileFrame = false;
        const onFrame = async () => {
            if (!isWebcamActive) return;
            evaluateMobileLowLight(dom.webcam);
            if (mobileHands && !isProcessingMobileFrame && dom.webcam && dom.webcam.readyState >= 2 && !dom.webcam.paused) {
                isProcessingMobileFrame = true;
                try {
                    await mobileHands.send({ image: dom.webcam });
                } catch (_) {}
                finally {
                    isProcessingMobileFrame = false;
                }
            }
            if ("requestVideoFrameCallback" in dom.webcam) {
                dom.webcam.requestVideoFrameCallback(onFrame);
            } else {
                mobileFrameLoopId = requestAnimationFrame(onFrame);
            }
        };

        if ("requestVideoFrameCallback" in dom.webcam) {
            dom.webcam.requestVideoFrameCallback(onFrame);
        } else {
            mobileFrameLoopId = requestAnimationFrame(onFrame);
        }

        showToast(`Active: ${currentFacingMode === "user" ? "Front (Selfie)" : "Rear (Environment)"} Camera`);
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

// ── Real-Time Telemetry & Client-Isolated Gesture Processing ───────────────
function applyMobileTelemetry(data) {
    if (!data) return;

    const liveSign = stripInternalSuffix(data.live_gesture || "—");
    const confVal = data.live_conf || data.confidence || 0;
    const confPct = Math.round(confVal * 100);
    const detHand = data.detected_hand || "none";

    const CONTROL_SIGNS_MAP = {
        "START": "START",
        "STOP": "STOP",
        "SPACE": "SPACE",
        "BACKSPACE": "BACKSPACE",
        "BACK SPACE": "BACKSPACE",
        "NEUTRAL": "NEUTRAL"
    };
    const upperLive = (liveSign || "").toUpperCase();
    const rawPredUpper = String(data.raw_pred || data.prediction || "").toUpperCase().trim();
    const cleanPredUpper = stripInternalSuffix(data.prediction || "").toUpperCase().trim();
    const DYNAMIC_GESTURES = new Set(["HOW ARE YOU", "NICE TO MEET YOU", "J", "Z"]);

    let displaySign = CONTROL_SIGNS_MAP[upperLive] || liveSign;
    if (CONTROL_SIGNS_MAP[upperLive] || CONTROL_SIGNS_MAP[cleanPredUpper]) {
        const ctrl = CONTROL_SIGNS_MAP[upperLive] || CONTROL_SIGNS_MAP[cleanPredUpper];
        if ((ctrl === "START" || ctrl === "STOP") && detHand !== "both") {
            displaySign = "—";
        } else {
            displaySign = ctrl;
        }
    } else if (clientDynamicGesture) {
        displaySign = `${clientDynamicGesture} (IN MOTION)`;
    } else if (rawPredUpper.endsWith("_END") || rawPredUpper.endsWith("_START") || DYNAMIC_GESTURES.has(cleanPredUpper)) {
        displaySign = cleanPredUpper;
    }

    // 1. Current Sign Hero Card
    const effectiveConf = data.live_conf || (confPct / 100) || confVal || 0;
    const smoothedDisplay = getSmoothedDisplaySign(displaySign, effectiveConf);
    if (dom.signVal) dom.signVal.textContent = smoothedDisplay;

    if (data.status === "ambiguous") {
        updateMobilePipelineStatusUI("Ambiguous");
    } else if (data.status === "low_confidence") {
        updateMobilePipelineStatusUI("Detecting...");
    } else if (data.status === "aligning") {
        updateMobilePipelineStatusUI("Aligning User...");
    } else if (mobileCurrentPipelineState === "Stable") {
        updateMobilePipelineStatusUI("Stable");
    } else {
        updateMobilePipelineStatusUI(isGestureRecordingActive ? "Live tracking" : "Paused");
    }

    if (dom.confPill) {
        dom.confPill.textContent = confPct + "%";
        dom.confPill.classList.remove("high", "mid", "low");
        if (confPct >= 70) dom.confPill.classList.add("high");
        else if (confPct >= 40) dom.confPill.classList.add("mid");
        else dom.confPill.classList.add("low");
    }

    if (dom.handPill) {
        const handMap = {
            right: "✋ Right Hand",
            left: "🤚 Left Hand",
            left_ignored: "🤚 Left (Ignored)",
            both: "🙌 Dual Hands",
            none: "💤 Standby"
        };
        dom.handPill.textContent = handMap[detHand] || "💤 Standby";
    }

    if (detHand === "left_ignored") {
        clientStabilityBuffer = [];
        updateStabilityDots(0);
        return;
    }

    const now = Date.now();
    const rawPred = String(data.raw_pred || data.prediction || "").toUpperCase().trim();

    // 2. Dynamic Gesture Sequence Countdown Handling
    if (clientDynamicGesture) {
        const elapsedSec = (now - clientDynamicStartTime) / 1000.0;
        const dynamicMaxSec = (clientDynamicGesture === "NICE TO MEET YOU") ? 4.2 : 3.0;
        const remainingSec = Math.max(0, dynamicMaxSec - elapsedSec);
        const ratio = Math.max(0, Math.min(100, (remainingSec / dynamicMaxSec) * 100));

        if (dom.dynBanner) dom.dynBanner.style.display = "flex";
        if (dom.dynTitle) dom.dynTitle.textContent = clientDynamicGesture;
        if (dom.dynTimer) dom.dynTimer.textContent = remainingSec.toFixed(1) + "s";
        if (dom.dynBarFill) dom.dynBarFill.style.width = ratio + "%";

        // Transient neutral poses during hand transitions are tolerated (allow up to 1.2s of neutral)
        const isFrameNeutral = (rawPred === "NEUTRAL" || data.is_neutral || data.prediction === "—" || !data.prediction);
        if (isFrameNeutral) {
            if (!clientDynamicNeutralStart) {
                clientDynamicNeutralStart = now;
            } else if (now - clientDynamicNeutralStart > 1200) {
                clientDynamicGesture = null;
                clientDynamicNeutralStart = 0;
                if (dom.dynBanner) dom.dynBanner.style.display = "none";
                clientStabilityBuffer = [];
                updateStabilityDots(0);
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
            if (dom.dynBanner) dom.dynBanner.style.display = "none";
            if (isGestureRecordingActive) {
                appendSignToTranscript(completedGesture);
                showToast(`Dynamic Sign: ${completedGesture}`);
            }
            clientStabilityBuffer = [];
            updateStabilityDots(0);
            return;
        }

        // Cancel if timeout reached without reaching the completion pose
        if (remainingSec <= 0) {
            clientDynamicGesture = null;
            clientDynamicNeutralStart = 0;
            if (dom.dynBanner) dom.dynBanner.style.display = "none";
            clientStabilityBuffer = [];
            updateStabilityDots(0);
            return;
        }
        return;
    } else {
        if (dom.dynBanner) dom.dynBanner.style.display = "none";
    }

    // 3. Initiate dynamic gesture ONLY on a genuine _START frame with solid confidence
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
        if (dom.dynBanner) {
            dom.dynBanner.style.display = "flex";
            if (dom.dynTitle) dom.dynTitle.textContent = candidate;
            if (dom.dynTimer) dom.dynTimer.textContent = candidateMaxSec.toFixed(1) + "s";
            if (dom.dynBarFill) dom.dynBarFill.style.width = "100%";
        }
        clientStabilityBuffer = [];
        return;
    }

    // Dynamic Frame Guard:
    // Any _END or _START frame arriving while NO dynamic gesture is active is an incomplete fragment.
    // Dynamic phrases (HOW ARE YOU, NICE TO MEET YOU, J, Z) can NEVER be committed as static signs!
    if (rawPredUpper.endsWith("_END") || rawPredUpper.endsWith("_START") || DYNAMIC_GESTURES.has(cleanPredUpper)) {
        mobilePredictionWindow.push("FRAGMENT");
        if (mobilePredictionWindow.length > PREDICTION_WINDOW_MAX) mobilePredictionWindow.shift();
        updateStabilityDots(0);
        return;
    }

    // 4. Neutral / Rest Pose Gate
    const isNeutralOrIdle = (
        !data.prediction ||
        data.prediction === "—" ||
        data.is_neutral ||
        detHand === "none" ||
        (data.prediction && data.prediction.toUpperCase() === "NEUTRAL") ||
        (data.raw_pred && data.raw_pred.toUpperCase() === "NEUTRAL")
    );

    if (isNeutralOrIdle) {
        if (now >= clientDynamicCooldownUntil) {
            dynamicReadyForStart = true;
            clientLastWord = "";
        }
        mobileHasPassedThroughNeutral = true;
        if (now >= lockUntil) {
            activeSign = null;
            candidateSign = null;
            candidateCount = 0;
            confirmationBuffer = [];
            updateStabilityDots(0);
        }

        if (clientDynamicGesture) {
            clientDynamicGesture = null;
            if (dom.dynBanner) dom.dynBanner.style.display = "none";
        }

        const nowHold = performance.now();
        const inHold = (nowHold - mobileConfirmedGestureTime < GESTURE_LOCK_HOLD_MS) || (now < lockUntil);
        if (!inHold) {
            updateMobilePipelineStatusUI(isGestureRecordingActive ? "Live tracking" : "Paused");
        }
        return;
    }

    // ── State Hysteresis & Latching Prediction Handler ──
    const cleanPred = displaySign || cleanPredUpper || stripInternalSuffix(data.prediction || "");

    // Backend Margin Gate handling:
    // If backend marked the prediction status as "ambiguous" (e.g. fist cluster margin < 0.15)
    if (data.status === "ambiguous") {
        updateMobilePipelineStatusUI("Ambiguous");
        if (activeSign !== null && dom.signVal) {
            // Drop rapid override swapping and hold prior confirmed state
            dom.signVal.textContent = activeSign;
        }
        return;
    }

    if (now < lockUntil && activeSign !== null) {
        // Cooldown lockout: maintain current sign without volatile interim swapping
        if (dom.signVal) dom.signVal.textContent = activeSign;
        updateMobilePipelineStatusUI("Stable");
    } else {
        handlePrediction(cleanPred, confVal, data.status);
        const displayWord = (activeSign !== null)
            ? activeSign
            : ((mobileLastConfirmedGesture && mobileLastConfirmedGesture !== "—") ? mobileLastConfirmedGesture : (cleanPred || "—"));
        if (dom.signVal) dom.signVal.textContent = displayWord;
    }
}

function commitSignToUI(sign) {
    if (!sign || sign === "—") return;
    const now = Date.now();
    const upper = sign.toUpperCase().trim();

    if (dom.signVal) dom.signVal.textContent = sign;
    mobileLastConfirmedGesture = sign;
    mobileConfirmedGestureTime = performance.now();
    updateMobilePipelineStatusUI("Stable");
    updateStabilityDots(CONFIRM_WIN_COUNT);

    const DYNAMIC_GESTURES = new Set(["HOW ARE YOU", "NICE TO MEET YOU", "J", "Z"]);
    if (upper === "NEUTRAL" || upper === "IDLE") {
        mobileHasPassedThroughNeutral = true;
        return;
    }
    if (DYNAMIC_GESTURES.has(upper)) {
        return;
    }

    if (upper === "START" || upper === "STOP") {
        if (upper === "START" && !isGestureRecordingActive) {
            isGestureRecordingActive = true;
            updateRecordingUI();
            speakText("Recording start");
            showToast("Recording started");
        } else if (upper === "STOP" && isGestureRecordingActive) {
            isGestureRecordingActive = false;
            updateRecordingUI();
            speakCompletedSentence(clientSentence);
            showToast("Recording paused");
        }
        return;
    }

    if (upper === "SPACE") {
        if (now - clientLastTriggerTime > REPEAT_DELAY_MS) {
            if (isGestureRecordingActive) appendSignToTranscript("SPACE");
            clientLastTriggerTime = now;
        }
        return;
    }

    if (upper === "BACKSPACE" || upper === "BACK SPACE") {
        if (now - clientLastTriggerTime > REPEAT_DELAY_MS) {
            backspaceTranscriptLocal();
            clientLastTriggerTime = now;
        }
        return;
    }

    // Dynamic completion guard if YOU is detected while dynamic gesture is active
    if (upper === "YOU" && clientDynamicGesture) {
        const completedGesture = clientDynamicGesture;
        clientDynamicGesture = null;
        clientDynamicNeutralStart = 0;
        clientDynamicCooldownUntil = now + REPEAT_DELAY_MS;
        clientSuppressYouUntil = now + REPEAT_DELAY_MS;
        clientLastWord = completedGesture;
        clientLastTriggerTime = now;
        if (dom.dynBanner) dom.dynBanner.style.display = "none";
        if (isGestureRecordingActive) {
            appendSignToTranscript(completedGesture);
            showToast(`Dynamic Sign: ${completedGesture}`);
        }
        return;
    }

    if (upper === "YOU" && now < clientSuppressYouUntil) {
        return;
    }

    if (sign !== clientLastWord || (now - clientLastTriggerTime > REPEAT_DELAY_MS)) {
        if (isGestureRecordingActive) {
            appendSignToTranscript(sign);
            showToast(`Sign: ${sign}`);
        }
        clientLastWord = sign;
        clientLastTriggerTime = now;
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

    updateStabilityDots(maxWins);

    if (!winningCandidate) {
        return; // Did not meet 3-of-4 confirmation threshold yet
    }

    // 350ms hysteresis lock:
    // Once a gesture commits to the UI, lock it from being replaced by an adjacent similar sign
    // unless the new sign is held consistently past the lock duration
    if (activeSign !== null) {
        if (winningCandidate === activeSign) {
            return;
        }
        if (now < lockUntil) {
            // Locked: suppress rapid swap
            return;
        }
    }

    // Commit confirmed winning candidate
    activeSign = winningCandidate;
    commitSignToUI(activeSign);
    lockUntil = now + HYSTERESIS_LOCK_MS; // 350ms hysteresis lock
}

// ── Control Actions ─────────────────────────────────────────────────────────

function toggleRecording() {
    triggerHaptic(40);
    isGestureRecordingActive = !isGestureRecordingActive;
    updateRecordingUI();
    showToast(isGestureRecordingActive ? "Recording Started" : "Recording Paused");
    if (isGestureRecordingActive) {
        speakText("Recording start");
    } else {
        speakCompletedSentence(clientSentence);
    }
}

function clearTranscript() {
    triggerHaptic(30);
    clientSentence = "";
    clientLastWord = "";
    clientStabilityBuffer = [];
    updateTranscriptDOM();
    showToast("Transcript cleared");
}

function backspaceTranscript() {
    triggerHaptic(25);
    backspaceTranscriptLocal();
}

function speakTranscript() {
    triggerHaptic(20);
    const trimmed = clientSentence.trim();
    if (!trimmed) {
        showToast("No sentence to speak yet");
        return;
    }
    speakText(trimmed);
    showToast("Speaking sentence…");
}

function copyTranscript() {
    triggerHaptic(20);
    const trimmed = clientSentence.trim();
    if (!trimmed) {
        showToast("Nothing to copy");
        return;
    }
    navigator.clipboard.writeText(trimmed)
        .then(() => showToast("Copied to clipboard!"))
        .catch(() => showToast("Failed to copy"));
}

// ── Voice Speech Canvas State Management & Actions ─────────────────────────
function updateMobileVoiceDOM() {
    if (dom.voiceTranscriptBox) {
        const textElem = dom.voiceTranscriptBox.querySelector(".m-voice-text-content") || dom.voiceTranscriptBox;
        const trimmed = mobileVoiceSentence.trim();
        if (!trimmed) {
            textElem.textContent = "(Waiting for voice… tap Voice to speak)";
            dom.voiceTranscriptBox.classList.add("empty");
        } else {
            textElem.textContent = mobileVoiceSentence;
            dom.voiceTranscriptBox.classList.remove("empty");
        }
    }

    if (dom.voiceCharStats) {
        const trimmed = mobileVoiceSentence.trim();
        const wordArr = trimmed ? trimmed.split(/\s+/) : [];
        dom.voiceCharStats.textContent = `${wordArr.length} words · ${mobileVoiceSentence.length} chars`;
    }
}

function clearVoiceTranscript() {
    triggerHaptic(30);
    mobileVoiceSentence = "";
    updateMobileVoiceDOM();
    showToast("Voice transcript cleared");
}

function speakVoiceTranscript() {
    triggerHaptic(20);
    const trimmed = mobileVoiceSentence.trim();
    if (!trimmed) {
        showToast("No voice text to speak");
        return;
    }
    speakText(trimmed);
    showToast("Speaking voice text…");
}

function copyVoiceTranscript() {
    triggerHaptic(20);
    const trimmed = mobileVoiceSentence.trim();
    if (!trimmed) {
        showToast("Nothing to copy");
        return;
    }
    navigator.clipboard.writeText(trimmed)
        .then(() => showToast("Voice text copied to clipboard!"))
        .catch(() => showToast("Failed to copy voice text"));
}

// ── Mobile Speech-To-Text (STT) Integration ─────────────────────────────────
const MobileSpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let mobileRecognition = null;
let isMobileVoiceRecording = false;
let mobileAudioCtx = null;
let mobileMicStream = null;
let mobileAudioPCM = [];
let mobileScriptProcessor = null;
let mobileRecognizedText = "";

function writeMobileString(view, offset, string) {
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

function downsampleMobileBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
    if (inputSampleRate === outputSampleRate || inputSampleRate <= 0) return buffer;
    const ratio = inputSampleRate / outputSampleRate;
    const newLength = Math.max(1, Math.round(buffer.length / ratio));
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
        const idx = Math.min(buffer.length - 1, Math.floor(i * ratio));
        result[i] = buffer[idx] || 0;
    }
    return result;
}

function encodeMobileWAV(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    writeMobileString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeMobileString(view, 8, 'WAVE');
    writeMobileString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeMobileString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return new Blob([view], { type: 'audio/wav' });
}

async function startMobileVoiceRecording() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showToast("Microphone not supported on this browser");
            return;
        }

        mobileAudioPCM = [];
        mobileRecognizedText = "";
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        mobileAudioCtx = new AudioContextClass();

        mobileMicStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        const micSource = mobileAudioCtx.createMediaStreamSource(mobileMicStream);
        mobileScriptProcessor = mobileAudioCtx.createScriptProcessor(4096, 1, 1);
        mobileScriptProcessor.onaudioprocess = (e) => {
            if (!isMobileVoiceRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);
            mobileAudioPCM.push(new Float32Array(inputData));
        };

        micSource.connect(mobileScriptProcessor);
        const silentGain = mobileAudioCtx.createGain();
        silentGain.gain.value = 0;
        mobileScriptProcessor.connect(silentGain);
        silentGain.connect(mobileAudioCtx.destination);

        isMobileVoiceRecording = true;

        if (dom.voiceBtn) dom.voiceBtn.classList.add("recording");
        if (dom.voiceBtnText) dom.voiceBtnText.textContent = "Listening…";
        if (dom.dockVoiceBtn) dom.dockVoiceBtn.classList.add("mic-active");
        if (dom.sttPreview) {
            dom.sttPreview.style.display = "flex";
            if (dom.sttPreviewText) dom.sttPreviewText.textContent = "🎙️ Listening... speak clearly";
        }

        if (MobileSpeechRecognition) {
            try {
                if (!mobileRecognition) {
                    mobileRecognition = new MobileSpeechRecognition();
                    mobileRecognition.continuous = false;
                    mobileRecognition.interimResults = true;
                    mobileRecognition.lang = "en-US";

                    mobileRecognition.onresult = (evt) => {
                        let text = "";
                        for (let i = evt.resultIndex; i < evt.results.length; ++i) {
                            text += evt.results[i][0].transcript;
                        }
                        if (text) {
                            mobileRecognizedText = text.trim();
                            if (dom.sttPreviewText) dom.sttPreviewText.textContent = `🎙️ "${mobileRecognizedText}"`;
                        }
                    };

                    mobileRecognition.onerror = (err) => {
                        console.warn("Mobile speech recognition event:", err);
                    };
                }
                mobileRecognition.start();
            } catch (e) {
                console.warn("Native speech start error:", e);
            }
        }

        showToast("Microphone listening — speak now");
    } catch (err) {
        console.error("Microphone access error:", err);
        showToast("Could not access microphone: " + (err.message || err.name));
        stopMobileVoiceRecordingLocally();
    }
}

function stopMobileVoiceRecordingLocally() {
    isMobileVoiceRecording = false;
    if (dom.voiceBtn) dom.voiceBtn.classList.remove("recording");
    if (dom.voiceBtnText) dom.voiceBtnText.textContent = "Voice";
    if (dom.dockVoiceBtn) dom.dockVoiceBtn.classList.remove("mic-active");
    if (dom.sttPreview) dom.sttPreview.style.display = "none";

    if (mobileRecognition) {
        try { mobileRecognition.stop(); } catch (_) {}
    }
    if (mobileScriptProcessor) {
        try { mobileScriptProcessor.disconnect(); } catch (_) {}
        mobileScriptProcessor = null;
    }
    if (mobileMicStream) {
        mobileMicStream.getTracks().forEach(t => {
            try { t.stop(); } catch (_) {}
        });
        mobileMicStream = null;
    }
    if (mobileAudioCtx && mobileAudioCtx.state !== "closed") {
        try { mobileAudioCtx.close(); } catch (_) {}
        mobileAudioCtx = null;
    }
}

async function stopMobileVoiceRecording() {
    if (!isMobileVoiceRecording) return;
    isMobileVoiceRecording = false;

    if (dom.voiceBtn) dom.voiceBtn.classList.remove("recording");
    if (dom.voiceBtnText) dom.voiceBtnText.textContent = "Voice";
    if (dom.dockVoiceBtn) dom.dockVoiceBtn.classList.remove("mic-active");

    if (dom.sttPreviewText) dom.sttPreviewText.textContent = "⏳ Transcribing speech…";

    if (mobileRecognition) {
        try { mobileRecognition.stop(); } catch (_) {}
    }

    if (mobileScriptProcessor) {
        try { mobileScriptProcessor.disconnect(); } catch (_) {}
        mobileScriptProcessor = null;
    }

    const inputSampleRate = mobileAudioCtx ? mobileAudioCtx.sampleRate : 44100;

    if (mobileMicStream) {
        mobileMicStream.getTracks().forEach(t => {
            try { t.stop(); } catch (_) {}
        });
        mobileMicStream = null;
    }

    if (mobileAudioCtx && mobileAudioCtx.state !== "closed") {
        try { await mobileAudioCtx.close(); } catch (_) {}
        mobileAudioCtx = null;
    }

    // 1. If Web Speech API already transcribed text
    if (mobileRecognizedText && mobileRecognizedText.trim()) {
        const finalTxt = mobileRecognizedText.trim();
        appendSpokenTextToSentence(finalTxt);
        showToast(`Voice transcribed: "${finalTxt}"`);
        if (dom.sttPreview) dom.sttPreview.style.display = "none";
        return;
    }

    // 2. Fallback to server-side WAV transcription
    if (mobileAudioPCM.length === 0) {
        showToast("No audio recorded");
        if (dom.sttPreview) dom.sttPreview.style.display = "none";
        return;
    }

    let totalLength = 0;
    for (const chunk of mobileAudioPCM) totalLength += chunk.length;
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of mobileAudioPCM) {
        merged.set(chunk, offset);
        offset += chunk.length;
    }

    const downsampled = downsampleMobileBuffer(merged, inputSampleRate, 16000);
    const wavBlob = encodeMobileWAV(downsampled, 16000);

    const formData = new FormData();
    formData.append("audio", wavBlob, "mobile_voice.wav");
    formData.append("language", "en-US");

    try {
        const resp = await fetch(`${API_BASE_URL}/api/stt/transcribe`, {
            method: "POST",
            body: formData
        });
        const data = await resp.json();
        if (data.ok && data.text) {
            appendSpokenTextToSentence(data.text);
            showToast(`Voice transcribed: "${data.text}"`);
        } else {
            showToast(data.error || "Could not transcribe audio");
        }
    } catch (e) {
        showToast("Speech transcription error");
    } finally {
        if (dom.sttPreview) dom.sttPreview.style.display = "none";
    }
}

function appendSpokenTextToSentence(text) {
    if (!text) return;
    const clean = text.trim();
    if (!clean) return;
    if (mobileVoiceSentence.length > 0 && !mobileVoiceSentence.endsWith(" ")) {
        mobileVoiceSentence += " ";
    }
    mobileVoiceSentence += clean + " ";
    updateMobileVoiceDOM();
}

async function toggleVoiceSTT() {
    triggerHaptic(35);
    if (isMobileVoiceRecording) {
        await stopMobileVoiceRecording();
    } else {
        await startMobileVoiceRecording();
    }
}

// Switch Front / Back Camera using facingMode
async function switchCamera() {
    triggerHaptic(35);
    currentFacingMode = (currentFacingMode === "user") ? "environment" : "user";
    isMirrored = (currentFacingMode === "user");
    updateCameraMirrorDisplay();
    showToast(`Switched to ${currentFacingMode === "user" ? "Front (Selfie)" : "Rear (Environment)"} camera`);
    await startMobileWebcam();
}

// Toggle Camera Mirror
function toggleCameraMirror() {
    triggerHaptic(20);
    isMirrored = !isMirrored;
    updateCameraMirrorDisplay();
    showToast(isMirrored ? "Feed mirrored" : "Feed normal");
}

// Dominant Hand Toggle
function setDominantHand(hand) {
    triggerHaptic(25);
    fetch(`${API_BASE_URL}/set_dominant_hand`, {
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

// ── Dictionary Bottom Sheet, Filter & Dual-View Skeleton Visualizer ─────────
const MOBILE_FINGER_PLACEMENT_BADGES = {
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

let mobileDictData = [];
let mobileDictMap = {};
let currentMobileDictSign = null;
let mobileDictAnimRunning = true;
let mobileDictAnimSpeed = 1.0;
let mobileDictKeyframeIdx = 0;
let mobileDictLastTimestamp = 0;
let mobileDictAnimReqId = null;

async function loadMobileDictionaryData() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/dictionary`);
        if (res.ok) {
            const data = await res.json();
            if (data.ok && data.dictionary) {
                mobileDictData = data.dictionary;
                mobileDictMap = {};
                for (const item of mobileDictData) {
                    mobileDictMap[item.id.toUpperCase()] = item;
                }
            }
        }
    } catch (e) {
        console.warn("Could not load mobile dictionary:", e);
    }
}

function openDictionary() {
    triggerHaptic(25);
    if (dom.dictModal) {
        dom.dictModal.classList.add("show");
        if (dom.dictSearch) {
            dom.dictSearch.value = "";
            filterDictionary("");
        }
    }
    if (mobileDictData.length === 0) {
        loadMobileDictionaryData();
    }
}

function closeDictionary() {
    if (dom.dictModal) dom.dictModal.classList.remove("show");
    if (mobileDictAnimReqId) {
        cancelAnimationFrame(mobileDictAnimReqId);
        mobileDictAnimReqId = null;
    }
}

function selectMobileDictSign(signName) {
    if (!signName) return;
    const clean = signName.toUpperCase().trim();
    let sign = mobileDictMap[clean];
    if (!sign) {
        for (const k in mobileDictMap) {
            if (k.includes(clean) || clean.includes(k)) {
                sign = mobileDictMap[k];
                break;
            }
        }
    }
    if (!sign) return;

    currentMobileDictSign = sign;
    mobileDictKeyframeIdx = 0;

    const viewer = document.getElementById("m-dict-viewer");
    if (viewer) viewer.style.display = "block";

    const titleEl = document.getElementById("m-dict-title");
    if (titleEl) titleEl.textContent = `Sign "${sign.name}"`;

    const catBadge = document.getElementById("m-dict-cat-badge");
    if (catBadge) {
        const isDyn = (sign.category === "dynamic");
        catBadge.textContent = isDyn ? "DYNAMIC" : "STATIC";
        catBadge.style.color = isDyn ? "#ffb020" : "#10b981";
    }

    const placementBadge = document.getElementById("m-dict-placement-badge");
    if (placementBadge) {
        placementBadge.textContent = MOBILE_FINGER_PLACEMENT_BADGES[clean] || "Finger alignment verified";
    }

    const descEl = document.getElementById("m-dict-desc");
    if (descEl) descEl.textContent = sign.description || "Canonical gesture posture.";

    startMobileDictAnimLoop();
    renderMobileDictionaryCanvas();
}

function toggleMobileDictAnim() {
    mobileDictAnimRunning = !mobileDictAnimRunning;
    const btn = document.getElementById("m-dict-play-pause");
    if (btn) btn.textContent = mobileDictAnimRunning ? "⏸ Pause" : "▶ Play";
}

function toggleMobileDictSpeed() {
    mobileDictAnimSpeed = (mobileDictAnimSpeed === 1.0) ? 0.5 : 1.0;
    const btn = document.getElementById("m-dict-speed");
    if (btn) btn.textContent = `⚡ Speed: ${mobileDictAnimSpeed}x`;
}

function restartMobileDictAnim() {
    mobileDictKeyframeIdx = 0;
    renderMobileDictionaryCanvas();
}

function startMobileDictAnimLoop() {
    if (mobileDictAnimReqId) cancelAnimationFrame(mobileDictAnimReqId);

    const loop = (timestamp) => {
        if (!mobileDictLastTimestamp) mobileDictLastTimestamp = timestamp;
        const elapsed = timestamp - mobileDictLastTimestamp;
        const frameInterval = 75 / mobileDictAnimSpeed;

        if (mobileDictAnimRunning && elapsed >= frameInterval) {
            mobileDictLastTimestamp = timestamp;
            if (currentMobileDictSign && currentMobileDictSign.keyframes && currentMobileDictSign.keyframes.length > 0) {
                mobileDictKeyframeIdx = (mobileDictKeyframeIdx + 1) % currentMobileDictSign.keyframes.length;
            }
            renderMobileDictionaryCanvas();
        } else if (!mobileDictAnimRunning) {
            renderMobileDictionaryCanvas();
        }

        mobileDictAnimReqId = requestAnimationFrame(loop);
    };

    mobileDictAnimReqId = requestAnimationFrame(loop);
}

function renderMobileDictionaryCanvas() {
    const canvas = document.getElementById("dictionary-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#070b16";
    ctx.fillRect(0, 0, W, H);

    const splitX = Math.floor(W * 0.38);

    // Left Panel: Upper Body Context
    ctx.save();
    ctx.font = '600 9px monospace';
    ctx.fillStyle = 'rgba(0, 240, 255, 0.8)';
    ctx.textAlign = 'center';
    ctx.fillText('BODY CONTEXT', splitX / 2, 16);

    const cx = splitX / 2;
    const headY = 38;
    const neckY = 54;
    const shoulderY = 64;
    const spineBottomY = 135;

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, headY, 13, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = '#00f0ff';
    ctx.beginPath();
    ctx.moveTo(cx - 6, headY);
    ctx.lineTo(cx + 6, headY);
    ctx.stroke();

    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(cx, headY + 13);
    ctx.lineTo(cx, neckY);
    ctx.stroke();

    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    ctx.moveTo(cx - 30, shoulderY);
    ctx.lineTo(cx + 30, shoulderY);
    ctx.stroke();

    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(cx, neckY);
    ctx.lineTo(cx, spineBottomY);
    ctx.stroke();

    // Right Arm
    let rElbowX = cx + 30 + 16;
    let rElbowY = shoulderY + 30;
    let rWristX = cx + 30 + 8;
    let rWristY = shoulderY + 62;

    if (currentMobileDictSign && currentMobileDictSign.category === 'dynamic') {
        rWristX += Math.sin((mobileDictKeyframeIdx / 12) * Math.PI * 2) * 5;
    }

    ctx.strokeStyle = '#00E5FF';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(cx + 30, shoulderY);
    ctx.lineTo(rElbowX, rElbowY);
    ctx.lineTo(rWristX, rWristY);
    ctx.stroke();

    // Left Arm resting at side
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(cx - 30, shoulderY);
    ctx.lineTo(cx - 36, shoulderY + 30);
    ctx.lineTo(cx - 34, shoulderY + 62);
    ctx.stroke();

    ctx.restore();

    // Divider
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(splitX, 8);
    ctx.lineTo(splitX, H - 8);
    ctx.stroke();
    ctx.restore();

    // Right Panel: Zoomed Hand Skeleton (21 Joints)
    ctx.save();
    const rightCenterX = splitX + (W - splitX) / 2;
    ctx.font = '600 9px monospace';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.textAlign = 'center';
    ctx.fillText('HAND SKELETON (21 PTS)', rightCenterX, 16);

    let points = null;
    if (currentMobileDictSign) {
        if (currentMobileDictSign.category === 'dynamic' && currentMobileDictSign.keyframes && currentMobileDictSign.keyframes.length > 0) {
            points = currentMobileDictSign.keyframes[mobileDictKeyframeIdx] || currentMobileDictSign.landmarks;
        } else {
            points = currentMobileDictSign.landmarks;
        }
    }

    if (points && points.length >= 21) {
        const JOINT_COLORS = {
            0: '#FFFFFF',
            1: '#FF9800', 2: '#FF9800', 3: '#FF9800', 4: '#FF9800',
            5: '#00E5FF', 6: '#00E5FF', 7: '#00E5FF', 8: '#00E5FF',
            9: '#00E676', 10: '#00E676', 11: '#00E676', 12: '#00E676',
            13: '#FFEA00', 14: '#FFEA00', 15: '#FFEA00', 16: '#FFEA00',
            17: '#FF4081', 18: '#FF4081', 19: '#FF4081', 20: '#FF4081'
        };

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (let i = 0; i < 21; i++) {
            const px = Number(points[i].x !== undefined ? points[i].x : points[i][0]);
            const py = Number(points[i].y !== undefined ? points[i].y : points[i][1]);
            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
        }

        const spanX = Math.max(0.01, maxX - minX);
        const spanY = Math.max(0.01, maxY - minY);
        const availW = W - splitX - 25;
        const availH = H - 45;
        const scale = Math.min(availW / spanX, availH / spanY) * 0.80;

        const targetCenterX = splitX + (W - splitX) / 2;
        const targetCenterY = 24 + availH / 2;
        const centerNormX = (minX + maxX) / 2;
        const centerNormY = (minY + maxY) / 2;

        const screenPts = points.map(p => {
            const px = Number(p.x !== undefined ? p.x : p[0]);
            const py = Number(p.y !== undefined ? p.y : p[1]);
            return {
                x: targetCenterX + (px - centerNormX) * scale,
                y: targetCenterY + (py - centerNormY) * scale
            };
        });

        // Bones
        ctx.lineWidth = 2.0;
        ctx.lineCap = 'round';
        for (const [i, j] of HAND_CONNECTIONS) {
            const p0 = screenPts[i];
            const p1 = screenPts[j];
            if (!p0 || !p1) continue;

            let boneColor = 'rgba(255, 255, 255, 0.4)';
            if (i >= 1 && j <= 4) boneColor = 'rgba(255, 152, 0, 0.7)';
            else if (i >= 5 && j <= 8) boneColor = 'rgba(0, 229, 255, 0.7)';
            else if (i >= 9 && j <= 12) boneColor = 'rgba(0, 230, 118, 0.7)';
            else if (i >= 13 && j <= 16) boneColor = 'rgba(255, 234, 0, 0.7)';
            else if (i >= 17 && j <= 20) boneColor = 'rgba(255, 64, 129, 0.7)';

            ctx.strokeStyle = boneColor;
            ctx.beginPath();
            ctx.moveTo(p0.x, p0.y);
            ctx.lineTo(p1.x, p1.y);
            ctx.stroke();
        }

        // Joints
        const tips = new Set([4, 8, 12, 16, 20]);
        for (let k = 0; k < 21; k++) {
            const sp = screenPts[k];
            if (!sp) continue;
            const color = JOINT_COLORS[k] || '#FFFFFF';

            ctx.beginPath();
            if (tips.has(k)) {
                ctx.arc(sp.x, sp.y, 5.0, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 1.2;
                ctx.stroke();
            } else if (k === 0) {
                ctx.arc(sp.x, sp.y, 4.0, 0, Math.PI * 2);
                ctx.fillStyle = '#FFFFFF';
                ctx.fill();
            } else {
                ctx.arc(sp.x, sp.y, 2.8, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
            }
        }
    }

    ctx.restore();
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

// Attach tap feedback and dual-view selection to gesture chips
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".m-gesture-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            triggerHaptic(20);
            const name = chip.getAttribute("data-name") || chip.textContent;
            selectMobileDictSign(name);
        });
    });
});

// Initialize Client Webcam & Isolated UI
updateCameraMirrorDisplay();
updateRecordingUI();
updateTranscriptDOM();
updateMobileVoiceDOM();
initMobileServerNegotiator();
startMobileWebcam();
loadMobileDictionaryData();

