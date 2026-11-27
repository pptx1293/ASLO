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
const CLOUD_BACKEND_URL = "https://aslo-1.onrender.com";
const isLocalHost = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
const LOCAL_IPV4_ORIGIN = (window.location.hostname === '127.0.0.1' && window.location.port)
    ? ''
    : `http://127.0.0.1:${window.location.port || '5000'}`;
const API_BASE_URL = isLocalHost
    ? LOCAL_IPV4_ORIGIN
    : (window.location.hostname.includes('onrender.com') ? '' : CLOUD_BACKEND_URL);

let currentInferenceUrl = API_BASE_URL;
let serverMode = localStorage.getItem("aslo_server_mode") || "auto";
let isLocalAlive = false;

async function checkLocalServerAlive() {
    try {
        const controller = new AbortController();
        const tid = setTimeout(() => controller.abort(), 600);
        const pingBase = LOCAL_IPV4_ORIGIN || "http://127.0.0.1:5000";
        const res = await fetch(`${pingBase}/ping`, {
            method: "GET",
            mode: "cors",
            signal: controller.signal
        });
        clearTimeout(tid);
        if (res.ok) {
            const data = await res.json();
            return Boolean(data && data.status === "online");
        }
    } catch (_) { }
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

    const fallbackCloudUrl = (window.location.hostname.includes('onrender.com') || isLocalHost) ? '' : CLOUD_BACKEND_URL;

    if (serverMode === "local") {
        if (isLocalAlive || isLocalHost) {
            currentInferenceUrl = isLocalHost ? LOCAL_IPV4_ORIGIN : "http://127.0.0.1:5000";
            if (pill) {
                pill.textContent = "Local (0ms)";
                pill.style.background = "#10b981";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "none";
        } else {
            currentInferenceUrl = fallbackCloudUrl || CLOUD_BACKEND_URL;
            if (pill) {
                pill.textContent = "Offline (Cloud)";
                pill.style.background = "#ef4444";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "inline";
        }
    } else if (serverMode === "cloud") {
        currentInferenceUrl = isLocalHost ? CLOUD_BACKEND_URL : (window.location.hostname.includes('onrender.com') ? '' : CLOUD_BACKEND_URL);
        if (pill) {
            pill.textContent = "Cloud";
            pill.style.background = "#0284c7";
            pill.style.color = "#fff";
        }
        if (downloadLink) downloadLink.style.display = "none";
    } else {
        // "auto" mode: use explicit IPv4 127.0.0.1 so Windows never hits 2s IPv6 [::1] SYN timeouts on localhost
        if (isLocalHost || isLocalAlive) {
            currentInferenceUrl = isLocalHost ? LOCAL_IPV4_ORIGIN : "http://127.0.0.1:5000";
            if (pill) {
                pill.textContent = "Auto: Local (0ms)";
                pill.style.background = "#10b981";
                pill.style.color = "#fff";
            }
            if (downloadLink) downloadLink.style.display = "none";
        } else {
            currentInferenceUrl = window.location.hostname.includes('onrender.com') ? '' : CLOUD_BACKEND_URL;
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
    setInterval(updateMobileServerNegotiation, 4000);
}

// ── State Hysteresis & Latching Constants & State ──────────────────────────
let activeSign = null;
let candidateSign = null;
let candidateCount = 0;
const COMMIT_THRESHOLD = 3; // Fast 3-frame confirmation (~130ms)
const CONFIRM_BUFFER_SIZE = 4;
const CONFIRM_WIN_COUNT = 3;
let confirmationBuffer = [];
const HYSTERESIS_LOCK_MS = 250; // Responsive 250ms hysteresis lock against adjacent sign overriding
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
const STABILITY_REQUIRED_COUNT = 3; // Fast 3-frame confirmation (~130ms)
const BUFFER_MAX_LEN = 4;

let isSoundMuted = localStorage.getItem("aslo_mobile_sound_muted") === "true";
let stabilitySegmentsCount = 5;
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
        } catch (_) { }
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
    constructor(minCutoff = 3.0, beta = 20.0, dCutoff = 1.5) {
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
 * Resolves the signer's actual physical hand from MediaPipe's handedness metadata.
 * Raw <video> frames passed into MediaPipe are always unflipped (CSS scaleX only flips display),
 * so MediaPipe always reports inverted handedness ("Left" for signer's Right, "Right" for signer's Left)
 * on BOTH front (mirror) and rear (true/environment) cameras.
 */
function getActualPhysicalHand(handednessObj) {
    if (!handednessObj) return "Right";
    const raw = (handednessObj.label || (handednessObj.classification && handednessObj.classification[0] && handednessObj.classification[0].label) || "").trim();
    if (raw === "Left") return "Right";
    if (raw === "Right") return "Left";
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

        let minX = wx, maxX = wx, minY = wy, maxY = wy;
        for (let k = 0; k < lms.length; k++) {
            const pk = lms[k];
            const px = Number(pk.x !== undefined ? pk.x : pk[0]);
            const py = Number(pk.y !== undefined ? pk.y : pk[1]);
            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
        }
        const bboxDiag = Math.hypot(maxX - minX, maxY - minY);
        const scale = Math.max(Math.hypot(wx - mx, wy - my), bboxDiag * 0.45);
        const rawH = multiHandedness && multiHandedness[idx];
        const physicalHand = getActualPhysicalHand(rawH);
        const distFromCenter = Math.abs(wx - 0.5);
        const inCenterRegion = (wx >= 0.08 && wx <= 0.92);

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

    // Discard background hands far from the camera (< 45% of the closest hand's scale)
    const valid = candidates.filter(c => c.scale >= 0.45 * maxScale);
    if (valid.length === 0) {
        return { hands: [], status: "aligning" };
    }

    const pool = [...valid];

    // Sort by CLOSEST TO CAMERA (largest hand scale = closest to lens), preferring active raised hands over bottom resting hands
    pool.sort((a, b) => {
        const aRest = a.wrist.y >= 0.85 ? 1 : 0;
        const bRest = b.wrist.y >= 0.85 ? 1 : 0;
        if (aRest !== bRest) return aRest - bRest;
        return b.scale - a.scale;
    });

    const primary = pool[0];

    // Single hand passed filter:
    if (pool.length === 1 || valid.length === 1) {
        return {
            hands: [primary],
            status: primary.inCenterRegion ? "isolated" : "aligning"
        };
    }

    // Two or more hands: pick the closest partner hand belonging to the SAME closest user (matching depth/scale)
    let pairedSecond = null;
    for (let j = 1; j < pool.length; j++) {
        const cand = pool[j];

        const dy = Math.abs(primary.wrist.y - cand.wrist.y);
        const dx = Math.abs(primary.wrist.x - cand.wrist.x);
        const wristDist = Math.hypot(primary.wrist.x - cand.wrist.x, primary.wrist.y - cand.wrist.y);

        if (cand.scale < 0.50 * primary.scale) continue;
        if (dy >= 0.42 || dx >= 0.68) continue;
        // Reject pairing a raised active signing hand with a passive resting hand at the bottom of the frame
        if ((primary.wrist.y < 0.72 && cand.wrist.y >= 0.85) || (cand.wrist.y < 0.72 && primary.wrist.y >= 0.85)) continue;
        if ((primary.wrist.x < 0.15 && cand.wrist.x > 0.85) || (cand.wrist.x < 0.15 && primary.wrist.x > 0.85)) continue;
        if (wristDist < 0.04) continue; // Only discard exact identical duplicate detections

        // Handedness pairing: opposite or spatial assignment if MediaPipe misclassified identical handedness
        let isOpposite = (
            (primary.physicalHand === "Right" && cand.physicalHand === "Left") ||
            (primary.physicalHand === "Left" && cand.physicalHand === "Right")
        );
        if (!isOpposite) {
            // Raw MediaPipe coordinates are ALWAYS unflipped regardless of CSS mirror display:
            // facing the camera, the signer's physical Right hand has smaller wrist.x than Left.
            if (cand.wrist.x < primary.wrist.x) {
                cand.physicalHand = "Right";
                primary.physicalHand = "Left";
            } else {
                cand.physicalHand = "Left";
                primary.physicalHand = "Right";
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

const mobileLandmarkEMASmoother = new MobileLandmarkEMASmoother(0.85);

// ── Mobile 25-Frame Rolling Sequence Buffer & Unified 3-MLP Cooperative Mode ──
const MOBILE_DYNAMIC_SEQ_LEN = 25;
let mobileSequenceBuffer = [];
let mobileMotionSpeed = 0.0;
let mobilePrevWristPos = null;
let mobilePrevWristTime = 0;
let mobileGestureRoutingMode = "auto";
try { localStorage.removeItem("aslo_gesture_routing_mode"); } catch (_) {}

function setMobileGestureRoutingMode(mode) {
    mobileGestureRoutingMode = "auto";
}

function updateMobileRollingSequence(landmarks, timestamp = performance.now()) {
    if (!landmarks || landmarks.length < 21) {
        mobileSequenceBuffer = [];
        mobileMotionSpeed = 0.0;
        mobilePrevWristPos = null;
        return;
    }

    if (mobilePrevWristTime > 0 && (timestamp - mobilePrevWristTime) < 35 && mobileSequenceBuffer.length >= 2) {
        return;
    }

    const framePts = toMobileLandmarkArray(landmarks);
    mobileSequenceBuffer.push(framePts);
    if (mobileSequenceBuffer.length > MOBILE_DYNAMIC_SEQ_LEN) {
        mobileSequenceBuffer.shift();
    }

    const w0 = framePts[0];
    const m9 = framePts[9];
    const i8 = framePts[8];
    const p20 = framePts[20];
    const scale = Math.max(0.05, Math.hypot(w0[0] - m9[0], w0[1] - m9[1]));

    if (mobilePrevWristPos && mobileSequenceBuffer.length >= 2) {
        const prevFrame = mobileSequenceBuffer[mobileSequenceBuffer.length - 2];
        const dw = Math.hypot(w0[0] - prevFrame[0][0], w0[1] - prevFrame[0][1]) / scale;
        const di = Math.hypot(i8[0] - prevFrame[8][0], i8[1] - prevFrame[8][1]) / scale;
        const dp = Math.hypot(p20[0] - prevFrame[20][0], p20[1] - prevFrame[20][1]) / scale;
        const instSpeed = Math.max(dw, di * 0.7, dp * 0.7);
        mobileMotionSpeed = 0.6 * instSpeed + 0.4 * mobileMotionSpeed;
    } else {
        mobileMotionSpeed = 0.0;
    }

    mobilePrevWristPos = { x: w0[0], y: w0[1] };
    mobilePrevWristTime = timestamp;
    if (dom.motionLabel) {
        dom.motionLabel.textContent = mobileMotionSpeed >= 0.035 ? "DYNAMIC" : "STATIC";
    }
}

function updateMobileModelTypeBadgeUI(gestureType, modelSource = "", specialistUsed = "") {
    const badge = document.getElementById("m-model-type-badge");
    if (!badge) return;
    const gType = String(gestureType || "").toLowerCase();
    const spec = String(specialistUsed || "").trim();
    if (gType === "dynamic") {
        badge.textContent = "DYNAMIC";
        badge.style.background = "rgba(245, 158, 11, 0.22)";
        badge.style.color = "#fbbf24";
        badge.style.borderColor = "rgba(245, 158, 11, 0.5)";
    } else if (gType === "static_words") {
        badge.textContent = "WORD";
        badge.style.background = "rgba(168, 85, 247, 0.22)";
        badge.style.color = "#c084fc";
        badge.style.borderColor = "rgba(168, 85, 247, 0.5)";
    } else if (gType === "static_asl") {
        if (spec && spec !== "None" && spec !== "ASL Router") {
            badge.textContent = `ASL • ${spec}`;
        } else {
            badge.textContent = modelSource.includes("fist") ? "ASL • Fist Specialist" : "ASL";
        }
        badge.style.background = "rgba(16, 185, 129, 0.22)";
        badge.style.color = "#34d399";
        badge.style.borderColor = "rgba(16, 185, 129, 0.5)";
    } else {
        badge.textContent = mobileGestureRoutingMode.toUpperCase();
        badge.style.background = "rgba(56, 189, 248, 0.18)";
        badge.style.color = "#38bdf8";
        badge.style.borderColor = "rgba(56, 189, 248, 0.4)";
    }
}

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
    constructor(minCutoff = 3.0, beta = 20.0, dCutoff = 1.5) {
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

        const pinkyUp = (p20.y < p18.y) && (p20.y < p17.y - 0.02 * scale);
        const fingersCurled = (landmarks[8].y > landmarks[6].y - 0.03) &&
            (landmarks[12].y > landmarks[10].y - 0.03) &&
            (landmarks[16].y > landmarks[14].y - 0.03);

        const d45 = Math.hypot(landmarks[4].x - landmarks[5].x, landmarks[4].y - landmarks[5].y);
        const thumbFlare = d45 / scale;

        return pinkyUp && (thumbFlare < 0.35 || (thumbFlare < 0.38 && fingersCurled));
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

        // 20-frame FIFO sliding buffer
        if (this.buffer.length > 20) {
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
            if (recentI && vy > 0.10 && (now - this.lastTriggerTime > 1000)) {
                this.state = "SWOOPING";
                this.startTime = now;
                this.isMoving = true;
            }
        }

        if (this.state === "SWOOPING") {
            const duration = now - this.startTime;
            if (duration > 1000) {
                this.reset();
                return { state: "IDLE", committed: false, suppressStatic: false };
            }

            const motionFrames = this.buffer.filter(b => b.t >= this.startTime - 120);
            if (motionFrames.length >= 6 && duration >= 350 && duration <= 1000) {
                const minY = Math.min(...motionFrames.map(b => b.y));
                const maxY = Math.max(...motionFrames.map(b => b.y));
                const lastY = motionFrames[motionFrames.length - 1].y;

                const downwardTravel = maxY - minY;
                const upwardRecovery = maxY - lastY; // Delta y < 0 (curving back upward)

                if (downwardTravel >= 0.040 && upwardRecovery >= 0.018) {
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

const mobileStabilizer = new MobileSpatialHandTracker(3.0, 20.0, 1.5);
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
const MOBILE_PREDICT_INTERVAL = 60; // ~16.5 Hz non-blocking inference stream
let mobileDroppedFramesCount = 0;
let mobileCanvasDirty = false;
let isMobileMpWarmupRunning = false;
let isMobileMpWarmupComplete = false;
let mobileMpWarmupPromise = null;

async function prewarmMobileMediaPipe() {
    if (isMobileMpWarmupComplete || !mobileHands) return;
    if (mobileMpWarmupPromise) return mobileMpWarmupPromise;
    mobileMpWarmupPromise = (async () => {
        isMobileMpWarmupRunning = true;
        try {
            if (typeof mobileHands.initialize === "function") {
                await mobileHands.initialize();
            }
            await new Promise((resolve) => {
                const img = new Image();
                img.crossOrigin = "anonymous";
                const timer = setTimeout(resolve, 2500);
                img.onload = async () => {
                    clearTimeout(timer);
                    try { await mobileHands.send({ image: img }); } catch (_) { }
                    resolve();
                };
                img.onerror = () => {
                    clearTimeout(timer);
                    resolve();
                };
                const base = currentInferenceUrl || "";
                img.src = `${base}/warmup_hand.jpg`;
            });
        } catch (_) {
        } finally {
            isMobileMpWarmupRunning = false;
            isMobileMpWarmupComplete = true;
        }
    })();
    return mobileMpWarmupPromise;
}

function initMobileMediaPipe() {
    if (mobileHands) return mobileHands;
    if (typeof Hands === "undefined") return null;

    try {
        mobileHands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        // Fixed options set ONCE (avoids expensive WASM graph rebuilds inside onResults;
        // minTrackingConfidence 0.50 retains lightweight ROI crop mode)
        mobileHands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.50,
            minTrackingConfidence: 0.50
        });

        mobileHands.onResults((results) => {
            if (isMobileMpWarmupRunning) return;
            const canvas = dom.landmarkCanvas;
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            const hasRawHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
            if (!hasRawHands) {
                mobileDroppedFramesCount++;
            } else {
                mobileDroppedFramesCount = 0;
            }

            const vw = (dom.webcam && dom.webcam.videoWidth) || 640;
            const vh = (dom.webcam && dom.webcam.videoHeight) || 480;
            const targetW = 480;
            const targetH = Math.max(240, Math.round((480 * vh) / Math.max(1, vw)));
            if (canvas.width !== targetW || canvas.height !== targetH) {
                canvas.width = targetW;
                canvas.height = targetH;
            }

            if (hasRawHands || mobileCanvasDirty) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                mobileCanvasDirty = hasRawHands;
            }

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

            // Continuously record 25-frame trajectory at ~26 FPS directly from mobileHands.onResults
            if (currentMobileStabilizedHands.length > 0) {
                const rightStab = currentMobileStabilizedHands.find(s => s.label === "Right") || currentMobileStabilizedHands[0];
                if (rightStab && rightStab.landmarks) {
                    updateMobileRollingSequence(rightStab.landmarks, now);
                }
            }

            if (currentMobileStabilizedHands.length === 0) {
                mobileJTrajectoryTracker.reset();
            }

            // 2. Throttled backend landmark prediction (non-blocking lock; dynamic J uses unified J_START -> Timer -> J_END flow)
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

// ── Static "I" vs Dynamic "J" ──
// "I" uses the same handshape as the first frame of "J" (J_START).
// Repositioning the hand (raising up, shifting sideways, or holding still) stays static "I".
// Dynamic "J" only starts when the pinky & wrist follow the downward/curved J drawing pattern.
let clientIPoseLastSeen = 0;
let clientLastICommitTime = 0;

function isDrawingJPattern(seqBuf, rawPredUpper) {
    if (!seqBuf || seqBuf.length < 5) return false;
    const n = seqBuf.length;
    const cur = seqBuf[n - 1];
    const lookback = Math.min(n - 1, 12);
    const startIdx = n - 1 - lookback;
    const scale = Math.max(0.05, Math.hypot(cur[0][0] - cur[9][0], cur[0][1] - cur[9][1]));

    let minPinkyY = cur[20][1];
    let minPinkyIdx = n - 1;
    for (let i = startIdx; i < n; i++) {
        if (seqBuf[i][20][1] < minPinkyY) {
            minPinkyY = seqBuf[i][20][1];
            minPinkyIdx = i;
        }
    }
    const ref = seqBuf[startIdx];
    const peak = seqBuf[minPinkyIdx];

    const pinkyDrop = (cur[20][1] - peak[20][1]) / scale;
    const wristDrop = (cur[0][1] - peak[0][1]) / scale;
    const netDy20 = (cur[20][1] - ref[20][1]) / scale;
    const netDx20 = Math.abs(cur[20][0] - ref[20][0]) / scale;
    const prev3 = seqBuf[Math.max(0, n - 4)];
    const recentDy20 = (cur[20][1] - prev3[20][1]) / scale;

    if (rawPredUpper === "J_END") {
        return (pinkyDrop >= 0.18 || wristDrop >= 0.14 || Math.hypot(netDx20, netDy20) >= 0.25);
    }

    return (
        pinkyDrop >= 0.32
        && wristDrop >= 0.16
        && netDy20 >= 0.24
        && recentDy20 >= 0.06
        && netDy20 >= 0.55 * netDx20
    );
}

function isDrawingZPattern(seqBuf) {
    if (!seqBuf || seqBuf.length < 5) return false;
    const n = seqBuf.length;
    const cur = seqBuf[n - 1];
    const lookback = Math.min(n - 1, 12);
    const startIdx = n - 1 - lookback;
    const scale = Math.max(0.05, Math.hypot(cur[0][0] - cur[9][0], cur[0][1] - cur[9][1]));
    const ref = seqBuf[startIdx];
    const prev3 = seqBuf[Math.max(0, n - 4)];

    const netDx8 = Math.abs(cur[8][0] - ref[8][0]) / scale;
    const netDy8 = (cur[8][1] - ref[8][1]) / scale;
    const recentDx8 = Math.abs(cur[8][0] - prev3[8][0]) / scale;
    return (
        netDx8 >= 0.30
        && recentDx8 >= 0.06
        && netDy8 > -0.35
        && netDx8 >= 0.65 * Math.abs(netDy8)
    );
}

function retractRecentStaticI(now) {
    if (clientLastICommitTime && (now - clientLastICommitTime) < 3500 && clientSentence.endsWith("I")) {
        clientSentence = clientSentence.slice(0, -1);
        updateTranscriptDOM();
    }
    clientLastICommitTime = 0;
}

function appendSignToTranscript(sign) {
    if (!sign || sign === "—" || sign.toUpperCase() === "NEUTRAL" || sign.toUpperCase() === "IDLE") return;
    const clean = stripInternalSuffix(sign).trim();
    if (!clean || clean.toUpperCase() === "NEUTRAL" || clean.toUpperCase() === "IDLE") return;

    const trimmedUpper = clientSentence.trim().toUpperCase();
    const cleanUpper = clean.toUpperCase();
    const now = Date.now();

    // Prevent duplicate consecutive multi-word phrases within 1.4s cooldown
    const isDynamicPhrase = cleanUpper.includes(" ") || cleanUpper === "HOW ARE YOU" || cleanUpper === "NICE TO MEET YOU" || cleanUpper === "HELLO";
    if (isDynamicPhrase) {
        if (cleanUpper === clientLastDynamicCommitted.gesture && (now - clientLastDynamicCommitted.time < 1400)) {
            return;
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
            if (cleanUpper === "I") clientLastICommitTime = now;
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
        mobileSequenceBuffer = [];
        mobileMotionSpeed = 0.0;
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
            updateMobileRollingSequence(emaSmoothed);
            const landmarks = toMobileLandmarkArray(emaSmoothed);
            const includeSeq = (mobileSequenceBuffer.length >= 8) && (mobileMotionSpeed >= 0.020 || clientDynamicGesture !== null);

            const payload = {
                landmarks: landmarks,
                handedness: "Right",
                hand_type: "right",
                sequence_buffer: includeSeq ? mobileSequenceBuffer : [],
                motion_speed: Number(mobileMotionSpeed.toFixed(4)),
                mode: mobileGestureRoutingMode,
                is_mirrored: true,
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

            // Reuse stabilizedHands from mobileStabilizer.process() to prevent track-swapping velocity spikes
            const stabRight = stabilizedHands && stabilizedHands.find(s => s.label === "Right");
            const stabLeft = stabilizedHands && stabilizedHands.find(s => s.label === "Left");
            const rSmoothedOneEuro = stabRight ? stabRight.landmarks : rHandObj.rawLandmarks;
            const lSmoothedOneEuro = stabLeft ? stabLeft.landmarks : lHandObj.rawLandmarks;

            const rEma = mobileLandmarkEMASmoother.smooth("Right", rSmoothedOneEuro);
            const lEma = mobileLandmarkEMASmoother.smooth("Left", lSmoothedOneEuro);
            updateMobileRollingSequence(rEma);

            const rPts = toMobileLandmarkArray(rEma);
            const lPts = toMobileLandmarkArray(lEma);
            const includeSeq = (mobileSequenceBuffer.length >= 8) && (mobileMotionSpeed >= 0.020 || clientDynamicGesture !== null);

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
                sequence_buffer: includeSeq ? mobileSequenceBuffer : [],
                motion_speed: Number(mobileMotionSpeed.toFixed(4)),
                mode: mobileGestureRoutingMode,
                is_mirrored: true,
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
let mobileLoopToken = 0;
let mobileMpInputCanvas = null;
let mobileMpInputCtx = null;
const MOBILE_MP_INPUT_WIDTH = 384;
const MOBILE_MP_ACTIVE_MS = 65; // ~15.5 FPS when hand(s) in view
const MOBILE_MP_IDLE_MS = 135;  // ~7.5 FPS when 0 hands in view
const MOBILE_MP_WATCHDOG_MS = 8000; // Catastrophe-only WASM lock timeout

async function startMobileWebcam() {
    try {
        initMobileMediaPipe();
        const mobileWarmupTask = prewarmMobileMediaPipe();

        // 1. Cancel previous frame loop
        if (mobileFrameLoopId) {
            cancelAnimationFrame(mobileFrameLoopId);
            mobileFrameLoopId = null;
        }

        // 2. Stop previous MediaPipe Camera helper if any
        if (mobileCamera) {
            try { await mobileCamera.stop(); } catch (e) { }
            mobileCamera = null;
        }

        // 3. Stop previous stream tracks cleanly to release sensor
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => {
                try { t.stop(); } catch (_) { }
            });
            webcamStream = null;
        }

        if (dom.webcam) {
            try { dom.webcam.pause(); } catch (_) { }
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
        await mobileWarmupTask;

        isWebcamActive = true;
        updateCameraMirrorDisplay();
        if (dom.camFallback) dom.camFallback.style.display = "none";

        if (!mobileMpInputCanvas) {
            mobileMpInputCanvas = document.createElement("canvas");
            mobileMpInputCanvas.width = MOBILE_MP_INPUT_WIDTH;
            mobileMpInputCanvas.height = 360;
            mobileMpInputCtx = mobileMpInputCanvas.getContext("2d", { alpha: false, desynchronized: true });
        }

        // 5. Self-healing rAF loop + downscaled MediaPipe input (prevents 0-FPS freezes & lag)
        const myMobileToken = ++mobileLoopToken;
        let isProcessingMobileFrame = false;
        let mobileProcStartMs = 0;
        let lastMobileMpMs = 0;

        const onFrame = async (nowMs) => {
            if (!isWebcamActive || myMobileToken !== mobileLoopToken) return;
            mobileFrameLoopId = requestAnimationFrame(onFrame);

            if (document.hidden) return;

            if (dom.webcam && dom.webcam.paused && dom.webcam.srcObject) {
                dom.webcam.play().catch(() => {});
                return;
            }
            if (!mobileHands || !dom.webcam || dom.webcam.readyState < 2 || dom.webcam.videoWidth === 0) return;

            const ts = nowMs || performance.now();
            if (isProcessingMobileFrame) {
                if (ts - mobileProcStartMs > MOBILE_MP_WATCHDOG_MS) {
                    isProcessingMobileFrame = false;
                } else {
                    return;
                }
            }

            const isMobileFastMotion = (mobileMotionSpeed >= 0.015 || clientDynamicGesture !== null);
            const mobileInterval = (mobileDroppedFramesCount >= 5)
                ? MOBILE_MP_IDLE_MS
                : (isMobileFastMotion ? 28 : 48);
            if (ts - lastMobileMpMs < mobileInterval) return;
            lastMobileMpMs = ts;

            const vw = dom.webcam.videoWidth;
            const vh = dom.webcam.videoHeight;
            const targetH = Math.max(180, Math.round((MOBILE_MP_INPUT_WIDTH * vh) / Math.max(1, vw)));
            if (mobileMpInputCanvas.height !== targetH) {
                mobileMpInputCanvas.height = targetH;
            }

            isProcessingMobileFrame = true;
            mobileProcStartMs = ts;
            try {
                mobileMpInputCtx.drawImage(dom.webcam, 0, 0, MOBILE_MP_INPUT_WIDTH, mobileMpInputCanvas.height);
                await mobileHands.send({ image: mobileMpInputCanvas });
            } catch (_) { }
            finally {
                isProcessingMobileFrame = false;
            }
        };

        mobileFrameLoopId = requestAnimationFrame(onFrame);

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

    if (data.gesture_type || data.model_source || data.specialist_used) {
        updateMobileModelTypeBadgeUI(data.gesture_type, data.model_source, data.specialist_used);
    }

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
    const DYNAMIC_GESTURES = new Set(["HOW ARE YOU", "NICE TO MEET YOU", "HELLO", "J", "Z"]);

    let displaySign = CONTROL_SIGNS_MAP[upperLive] || liveSign;
    if (CONTROL_SIGNS_MAP[upperLive] || CONTROL_SIGNS_MAP[cleanPredUpper]) {
        const ctrl = CONTROL_SIGNS_MAP[upperLive] || CONTROL_SIGNS_MAP[cleanPredUpper];
        displaySign = ctrl;
    } else if (clientDynamicGesture) {
        displaySign = `${clientDynamicGesture} (IN MOTION)`;
    } else if (rawPredUpper === "J_START") {
        displaySign = "I";
    } else if (rawPredUpper.endsWith("_START") || DYNAMIC_GESTURES.has(cleanPredUpper)) {
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
    if (typeof window._mDynOtherStreak === "undefined") {
        window._mDynOtherStreak = 0;
        window._mDynStaticBStreak = 0;
        window._mHelloStartStreak = 0;
    }
    if (clientDynamicGesture) {
        const elapsedSec = (now - clientDynamicStartTime) / 1000.0;
        const dynamicMaxSec = (clientDynamicGesture === "NICE TO MEET YOU")
            ? 4.2
            : (clientDynamicGesture === "HELLO" ? 2.0 : 3.0);
        const remainingSec = Math.max(0, dynamicMaxSec - elapsedSec);
        const ratio = Math.max(0, Math.min(100, (remainingSec / dynamicMaxSec) * 100));

        if (dom.dynBanner) dom.dynBanner.style.display = "flex";
        if (dom.dynTitle) dom.dynTitle.textContent = clientDynamicGesture;
        if (dom.dynTimer) dom.dynTimer.textContent = remainingSec.toFixed(1) + "s";
        if (dom.dynBarFill) dom.dynBarFill.style.width = ratio + "%";

        const isFrameNeutral = (rawPred === "NEUTRAL" || data.is_neutral || data.prediction === "—" || !data.prediction || detHand === "none");
        if (isFrameNeutral) {
            const neutralLimitMs = (detHand === "none") ? 350 : 900;
            if (!clientDynamicNeutralStart) {
                clientDynamicNeutralStart = now;
            } else if (now - clientDynamicNeutralStart > neutralLimitMs) {
                clientDynamicGesture = null;
                clientDynamicNeutralStart = 0;
                window._mDynOtherStreak = 0;
                window._mDynStaticBStreak = 0;
                if (dom.dynBanner) dom.dynBanner.style.display = "none";
                clientStabilityBuffer = [];
                updateStabilityDots(0);
                return;
            }
        } else {
            clientDynamicNeutralStart = 0;
        }

        // Check completion trigger: requires minimum elapsed movement time AND consecutive _END hold frames
        if (typeof window._mDynEndStreak === "undefined") window._mDynEndStreak = 0;
        let isEndFrame = false;
        let requiredEndFrames = 2;
        if (clientDynamicGesture === "NICE TO MEET YOU") {
            const isPointingYou = (rawPredUpper === "YOU" || cleanPredUpper === "YOU" || (data.right_gesture && String(data.right_gesture).toUpperCase() === "YOU"));
            const isNiceEndPose = (rawPredUpper === "NICE TO MEET YOU_END" || (rawPredUpper.endsWith("_END") && cleanPredUpper === "NICE TO MEET YOU"));
            requiredEndFrames = 3;
            if (elapsedSec >= 1.65 && (isNiceEndPose || isPointingYou)) {
                isEndFrame = true;
            }
        } else if (clientDynamicGesture === "HOW ARE YOU") {
            const isPointingYou = (rawPredUpper === "YOU" || cleanPredUpper === "YOU" || (data.right_gesture && String(data.right_gesture).toUpperCase() === "YOU"));
            const isHowEndPose = (rawPredUpper === "HOW ARE YOU_END" || (rawPredUpper.endsWith("_END") && cleanPredUpper === "HOW ARE YOU"));
            requiredEndFrames = 3;
            if (elapsedSec >= 1.15 && (isPointingYou || isHowEndPose)) {
                isEndFrame = true;
            }
        } else if (clientDynamicGesture === "HELLO") {
            const isHelloEnd = (rawPredUpper === "HELLO_END" || (rawPredUpper.endsWith("_END") && cleanPredUpper === "HELLO"));
            requiredEndFrames = 2;
            if (isHelloEnd && elapsedSec >= 0.40) {
                isEndFrame = true;
            }
        } else {
            // Other dynamic gestures (e.g. J, Z)
            requiredEndFrames = 2;
            if (rawPredUpper === `${clientDynamicGesture}_END` && elapsedSec >= 0.50) {
                isEndFrame = true;
            }
        }
        if (isEndFrame) {
            window._mDynEndStreak++;
        } else if (!isFrameNeutral) {
            window._mDynEndStreak = 0;
        }
        const isEndSignal = (window._mDynEndStreak >= requiredEndFrames);
        if (isEndSignal) {
            window._mDynEndStreak = 0;
            const completedGesture = clientDynamicGesture;
            if (completedGesture === "J") retractRecentStaticI(now);
            clientDynamicGesture = null;
            clientDynamicNeutralStart = 0;
            window._mDynOtherStreak = 0;
            window._mDynStaticBStreak = 0;
            clientDynamicCooldownUntil = now + 1000;
            clientSuppressYouUntil = now + 1800;
            activeSign = completedGesture;
            mobileLastConfirmedGesture = completedGesture;
            mobileConfirmedGestureTime = performance.now();
            lockUntil = now + 1100;
            clientLastWord = completedGesture;
            clientLastTriggerTime = now;
            if (dom.dynBanner) dom.dynBanner.style.display = "none";
            if (dom.signVal) dom.signVal.textContent = completedGesture;
            if (isGestureRecordingActive) {
                appendSignToTranscript(completedGesture);
            }
            showToast(`Dynamic Sign: ${completedGesture}`);
            clientStabilityBuffer = [];
            updateStabilityDots(CONFIRM_WIN_COUNT);
            return;
        }

        // Preempt / stop dynamic timer immediately when user performs another gesture!
        let mCancelledByOther = false;
        const mIsOneHandDyn = (clientDynamicGesture === "HELLO" || clientDynamicGesture === "J" || clientDynamicGesture === "Z");
        if (!isFrameNeutral) {
            const mIsOtherStart = rawPredUpper.endsWith("_START") && cleanPredUpper !== clientDynamicGesture && (mIsOneHandDyn || detHand === "both");
            const mIsTwoHandSwitch = mIsOneHandDyn && (detHand === "both" || Boolean(CONTROL_SIGNS_MAP[cleanPredUpper]));
            let mIncompat = false;
            if (clientDynamicGesture === "HELLO") {
                if (cleanPredUpper === "B" && mobileMotionSpeed < 0.018) {
                    window._mDynStaticBStreak++;
                } else {
                    window._mDynStaticBStreak = 0;
                }
                if (cleanPredUpper !== "HELLO" && cleanPredUpper !== "B" && confVal >= 0.55) {
                    mIncompat = true;
                }
            } else if (clientDynamicGesture === "J") {
                if (
                    (cleanPredUpper === "Y" && mobileMotionSpeed < 0.020)
                    || (rawPredUpper === "J_START" && mobileMotionSpeed < 0.018)
                ) {
                    window._mDynStaticBStreak++;
                } else {
                    window._mDynStaticBStreak = 0;
                }
                if (cleanPredUpper !== "J" && cleanPredUpper !== "I" && cleanPredUpper !== "Y" && confVal >= 0.62) {
                    mIncompat = true;
                }
            } else if (clientDynamicGesture === "Z") {
                if (["D", "X", "P", "Q", "Z"].includes(cleanPredUpper) && rawPredUpper !== "Z_END" && mobileMotionSpeed < 0.018) {
                    window._mDynStaticBStreak++;
                } else {
                    window._mDynStaticBStreak = 0;
                }
                if (!["Z", "D", "G", "L", "X", "P", "Q", "YOU"].includes(cleanPredUpper) && confVal >= 0.62) {
                    mIncompat = true;
                }
            } else if ((cleanPredUpper === "START" || cleanPredUpper === "STOP") && confVal >= 0.65) {
                mIncompat = true;
            }

            if (mIsTwoHandSwitch || mIsOtherStart || mIncompat) {
                window._mDynOtherStreak++;
            } else {
                window._mDynOtherStreak = 0;
            }

            if (window._mDynOtherStreak >= 2 || window._mDynStaticBStreak >= 6) {
                clientDynamicGesture = null;
                clientDynamicNeutralStart = 0;
                window._mDynOtherStreak = 0;
                window._mDynStaticBStreak = 0;
                if (dom.dynBanner) dom.dynBanner.style.display = "none";
                mCancelledByOther = true;
            }
        }

        if (!mCancelledByOther) {
            // Cancel if timeout reached without reaching the completion pose
            if (remainingSec <= 0) {
                clientDynamicGesture = null;
                clientDynamicNeutralStart = 0;
                window._mDynOtherStreak = 0;
                window._mDynStaticBStreak = 0;
                if (dom.dynBanner) dom.dynBanner.style.display = "none";
                clientStabilityBuffer = [];
                updateStabilityDots(0);
                return;
            }
            return;
        }
    } else {
        if (dom.dynBanner) dom.dynBanner.style.display = "none";
    }

    // 3. Initiate dynamic gesture ONLY on a genuine _START frame with solid confidence
    // ── Static "I" vs Dynamic "J" ──
    let forceJStart = false;
    let forceJBackdateMs = 0;
    const mIsJMotionPattern = isDrawingJPattern(mobileSequenceBuffer, rawPredUpper);
    if (!clientDynamicGesture && rawPredUpper === "J_START") {
        clientIPoseLastSeen = now;
        if (now < clientDynamicCooldownUntil) {
            updateStabilityDots(0);
            return;
        }
        if (!mIsJMotionPattern) {
            // Holding or repositioning the hand (upward/sideways/slight shift) in I pose -> static letter "I"
            if (now < lockUntil && activeSign !== null) {
                if (dom.signVal) dom.signVal.textContent = activeSign;
            } else {
                handlePrediction("I", confVal, data.status);
                if (dom.signVal) dom.signVal.textContent = (activeSign !== null) ? activeSign : "I";
            }
            return;
        }
        // Hand is actively drawing the downward J stroke -> start dynamic "J"
        forceJStart = true;
    } else if (
        !clientDynamicGesture
        && now >= clientDynamicCooldownUntil
        && (now - clientIPoseLastSeen) < 1400
        && mIsJMotionPattern
        && (cleanPredUpper === "J" || cleanPredUpper === "Y" || cleanPredUpper === "I")
    ) {
        // Hand just left the "I" pose and is drawing the J curve (mid-swoop / J_END) -> start "J"
        forceJStart = true;
        if (rawPredUpper === "J_END") forceJBackdateMs = 400;
    }

    const isStartCandidate = (now >= clientDynamicCooldownUntil) && (
        rawPredUpper.endsWith("_START")
    );
    if (forceJStart || (isStartCandidate && confVal >= 0.50 && DYNAMIC_GESTURES.has(cleanPredUpper))) {
        if (!forceJStart && cleanPredUpper === "Z" && !isDrawingZPattern(mobileSequenceBuffer)) {
            return;
        }
        if (!forceJStart && cleanPredUpper === "HELLO") {
            window._mHelloStartStreak = (window._mHelloStartStreak || 0) + 1;
            if (window._mHelloStartStreak < 2) {
                return;
            }
        } else {
            window._mHelloStartStreak = 0;
        }
        const candidate = forceJStart ? "J" : cleanPredUpper;
        clientDynamicGesture = candidate;
        clientDynamicStartTime = now - forceJBackdateMs;
        clientDynamicNeutralStart = 0;
        window._mDynOtherStreak = 0;
        window._mDynStaticBStreak = 0;
        window._mHelloStartStreak = 0;
        const candidateMaxSec = (candidate === "NICE TO MEET YOU") ? 4.2 : (candidate === "HELLO" ? 2.0 : 3.0);
        if (dom.dynBanner) {
            dom.dynBanner.style.display = "flex";
            if (dom.dynTitle) dom.dynTitle.textContent = candidate;
            if (dom.dynTimer) dom.dynTimer.textContent = candidateMaxSec.toFixed(1) + "s";
            if (dom.dynBarFill) dom.dynBarFill.style.width = "100%";
        }
        clientStabilityBuffer = [];
        return;
    } else {
        window._mHelloStartStreak = 0;
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
        activeSign = null;
        candidateSign = null;
        candidateCount = 0;
        confirmationBuffer = [];
        updateStabilityDots(0);

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

    const candUpper = (cleanPred || "").toUpperCase().trim();
    const predUpper = (data.prediction || "").toUpperCase().trim();

    if (now < clientSuppressYouUntil && (candUpper === "YOU" || predUpper === "YOU")) {
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
            : (cleanPred || mobileLastConfirmedGesture || "—");
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

    const DYNAMIC_GESTURES = new Set(["HOW ARE YOU", "NICE TO MEET YOU", "HELLO", "J", "Z"]);
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

    // 6-frame confirmation buffer:
    // Push new inference label into the sliding window of size 6
    confirmationBuffer.push(predictedLabel);
    if (confirmationBuffer.length > CONFIRM_BUFFER_SIZE) {
        confirmationBuffer.shift();
    }

    // A gesture must win at least 5 out of 6 consecutive inference frames to become the active candidate
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
        return; // Did not meet 5-of-6 confirmation threshold yet
    }

    // Allow repeating the exact same gesture (e.g., "U" -> "UU", "L" -> "LL", "BACKSPACE"):
    // 1) Immediately after brief neutral/hand-drop (activeSign reset to null), OR
    // 2) By holding the same gesture steadily past SAME_SIGN_REPEAT_MS (1100ms).
    const SAME_SIGN_REPEAT_MS = 1100;
    const candUp = (winningCandidate || "").toUpperCase().trim();
    if (activeSign !== null) {
        if (winningCandidate === activeSign) {
            const isStartStop = (candUp === "START" || candUp === "STOP");
            if (isStartStop || (now - clientLastTriggerTime < SAME_SIGN_REPEAT_MS)) {
                return;
            }
            confirmationBuffer = [];
        } else if (now < lockUntil) {
            // Locked: suppress rapid swap to a different sign
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
        try { mobileRecognition.stop(); } catch (_) { }
    }
    if (mobileScriptProcessor) {
        try { mobileScriptProcessor.disconnect(); } catch (_) { }
        mobileScriptProcessor = null;
    }
    if (mobileMicStream) {
        mobileMicStream.getTracks().forEach(t => {
            try { t.stop(); } catch (_) { }
        });
        mobileMicStream = null;
    }
    if (mobileAudioCtx && mobileAudioCtx.state !== "closed") {
        try { mobileAudioCtx.close(); } catch (_) { }
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
        try { mobileRecognition.stop(); } catch (_) { }
    }

    if (mobileScriptProcessor) {
        try { mobileScriptProcessor.disconnect(); } catch (_) { }
        mobileScriptProcessor = null;
    }

    const inputSampleRate = mobileAudioCtx ? mobileAudioCtx.sampleRate : 44100;

    if (mobileMicStream) {
        mobileMicStream.getTracks().forEach(t => {
            try { t.stop(); } catch (_) { }
        });
        mobileMicStream = null;
    }

    if (mobileAudioCtx && mobileAudioCtx.state !== "closed") {
        try { await mobileAudioCtx.close(); } catch (_) { }
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

