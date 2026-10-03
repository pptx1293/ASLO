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

function getSmoothedDisplaySign(newDisplay) {
    if (!newDisplay || newDisplay === "—") {
        mobileDisplayHistory.push("—");
        if (mobileDisplayHistory.length > 3) mobileDisplayHistory.shift();
        if (mobileDisplayHistory.filter(x => x === "—").length >= 2) {
            mobileSmoothedDisplaySign = "—";
        }
        return mobileSmoothedDisplaySign;
    }

    if (newDisplay.includes("(IN MOTION)")) {
        mobileDisplayHistory = [newDisplay];
        mobileSmoothedDisplaySign = newDisplay;
        return newDisplay;
    }

    mobileDisplayHistory.push(newDisplay);
    if (mobileDisplayHistory.length > 3) mobileDisplayHistory.shift();

    const counts = {};
    for (const g of mobileDisplayHistory) {
        counts[g] = (counts[g] || 0) + 1;
    }
    let topSign = mobileSmoothedDisplaySign;
    let maxCount = 0;
    for (const [g, count] of Object.entries(counts)) {
        if (count > maxCount) {
            maxCount = count;
            topSign = g;
        }
    }
    if (maxCount >= 2) {
        mobileSmoothedDisplaySign = topSign;
    }
    return mobileSmoothedDisplaySign;
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
const REPEAT_DELAY_MS = 1200;
const STABILITY_REQUIRED_COUNT = 3;
const BUFFER_MAX_LEN = 5;

let isSoundMuted = localStorage.getItem("aslo_mobile_sound_muted") === "true";
let stabilitySegmentsCount = 8;
let webcamStream = null;
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

            // Only reset inactive track after 350ms grace period
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

            results.push({ trackId: this.tracks[0].id, label: label0, landmarks: smoothed0 });
            results.push({ trackId: this.tracks[1].id, label: label1, landmarks: smoothed1 });
        }

        return results;
    }
}

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
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.6
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

            const now = performance.now();
            const stabilizedList = mobileStabilizer.process(
                results.multiHandLandmarks,
                results.multiHandedness,
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
                        color: "#FF0000",
                        fillColor: "#FF0000",
                        lineWidth: 1.0,
                        radius: 3.5
                    });
                } else {
                    ctx.save();
                    ctx.fillStyle = "#FF0000";
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

            // 2. Throttled backend landmark prediction
            const nowTime = performance.now();
            if (nowTime - lastMobilePredictTime >= MOBILE_PREDICT_INTERVAL && !isPredicting) {
                lastMobilePredictTime = nowTime;
                sendMobileLandmarks(results, currentMobileStabilizedHands);
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

async function sendMobileLandmarks(results, stabilizedHands = []) {
    if (isPredicting) return;

    const numHands = (results && results.multiHandLandmarks) ? results.multiHandLandmarks.length : 0;
    if (numHands === 0) {
        mobileStabilizer.resetAll();
        applyMobileTelemetry({
            ok: true,
            prediction: "—",
            live_gesture: "—",
            live_conf: 0.0,
            detected_hand: "none"
        });
        return;
    }

    if (numHands === 1) {
        const rawH = results.multiHandedness && results.multiHandedness[0];
        const physicalHand = getActualPhysicalHand(rawH);

        if (physicalHand === "Left") {
            mobileStabilizer.resetHand("Left");
            applyMobileTelemetry({
                ok: true,
                prediction: null,
                message: "Please use your Right Hand for single-hand signs",
                detected_hand: "left_ignored",
                live_gesture: "—",
                live_conf: 0.0
            });
            const now = performance.now();
            if (now - lastLeftHandToastTime > 3500) {
                lastLeftHandToastTime = now;
                showToast("Please use your Right Hand for single-hand signs", 3500);
            }
            return;
        }

        isPredicting = true;
        try {
            const rawLandmarks = results.multiHandLandmarks[0];
            const smoothed = (stabilizedHands && stabilizedHands[0])
                ? stabilizedHands[0].landmarks
                : mobileStabilizer.filterHand("Right", rawLandmarks);
            const landmarks = toMobileLandmarkArray(smoothed);

            const payload = {
                landmarks: landmarks,
                handedness: "Right",
                hand_type: "right",
                is_mirrored: isMirrored
            };

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
            // Tolerated frame drop
        } finally {
            isPredicting = false;
        }
        return;
    }

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
                h0Smoothed = mobileStabilizer.filterHand(h0Hand, hand0);
                h1Smoothed = mobileStabilizer.filterHand(h1Hand, hand1);
            }

            // Phantom Duplicate Hand Guard:
            const w0 = h0Smoothed[0];
            const w1 = h1Smoothed[0];
            const x0 = w0.x !== undefined ? w0.x : w0[0];
            const y0 = w0.y !== undefined ? w0.y : w0[1];
            const x1 = w1.x !== undefined ? w1.x : w1[0];
            const y1 = w1.y !== undefined ? w1.y : w1[1];
            const wristDist = Math.hypot(x0 - x1, y0 - y1);

            // If wrists are in the exact same location (< 0.05), it is a single physical hand detected twice
            if (wristDist < 0.05) {
                const primaryHand = (h0Hand === "Right") ? h0Smoothed : ((h1Hand === "Right") ? h1Smoothed : h0Smoothed);
                const primaryLabel = (h0Hand === "Right" || h1Hand === "Right") ? "Right" : h0Hand;
                if (primaryLabel === "Left") {
                    mobileStabilizer.resetHand("Left");
                    return;
                }
                const landmarks = toMobileLandmarkArray(primaryHand);
                const payload = {
                    landmarks: landmarks,
                    handedness: "Right",
                    hand_type: "right",
                    is_mirrored: isMirrored
                };
                const resp = await fetch("/predict_landmarks", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (resp.ok) {
                    const data = await resp.json();
                    applyMobileTelemetry(data);
                }
                return;
            }

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

            const h0Pts = toMobileLandmarkArray(h0Smoothed);
            const h1Pts = toMobileLandmarkArray(h1Smoothed);

            const allHands = [
                { label: h0Hand, points: h0Pts },
                { label: h1Hand, points: h1Pts }
            ];

            const payload = {
                landmarks: (h0Hand === "Right" ? h0Pts : h1Pts),
                all_hands: allHands,
                hands: allHands,
                handedness: "Both",
                hand_type: "both",
                is_mirrored: isMirrored
            };

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
            // Tolerated frame drop
        } finally {
            isPredicting = false;
        }
        return;
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
        let stream = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { exact: currentFacingMode }
                },
                audio: false
            });
        } catch (exactErr) {
            console.warn("[Mobile] Exact facingMode failed, falling back to ideal:", exactErr);
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: currentFacingMode }
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

        // 5. Continuous frame processing loop directly into MediaPipe Hands with WASM mutex lock
        let isProcessingMobileFrame = false;
        const onFrame = async () => {
            if (!isWebcamActive) return;
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
    const smoothedDisplay = getSmoothedDisplaySign(displaySign);
    if (dom.signVal) dom.signVal.textContent = smoothedDisplay;
    if (dom.signSub) dom.signSub.textContent = isGestureRecordingActive ? "Live tracking" : "Paused";

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
    const cleanPred = stripInternalSuffix(data.prediction).trim();

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
        clientStabilityBuffer = [];
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
        if (dom.signVal) dom.signVal.textContent = "NEUTRAL";
        if (dom.signSub) dom.signSub.textContent = "Resting / Neutral";
        clientStabilityBuffer = [];
        updateStabilityDots(0);
        return;
    }

    // 5. Stability Buffer Smoothing (5 frames window, >=3 agreement, confidence >= 0.55)
    clientStabilityBuffer.push(cleanPred);
    if (clientStabilityBuffer.length > BUFFER_MAX_LEN) clientStabilityBuffer.shift();

    const matchCount = clientStabilityBuffer.filter(p => p === cleanPred).length;
    updateStabilityDots(matchCount);

    if (matchCount >= STABILITY_REQUIRED_COUNT && confVal >= 0.40) {
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
            if (detHand !== "both" || confVal < 0.35) {
                clientStabilityBuffer = [];
                return;
            }
            if (upper === "START") {
                if (!isGestureRecordingActive) {
                    isGestureRecordingActive = true;
                    updateRecordingUI();
                    speakText("Recording start");
                    showToast("Recording started");
                }
                clientStabilityBuffer = [];
                return;
            }
            if (upper === "STOP") {
                if (isGestureRecordingActive) {
                    isGestureRecordingActive = false;
                    updateRecordingUI();
                    speakCompletedSentence(clientSentence);
                    showToast("Recording paused");
                }
                clientStabilityBuffer = [];
                return;
            }
        }
        if (upper === "SPACE") {
            if (now - clientLastTriggerTime > REPEAT_DELAY_MS) {
                if (isGestureRecordingActive) appendSignToTranscript("SPACE");
                clientLastTriggerTime = now;
            }
            clientStabilityBuffer = [];
            return;
        }
        if (upper === "BACKSPACE" || upper === "BACK SPACE") {
            if (now - clientLastTriggerTime > REPEAT_DELAY_MS) {
                backspaceTranscriptLocal();
                clientLastTriggerTime = now;
            }
            clientStabilityBuffer = [];
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
            clientStabilityBuffer = [];
            updateStabilityDots(0);
            return;
        }

        // Suppress trailing 'YOU' after dynamic phrase
        if (upper === "YOU" && now < clientSuppressYouUntil) {
            clientStabilityBuffer = [];
            return;
        }

        // Regular vocabulary sign
        if (cleanPred !== clientLastWord || (now - clientLastTriggerTime > REPEAT_DELAY_MS)) {
            if (isGestureRecordingActive) {
                appendSignToTranscript(cleanPred);
                showToast(`Sign: ${cleanPred}`);
            }
            clientLastWord = cleanPred;
            clientLastTriggerTime = now;
            clientStabilityBuffer = [];
        }
    }
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
        const resp = await fetch("/api/stt/transcribe", {
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

// Initialize Client Webcam & Isolated UI
updateCameraMirrorDisplay();
updateRecordingUI();
updateTranscriptDOM();
updateMobileVoiceDOM();
startMobileWebcam();
