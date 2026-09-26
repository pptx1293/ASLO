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

// Client-Side Session State & Camera Tracking
let currentFacingMode = "user"; // 'user' (front) or 'environment' (rear)
let isMirrored = true;
let isGestureRecordingActive = true;
let clientSentence = "";
let clientLastWord = "";
let clientStabilityBuffer = [];
let clientLastTriggerTime = 0;
let clientDynamicGesture = null;
let clientDynamicStartTime = 0;
const CLIENT_DYNAMIC_MAX_SEC = 2.8;
let clientSuppressYouUntil = 0;
let lastLeftHandToastTime = 0;
const REPEAT_DELAY_MS = 1300;
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

async function sendMobileLandmarks(results) {
    if (isPredicting) return;

    const numHands = (results && results.multiHandLandmarks) ? results.multiHandLandmarks.length : 0;
    if (numHands === 0) {
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
            const primaryHand = results.multiHandLandmarks[0];
            const landmarks = primaryHand.map(pt => [
                Number(pt.x.toFixed(5)),
                Number(pt.y.toFixed(5)),
                Number((pt.z || 0).toFixed(5))
            ]);

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
            const hand0 = results.multiHandLandmarks[0];
            const hand1 = results.multiHandLandmarks[1];
            let h0Hand = getActualPhysicalHand(results.multiHandedness && results.multiHandedness[0]);
            let h1Hand = getActualPhysicalHand(results.multiHandedness && results.multiHandedness[1]);

            const h0Pts = hand0.map(pt => [Number(pt.x.toFixed(5)), Number(pt.y.toFixed(5)), Number((pt.z || 0).toFixed(5))]);
            const h1Pts = hand1.map(pt => [Number(pt.x.toFixed(5)), Number(pt.y.toFixed(5)), Number((pt.z || 0).toFixed(5))]);

            if (h0Hand === h1Hand) {
                const avgX0 = h0Pts.reduce((acc, p) => acc + p[0], 0) / h0Pts.length;
                const avgX1 = h1Pts.reduce((acc, p) => acc + p[0], 0) / h1Pts.length;
                if (avgX0 <= avgX1) {
                    h0Hand = "Right";
                    h1Hand = "Left";
                } else {
                    h0Hand = "Left";
                    h1Hand = "Right";
                }
            }

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
        let stream = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { exact: currentFacingMode },
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });
        } catch (exactErr) {
            console.warn("[Mobile] Exact facingMode failed, falling back to ideal:", exactErr);
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: currentFacingMode },
                    width: { ideal: 640 },
                    height: { ideal: 480 }
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

        // 5. Continuous frame processing loop directly into MediaPipe Hands
        const onFrame = async () => {
            if (!isWebcamActive) return;
            if (mobileHands && dom.webcam && dom.webcam.readyState >= 2 && !dom.webcam.paused) {
                try {
                    await mobileHands.send({ image: dom.webcam });
                } catch (_) {}
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

    // 1. Current Sign Hero Card
    if (dom.signVal) dom.signVal.textContent = liveSign;
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

    const isNeutralOrIdle = (
        !data.prediction ||
        data.prediction === "—" ||
        data.is_neutral ||
        (data.prediction && data.prediction.toUpperCase() === "NEUTRAL") ||
        (data.raw_pred && data.raw_pred.toUpperCase() === "NEUTRAL")
    );

    if (isNeutralOrIdle) {
        if (dom.signVal) dom.signVal.textContent = "NEUTRAL";
        if (dom.signSub) dom.signSub.textContent = "Resting / Neutral";
        if (clientDynamicGesture) {
            clientDynamicGesture = null;
            if (dom.dynBanner) dom.dynBanner.style.display = "none";
        }
        clientStabilityBuffer = [];
        clientLastWord = "";
        updateStabilityDots(0);
        return;
    }

    const now = Date.now();
    const rawPred = (data.raw_pred || data.prediction || "").toUpperCase();
    const cleanPred = stripInternalSuffix(data.prediction).trim();

    // 2. Dynamic Gesture Sequence Countdown Handling
    if (clientDynamicGesture) {
        const elapsedSec = (now - clientDynamicStartTime) / 1000.0;
        const remainingSec = Math.max(0, CLIENT_DYNAMIC_MAX_SEC - elapsedSec);
        const ratio = Math.max(0, Math.min(100, (remainingSec / CLIENT_DYNAMIC_MAX_SEC) * 100));

        if (dom.dynBanner) dom.dynBanner.style.display = "flex";
        if (dom.dynTitle) dom.dynTitle.textContent = clientDynamicGesture;
        if (dom.dynTimer) dom.dynTimer.textContent = remainingSec.toFixed(1) + "s";
        if (dom.dynBarFill) dom.dynBarFill.style.width = ratio + "%";

        // If user drops to neutral, abort dynamic gesture without committing!
        if (rawPred === "NEUTRAL" || data.is_neutral) {
            clientDynamicGesture = null;
            if (dom.dynBanner) dom.dynBanner.style.display = "none";
            clientStabilityBuffer = [];
            updateStabilityDots(0);
            return;
        }

        const isEndSignal = rawPred.endsWith("_END") || (rawPred === "YOU" && clientDynamicGesture.includes("YOU")) || (rawPred === clientDynamicGesture && elapsedSec >= 1.0);
        if (isEndSignal) {
            const completedGesture = clientDynamicGesture;
            clientDynamicGesture = null;
            if (dom.dynBanner) dom.dynBanner.style.display = "none";
            if (isGestureRecordingActive) {
                appendSignToTranscript(completedGesture);
                clientSuppressYouUntil = now + 2500;
                showToast(`Dynamic Sign: ${completedGesture}`);
            }
            clientStabilityBuffer = [];
            clientLastTriggerTime = now;
            updateStabilityDots(0);
            return;
        }

        // Cancel if timeout reached without reaching the completion pose
        if (remainingSec <= 0) {
            clientDynamicGesture = null;
            if (dom.dynBanner) dom.dynBanner.style.display = "none";
            clientStabilityBuffer = [];
            updateStabilityDots(0);
            return;
        }
        return;
    } else {
        if (dom.dynBanner) dom.dynBanner.style.display = "none";
    }

    // 3. Initiate dynamic gesture countdown ONLY if _START gesture detected with high confidence
    if (rawPred.endsWith("_START") && confVal >= 0.60) {
        const candidate = cleanPred.toUpperCase();
        clientDynamicGesture = candidate;
        clientDynamicStartTime = now;
        if (dom.dynBanner) dom.dynBanner.style.display = "flex";
        if (dom.dynTitle) dom.dynTitle.textContent = candidate;
        if (dom.dynTimer) dom.dynTimer.textContent = CLIENT_DYNAMIC_MAX_SEC.toFixed(1) + "s";
        if (dom.dynBarFill) dom.dynBarFill.style.width = "100%";
        clientStabilityBuffer = [];
        return;
    }

    // 4. Stability Buffer Smoothing (5 frames window, >=3 agreement, confidence >= 0.55)
    clientStabilityBuffer.push(cleanPred);
    if (clientStabilityBuffer.length > BUFFER_MAX_LEN) clientStabilityBuffer.shift();

    const matchCount = clientStabilityBuffer.filter(p => p === cleanPred).length;
    updateStabilityDots(matchCount);

    if (matchCount >= STABILITY_REQUIRED_COUNT && confVal >= 0.55) {
        const upper = cleanPred.toUpperCase();
        if (upper === "NEUTRAL" || upper === "IDLE" || upper === "—") {
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
startMobileWebcam();
