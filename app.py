import base64
import collections
import csv
import os
import re
import subprocess
import sys
import threading
import time
import traceback

import cv2
from flask import Flask, Response, jsonify, render_template, request, make_response
try:
    from flask_cors import CORS
except ImportError:
    CORS = None

from keras import models
import mediapipe as mp
import numpy as np

import aslo_features
from aslo_pipeline import (
    KeypointInterpolator,
    LowLightEnhancer,
    MotionBasedRouter,
    TimerDynamicGestureManager,
)
from speech_to_text import SUPPORTED_LANGUAGES, engine as stt_engine

app = Flask(__name__, static_folder="templates", static_url_path="")
if CORS is not None:
    CORS(app)
app.config["TEMPLATES_AUTO_RELOAD"] = True

model_path = "gesture_model.keras"
labels_path = "label_classes.npy"

model = None
label_classes = None

if os.path.exists(model_path) and os.path.exists(labels_path):
    print("Loading model and labels...")
    model = models.load_model(model_path)
    label_classes = np.load(labels_path, allow_pickle=True)
    print(f"Loaded {len(label_classes)} classes.")
else:
    print("Warning: gesture_model.keras or label_classes.npy not found.")
    print("Please run collect_data.py and train_model.py first.")

# Server-side MediaPipe Hands is disabled: Hand tracking is now executed
# zero-latency in the client browser, transmitting only 21 landmark coordinates.
hands = None

# ── ASLO Pipeline Configurations ───────────────────────────────────────────
low_light_enhancer = LowLightEnhancer(base_clip_limit=2.5, tile_grid_size=(8, 8))

TWO_HANDED_LABELS = {
    "START", "STOP", "SPACE", "BACK SPACE", "BACKSPACE", "NEUTRAL",
    "FINE", "ME", "I LOVE YOU", "LOVE",
    "HOW ARE YOU_START", "HOW ARE YOU_END",
    "NICE TO MEET YOU_START", "NICE TO MEET YOU_END"
}
LEFT_HAND_ALLOWED_LABELS = {
    "START", "STOP", "SPACE", "BACK SPACE", "BACKSPACE", "NEUTRAL"
}

latest_features = None
is_retraining = False
_inference_lock = threading.Lock()

# ── Clean Dataset Folder Mapping ───────────────────────────────────────────
DATASET_DIR = "picture_file"
FOLDER_NAMES_MAP = {}
if os.path.exists(DATASET_DIR):
    for fname in os.listdir(DATASET_DIR):
        if os.path.isdir(os.path.join(DATASET_DIR, fname)):
            FOLDER_NAMES_MAP[fname.lower().strip()] = fname
            FOLDER_NAMES_MAP[fname.lower().replace(" ", "_").strip()] = fname

NEUTRAL_INDEX = None
if label_classes is not None:
    classes_list = list(label_classes)
    if "neutral" in classes_list:
        NEUTRAL_INDEX = classes_list.index("neutral")


def get_clean_folder_label(label: str):
    if not label or label in ("—", "none", "NONE"):
        return None
    cleaned = re.sub(r'_(START|END)$', '', str(label), flags=re.IGNORECASE).strip()
    key = cleaned.lower()
    if key in ("dynamic_gesture", "dynamic gesture", "hand detected", "neutral", "tracked", "—", "hand_detected"):
        return None
    if key in FOLDER_NAMES_MAP:
        return FOLDER_NAMES_MAP[key]
    key_norm = key.replace("_", " ")
    if key_norm in FOLDER_NAMES_MAP:
        return FOLDER_NAMES_MAP[key_norm]
    key_under = key.replace(" ", "_")
    if key_under in FOLDER_NAMES_MAP:
        return FOLDER_NAMES_MAP[key_under]
    return cleaned


def clean_display_label(label: str) -> str:
    if not label or str(label).strip() in ("—", "none", "NONE", "IDLE"):
        return "—"
    lbl_strip = str(label).strip()
    lbl_upper = lbl_strip.upper()
    if lbl_upper in ("START", "STOP", "SPACE", "BACKSPACE", "BACK SPACE", "NEUTRAL", "IDLE"):
        return "BACKSPACE" if "BACK" in lbl_upper else lbl_upper
    folder = get_clean_folder_label(label)
    if folder:
        f_upper = folder.strip().upper()
        if f_upper in ("START", "STOP", "SPACE", "BACKSPACE", "BACK SPACE", "NEUTRAL", "IDLE"):
            return "BACKSPACE" if "BACK" in f_upper else f_upper
        return folder
    cleaned = re.sub(r'_(START|END)$', '', lbl_strip, flags=re.IGNORECASE).strip()
    if cleaned.upper() in ("START", "STOP", "SPACE", "BACKSPACE", "BACK SPACE", "NEUTRAL", "IDLE"):
        return "BACKSPACE" if "BACK" in cleaned.upper() else cleaned.upper()
    return cleaned if cleaned else "—"


def apply_clahe_preprocessing(frame_bgr):
    return low_light_enhancer.process(frame_bgr)


# ── Device Type Detection Helpers ──────────────────────────────────────────
MOBILE_UA_KEYWORDS = (
    "mobile", "android", "iphone", "ipad", "ipod", "webos",
    "blackberry", "iemobile", "opera mini", "windows phone", "tablet"
)


def is_mobile_request(req):
    """Inspects User-Agent header to determine if requesting client is mobile."""
    ua = req.headers.get("User-Agent", "").lower()
    return any(k in ua for k in MOBILE_UA_KEYWORDS)


def process_landmarks_data(landmarks_data, handedness="Right", all_hands=None, is_mirrored=True, hand_type=None):
    """
    Core cloud inference engine. Purely stateless and multi-user isolated:
    1. Zero-latency: uses client MediaPipe landmarks, saving 100% video bandwidth & decoding CPU
    2. Enforces dataset validation: Single Left-hand gestures are rejected (dataset only has Right and Both hands)
    3. Routes single Right-hand data to single-hand extractor, and Two-hand data to two-hand extractor
    4. Evaluates static and dynamic ASL gestures using gesture_model.keras
    5. Returns prediction strictly in the response body without mutating any shared server globals
    """
    hands_detected = bool(landmarks_data and len(landmarks_data) >= 21)

    hand_type_clean = (hand_type or "").lower().strip()
    handedness_clean = (handedness or "").capitalize().strip()

    # Check if request contains only a Left hand:
    # Dataset contains ONLY Right Hand and Both Hands gestures (no single Left hand gestures).
    is_left_only = False
    if hand_type_clean == "left":
        is_left_only = True
    elif all_hands and len(all_hands) == 1:
        h0_label = (all_hands[0].get("label") or "").capitalize().strip()
        if h0_label == "Left" or (hand_type_clean != "right" and handedness_clean == "Left"):
            is_left_only = True
    elif not all_hands or len(all_hands) == 0:
        if handedness_clean == "Left" and hand_type_clean != "right":
            is_left_only = True

    if hands_detected and is_left_only:
        return {
            "ok": True,
            "prediction": None,
            "message": "Left-hand-only signs are not supported. Please use your right hand.",
            "detected_hand": "left_ignored",
            "live_gesture": "—",
            "live_conf": 0.0,
            "right_gesture": "—",
            "right_conf": 0.0,
            "left_gesture": "UNSUPPORTED",
            "left_conf": 0.0,
            "both_gesture": "—",
            "both_conf": 0.0,
            "status": "running"
        }

    if not hands_detected:
        return {
            "ok": True,
            "prediction": "—",
            "confidence": 0.0,
            "detected_hand": "none",
            "live_gesture": "—",
            "live_conf": 0.0,
            "right_gesture": "—",
            "right_conf": 0.0,
            "left_gesture": "—",
            "left_conf": 0.0,
            "both_gesture": "—",
            "both_conf": 0.0,
            "status": "running"
        }

    def _to_mirrored(raw_pts):
        if not raw_pts:
            return []
        return [
            [(1.0 - p[0]) if is_mirrored else p[0], p[1], p[2] if len(p) > 2 else 0.0]
            for p in raw_pts
        ]

    pred_label = "—"
    confidence = 0.0
    detected_hand = "none"
    right_live_gesture = "—"
    right_live_conf = 0.0
    left_live_gesture = "—"
    left_live_conf = 0.0
    both_live_gesture = "—"
    both_live_conf = 0.0

    if model is None:
        return {
            "ok": True,
            "prediction": "HAND DETECTED",
            "confidence": 1.0,
            "detected_hand": "right",
            "live_gesture": "HAND DETECTED",
            "live_conf": 1.0,
            "status": "running"
        }

    is_two_hand_inference = False
    if all_hands and len(all_hands) >= 2:
        h0_pts = all_hands[0].get("points", [])
        h1_pts = all_hands[1].get("points", [])

        # Phantom Hand Filter: wrists must be physically separated in space
        is_phantom_duplicate = False
        if h0_pts and h1_pts and len(h0_pts) >= 1 and len(h1_pts) >= 1:
            w0 = np.array([h0_pts[0][0], h0_pts[0][1]])
            w1 = np.array([h1_pts[0][0], h1_pts[0][1]])
            d_wrists = float(np.linalg.norm(w0 - w1))
            if d_wrists < 0.05:
                is_phantom_duplicate = True

        if is_phantom_duplicate:
            landmarks_data = h0_pts
        else:
            is_two_hand_inference = True

    if is_two_hand_inference:
        h0_pts = all_hands[0].get("points", [])
        h1_pts = all_hands[1].get("points", [])
        h0_label = (all_hands[0].get("label") or "Right").capitalize().strip()
        h1_label = (all_hands[1].get("label") or "Left").capitalize().strip()

        if h0_label == "Right" and h1_label == "Left":
            r_pts, l_pts = h0_pts, h1_pts
        elif h0_label == "Left" and h1_label == "Right":
            r_pts, l_pts = h1_pts, h0_pts
        else:
            avg_x0 = sum(p[0] for p in h0_pts) / len(h0_pts) if h0_pts else 0.5
            avg_x1 = sum(p[0] for p in h1_pts) / len(h1_pts) if h1_pts else 0.5
            if avg_x0 <= avg_x1:
                r_pts, l_pts = h0_pts, h1_pts
            else:
                r_pts, l_pts = h1_pts, h0_pts

        r_lms = aslo_features.HandLandmarks(_to_mirrored(r_pts))
        l_lms = aslo_features.HandLandmarks(_to_mirrored(l_pts))

        # Two-hand inference
        feats_both = aslo_features.extract_two_hand_features(r_lms, l_lms)
        p_both = np.array(model(np.array([feats_both], dtype=np.float32), training=False))[0]
        idx_both = int(np.argmax(p_both))
        label_both = str(label_classes[idx_both])
        conf_both = float(p_both[idx_both])
        both_live_gesture = label_both.upper()
        both_live_conf = conf_both

        # Check if two-hand inference is neutral
        prob_both_neutral = float(p_both[NEUTRAL_INDEX]) if (NEUTRAL_INDEX is not None and NEUTRAL_INDEX < len(p_both)) else 0.0
        if label_both.lower() == "neutral" or (prob_both_neutral >= 0.60 and conf_both < 0.30):
            both_live_gesture = "NEUTRAL"
            conf_both = max(prob_both_neutral, conf_both)

        # Right single-hand inference
        feats_r = aslo_features.extract_single_hand_features(r_lms, is_left_hand=False)
        p_r = np.array(model(np.array([feats_r], dtype=np.float32), training=False))[0]
        idx_r = int(np.argmax(p_r))
        label_r = str(label_classes[idx_r])
        conf_r = float(p_r[idx_r])
        heur_r = aslo_features.apply_heuristics(r_lms, label_r, is_left_hand=False, confidence=conf_r)
        right_live_gesture = heur_r
        right_live_conf = conf_r

        # Left single-hand inference
        feats_l = aslo_features.extract_single_hand_features(l_lms, is_left_hand=True)
        p_l = np.array(model(np.array([feats_l], dtype=np.float32), training=False))[0]
        idx_l = int(np.argmax(p_l))
        label_l = str(label_classes[idx_l])
        conf_l = float(p_l[idx_l])
        left_live_gesture = label_l
        left_live_conf = conf_l

        pred_label = label_both
        confidence = conf_both
        detected_hand = "both"

    # Case 2: Single Right hand detected
    else:
        single_pts = _to_mirrored(landmarks_data)
        single_lms = aslo_features.HandLandmarks(single_pts)
        feats_single = aslo_features.extract_single_hand_features(single_lms, is_left_hand=False)
        probs = np.array(model(np.array([feats_single], dtype=np.float32), training=False))[0]
        idx = int(np.argmax(probs))
        raw_pred = str(label_classes[idx])
        conf = float(probs[idx])

        # Check if neutral probability is elevated or hand is in resting position
        wrist_y = single_lms.landmark[0].y
        prob_neutral = float(probs[NEUTRAL_INDEX]) if (NEUTRAL_INDEX is not None and NEUTRAL_INDEX < len(probs)) else 0.0

        is_neutral = (
            raw_pred.lower() == "neutral"
            or (prob_neutral >= 0.60 and conf < 0.30)
            or wrist_y > 0.90
        )

        if is_neutral:
            heur = "NEUTRAL"
            conf = max(prob_neutral, conf if raw_pred.lower() == "neutral" else 0.60)
        else:
            heur = raw_pred.upper()
            cand_clean = clean_display_label(heur).lower()

            # START and STOP strictly require both hands; never allow them from single-hand inference
            if str(heur).upper() in ("START", "STOP") or cand_clean in ("start", "stop"):
                heur = "—"
                conf = 0.0

        pred_label = heur
        confidence = conf
        detected_hand = "right"
        right_live_gesture = heur
        right_live_conf = conf

    clean_pred = clean_display_label(pred_label)
    clean_r = clean_display_label(right_live_gesture) if right_live_gesture != "TRACKED" else "TRACKED"
    clean_l = clean_display_label(left_live_gesture) if left_live_gesture != "TRACKED" else "TRACKED"
    clean_both = clean_display_label(both_live_gesture)

    # Final safeguard: START and STOP strictly require both hands!
    if detected_hand != "both":
        if str(pred_label).upper() in ("START", "STOP"):
            pred_label = "—"
            confidence = 0.0
        if str(clean_pred).upper() in ("START", "STOP"):
            clean_pred = "—"
            confidence = 0.0
        if str(right_live_gesture).upper() in ("START", "STOP"):
            right_live_gesture = "—"
            right_live_conf = 0.0

    is_neutral_signal = (clean_pred == "NEUTRAL" or pred_label == "NEUTRAL" or clean_pred == "—")
    is_dynamic_signal = bool(
        pred_label and (
            str(pred_label).upper().endswith("_START") or
            str(pred_label).upper().endswith("_END") or
            str(pred_label).upper() in ("HOW ARE YOU", "NICE TO MEET YOU")
        )
    )

    return {
        "ok": True,
        "prediction": clean_pred,
        "raw_pred": pred_label,
        "is_dynamic": is_dynamic_signal,
        "confidence": float(round(confidence, 2)),
        "detected_hand": detected_hand,
        "dominant_hand": "right",
        "live_gesture": clean_pred,
        "live_conf": float(round(confidence, 2)),
        "is_neutral": is_neutral_signal,
        "right_gesture": clean_r,
        "right_conf": float(round(right_live_conf, 2)),
        "left_gesture": clean_l,
        "left_conf": float(round(left_live_conf, 2)),
        "both_gesture": clean_both,
        "both_conf": float(round(both_live_conf, 2)),
        "status": "running"
    }


# ── Flask API Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    """Root route with automatic mobile/desktop device detection and manual overrides."""
    view_pref = request.args.get("view") or request.cookies.get("view_pref")
    if view_pref == "mobile":
        resp = make_response(render_template("mobile.html"))
        resp.set_cookie("view_pref", "mobile", max_age=86400 * 30)
        return resp
    elif view_pref == "desktop":
        resp = make_response(render_template("index.html"))
        resp.set_cookie("view_pref", "desktop", max_age=86400 * 30)
        return resp

    if is_mobile_request(request):
        return render_template("mobile.html")
    return render_template("index.html")


@app.route("/mobile")
def mobile():
    """Explicit mobile view route with cookie preference persistence."""
    resp = make_response(render_template("mobile.html"))
    resp.set_cookie("view_pref", "mobile", max_age=86400 * 30)
    return resp


@app.route("/desktop")
def desktop():
    """Explicit desktop view route with cookie preference persistence."""
    resp = make_response(render_template("index.html"))
    resp.set_cookie("view_pref", "desktop", max_age=86400 * 30)
    return resp


@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    """Client-side landmark inference endpoint.
    Stateless and isolated per HTTP request: accepts lightweight 21-point
    hand landmark coordinates JSON and returns predicted translation."""
    try:
        data = request.get_json(silent=True) or {}
        landmarks = data.get("landmarks", [])
        handedness = data.get("handedness", "Right")
        all_hands = data.get("all_hands") or data.get("hands") or []
        is_mirrored = bool(data.get("is_mirrored", True))
        hand_type = data.get("hand_type")

        with _inference_lock:
            result = process_landmarks_data(
                landmarks_data=landmarks,
                handedness=handedness,
                all_hands=all_hands,
                is_mirrored=is_mirrored,
                hand_type=hand_type
            )

        return jsonify(result)
    except Exception as e:
        print(f"[Error] /predict_landmarks exception: {e}")
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
            "prediction": "—",
            "live_gesture": "—",
            "live_conf": 0.0
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Backwards-compatible inference endpoint for client landmarks or legacy requests."""
    try:
        data = request.get_json(silent=True) or {}
        if "landmarks" in data:
            landmarks = data.get("landmarks", [])
            handedness = data.get("handedness", "Right")
            all_hands = data.get("all_hands") or data.get("hands") or []
            is_mirrored = bool(data.get("is_mirrored", True))
            hand_type = data.get("hand_type")

            with _inference_lock:
                result = process_landmarks_data(
                    landmarks_data=landmarks,
                    handedness=handedness,
                    all_hands=all_hands,
                    is_mirrored=is_mirrored,
                    hand_type=hand_type
                )
            return jsonify(result)

        return jsonify({
            "ok": True,
            "prediction": "—",
            "confidence": 0.0,
            "live_gesture": "—",
            "live_conf": 0.0,
            "message": "Client-side MediaPipe is active. Send landmarks to /predict_landmarks."
        })
    except Exception as e:
        print(f"[Error] /predict exception: {e}")
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
            "prediction": "—",
            "live_gesture": "—",
            "live_conf": 0.0
        }), 500


@app.route("/gesture")
def gesture_status():
    """Stateless status endpoint."""
    return jsonify({
        "status": "running",
        "ok": True
    })


@app.route("/video_feed")
def video_feed():
    """Dummy endpoint for backwards compatibility."""
    return Response(b"", mimetype="text/plain")


@app.route("/backspace", methods=["POST"])
def backspace():
    return jsonify({"ok": True, "status": "backspaced"})


@app.route("/clear", methods=["POST"])
def clear():
    return jsonify({"ok": True, "status": "cleared"})


@app.route("/set_dominant_hand", methods=["POST"])
def set_dominant_hand():
    data = request.get_json(silent=True) or {}
    hand = str(data.get("hand", "right")).lower().strip()
    return jsonify({"ok": True, "dominant_hand": hand})


@app.route("/get_dominant_hand", methods=["GET"])
def get_dominant_hand():
    return jsonify({"ok": True, "dominant_hand": "right"})


@app.route("/toggle_active", methods=["POST"])
def toggle_active():
    data = request.get_json(silent=True) or {}
    active = bool(data.get("active", True))
    return jsonify({"ok": True, "active": active})


@app.route("/append_text", methods=["POST"])
def append_text():
    return jsonify({"ok": True})


@app.route("/api/camera/list", methods=["GET"])
def list_cameras():
    return jsonify({"ok": True, "camera_index": 0, "available_cameras": [0]})


@app.route("/api/camera/switch", methods=["POST"])
def switch_camera():
    return jsonify({"ok": True, "camera_index": 0, "available_cameras": [0], "message": "Camera switch handled client-side in browser"})


@app.route("/api/stt/languages", methods=["GET"])
def get_stt_languages():
    return jsonify({
        "ok": True,
        "languages": SUPPORTED_LANGUAGES,
        "default": "en-US"
    })


@app.route("/api/stt/transcribe", methods=["POST"])
def api_transcribe_audio():
    language = request.form.get("language") or request.args.get("language") or "en-US"

    audio_bytes = None
    if "audio" in request.files:
        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()
    else:
        audio_bytes = request.get_data()

    if not audio_bytes:
        return jsonify({"ok": False, "error": "No audio data received."}), 400

    result = stt_engine.transcribe_wav_bytes(audio_bytes, language=language)
    return jsonify(result)


def _retrain_model_background():
    global is_retraining, model, label_classes, NEUTRAL_INDEX
    try:
        subprocess.run([sys.executable, "train_model.py"], check=True)
        model = models.load_model(model_path)
        label_classes = np.load(labels_path, allow_pickle=True)
        if label_classes is not None:
            classes_list = list(label_classes)
            NEUTRAL_INDEX = classes_list.index("neutral") if "neutral" in classes_list else None
        print(f"Model successfully retrained and reloaded. Now has {len(label_classes)} classes.")
    except Exception as e:
        print(f"Error during retraining: {e}")
    finally:
        is_retraining = False


@app.route("/correct_gesture", methods=["POST"])
def correct_gesture():
    global latest_features, is_retraining
    if is_retraining:
        return jsonify({"ok": False, "message": "Model is currently retraining. Please wait."})

    data = request.get_json(silent=True) or {}
    if "label" not in data:
        return jsonify({"ok": False, "message": "No label provided."})

    correct_label = str(data["label"]).upper().strip()
    if latest_features is None:
        return jsonify({"ok": False, "message": "No gesture detected to correct."})

    with open("gesture_data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        row = list(latest_features) + [correct_label]
        for _ in range(20):
            writer.writerow(row)

    is_retraining = True
    threading.Thread(target=_retrain_model_background, daemon=True).start()
    return jsonify({"ok": True, "message": f'Learning "{correct_label}" in background...'})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "aslo-engine"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting ASLO Flask Server on port {port}...")
    print(f"Navigate to http://localhost:{port} in your browser.")
    app.run(host="0.0.0.0", port=port, debug=False)
