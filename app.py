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
import tensorflow as tf

import aslo_features
from aslo_pipeline import (
    KeypointInterpolator,
    LowLightEnhancer,
    MotionBasedRouter,
    SoftmaxMarginVerifier,
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

@tf.function(reduce_retracing=True)
def fast_predict(x):
    return model(x, training=False)

if model is not None:
    try:
        fast_predict(tf.zeros((1, aslo_features.FEATURE_LEN), dtype=tf.float32))
        fast_predict(tf.zeros((3, aslo_features.FEATURE_LEN), dtype=tf.float32))
    except Exception as _e:
        print(f"Inference graph warmup note: {_e}")

# Server-side MediaPipe Hands is disabled: Hand tracking is now executed
# zero-latency in the client browser, transmitting only 21 landmark coordinates.
hands = None

# ── ASLO Pipeline Configurations ───────────────────────────────────────────
low_light_enhancer = LowLightEnhancer(base_clip_limit=2.5, tile_grid_size=(8, 8))
margin_verifier = SoftmaxMarginVerifier(min_confidence=0.55, min_margin=0.10)
N_FRAME = 5
N_FRAMES = 5

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
        pts = []
        for p in raw_pts:
            if isinstance(p, dict):
                px = float(p.get("x", 0.0))
                py = float(p.get("y", 0.0))
                pz = float(p.get("z", 0.0))
            else:
                px = float(p[0])
                py = float(p[1])
                pz = float(p[2]) if len(p) > 2 else 0.0
            pts.append([px, py, pz])
        return pts

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
        h0_pts = _to_mirrored(all_hands[0].get("points", []))
        h1_pts = _to_mirrored(all_hands[1].get("points", []))
        h0_lbl = (all_hands[0].get("label") or "").capitalize().strip()
        h1_lbl = (all_hands[1].get("label") or "").capitalize().strip()

        # Genuine two-handed gestures strictly require one Right and one Left hand
        has_opposite_handedness = (h0_lbl == "Right" and h1_lbl == "Left") or (h0_lbl == "Left" and h1_lbl == "Right")
        d_wrists = 1.0
        if h0_pts and h1_pts and len(h0_pts) >= 1 and len(h1_pts) >= 1:
            w0 = np.array([h0_pts[0][0], h0_pts[0][1]])
            w1 = np.array([h1_pts[0][0], h1_pts[0][1]])
            d_wrists = float(np.linalg.norm(w0 - w1))

        if d_wrists >= 0.04 and (has_opposite_handedness or (hand_type_clean == "both") or len(all_hands) == 2):
            is_two_hand_inference = True
        else:
            # Fall back to single hand: prioritize Right hand if detected
            if h0_lbl == "Right":
                landmarks_data = h0_pts
            elif h1_lbl == "Right":
                landmarks_data = h1_pts
            else:
                landmarks_data = h0_pts

    if is_two_hand_inference:
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

        r_lms = aslo_features.HandLandmarks(r_pts)
        l_lms = aslo_features.HandLandmarks(l_pts)

        feats_both = aslo_features.validate_feature_vector(aslo_features.extract_two_hand_features(r_lms, l_lms))
        feats_r = aslo_features.validate_feature_vector(aslo_features.extract_single_hand_features(r_lms, is_left_hand=False))
        feats_l = aslo_features.validate_feature_vector(aslo_features.extract_single_hand_features(l_lms, is_left_hand=True))

        feats_batch = np.concatenate([feats_both, feats_r, feats_l], axis=0)
        p_batch = fast_predict(tf.constant(feats_batch, dtype=tf.float32)).numpy()
        p_both, p_r, p_l = p_batch[0], p_batch[1], p_batch[2]

        verified_both, conf_both, status_both, telem_both = margin_verifier.verify(p_both, label_classes)
        pred_status = status_both
        margin_value = telem_both["margin"]
        top_class_name = telem_both["top_label"]
        second_class_name = telem_both["second_label"]
        p_top_both = telem_both["p_top"]
        p_second_both = telem_both["p_second"]

        print(f"[PREDICT] Detected: {top_class_name} ({p_top_both:.2f}), Runner-up: {second_class_name} ({p_second_both:.2f}), Diff: {margin_value:.2f}")

        # Check if two-hand inference is neutral
        prob_both_neutral = float(p_both[NEUTRAL_INDEX]) if (NEUTRAL_INDEX is not None and NEUTRAL_INDEX < len(p_both)) else 0.0
        if (verified_both and verified_both.lower() == "neutral") or (prob_both_neutral >= 0.60 and conf_both < 0.30):
            both_live_gesture = "NEUTRAL"
            conf_both = max(prob_both_neutral, conf_both)
            verified_both = "NEUTRAL"
            pred_status = "valid"
        else:
            if top_class_name and str(top_class_name).lower() in ("start", "stop") and p_top_both >= 0.40:
                verified_both = str(top_class_name).upper()
                conf_both = p_top_both
                pred_status = "valid"
            both_live_gesture = (verified_both.upper() if verified_both else "—")
            both_live_conf = conf_both

        # Right single-hand inference
        idx_r = int(np.argmax(p_r))
        label_r = str(label_classes[idx_r])
        conf_r = float(p_r[idx_r])
        heur_r = aslo_features.apply_heuristics(r_lms, label_r, is_left_hand=False, confidence=conf_r)
        right_live_gesture = heur_r
        right_live_conf = conf_r

        # Left single-hand inference
        idx_l = int(np.argmax(p_l))
        label_l = str(label_classes[idx_l])
        conf_l = float(p_l[idx_l])
        left_live_gesture = label_l
        left_live_conf = conf_l

        pred_label = verified_both
        confidence = conf_both
        detected_hand = "both"

    # Case 2: Single Right hand detected
    else:
        single_pts = _to_mirrored(landmarks_data)
        single_lms = aslo_features.HandLandmarks(single_pts)
        feats_single = aslo_features.validate_feature_vector(
            aslo_features.extract_single_hand_features(single_lms, is_left_hand=False)
        )
        probs = fast_predict(tf.constant(feats_single, dtype=tf.float32)).numpy()[0]

        wrist_y = single_lms.landmark[0].y
        prob_neutral = float(probs[NEUTRAL_INDEX]) if (NEUTRAL_INDEX is not None and NEUTRAL_INDEX < len(probs)) else 0.0
        is_resting = wrist_y > 0.90

        verified_single, conf_single, status_single, telem_single = margin_verifier.verify(probs, label_classes)
        pred_status = status_single
        margin_value = telem_single["margin"]
        top_class_name = telem_single["top_label"]
        second_class_name = telem_single["second_label"]
        p_top_single = telem_single["p_top"]
        p_second_single = telem_single["p_second"]

        print(f"[PREDICT] Detected: {top_class_name} ({p_top_single:.2f}), Runner-up: {second_class_name} ({p_second_single:.2f}), Diff: {margin_value:.2f}")

        if is_resting or (prob_neutral >= 0.60 and conf_single < 0.30):
            heur = "NEUTRAL"
            conf = max(prob_neutral, 0.75 if is_resting else conf_single)
            pred_status = "valid"
        elif verified_single is None:
            # Check if ambiguity is between known disambiguation pairs
            pair_set = {str(top_class_name).upper().strip(), str(second_class_name).upper().strip()}
            can_disambiguate = (
                bool(pair_set.intersection({"U", "V", "R"})) or
                bool(pair_set.intersection({"A", "S"})) or
                bool(pair_set.intersection({"S", "O"})) or
                bool(pair_set.intersection({"P", "Q", "Z", "Z_START", "Z_END"})) or
                bool(pair_set.intersection({"N", "T"}))
            )
            if can_disambiguate and p_top_single >= 0.40:
                resolved = aslo_features.apply_heuristics(single_lms, top_class_name, is_left_hand=False, confidence=p_top_single)
                heur = resolved
                conf = p_top_single
                pred_status = "valid"
            else:
                heur = None
                conf = conf_single
        else:
            if verified_single.lower() == "neutral":
                heur = "NEUTRAL"
                conf = conf_single
            else:
                heur = aslo_features.apply_heuristics(single_lms, verified_single, is_left_hand=False, confidence=conf_single)
                cand_clean = clean_display_label(heur).lower()

                # START and STOP strictly require both hands; never allow them from single-hand inference
                if str(heur).upper() in ("START", "STOP") or cand_clean in ("start", "stop"):
                    heur = None
                    pred_status = "ambiguous"
                conf = conf_single

        pred_label = heur
        confidence = conf
        detected_hand = "right"
        right_live_gesture = heur if heur else "—"
        right_live_conf = conf

    clean_pred = clean_display_label(pred_label) if pred_label else None
    if clean_pred in ("—", "NONE", ""):
        clean_pred = None

    clean_r = clean_display_label(right_live_gesture) if right_live_gesture != "TRACKED" else "TRACKED"
    clean_l = clean_display_label(left_live_gesture) if left_live_gesture != "TRACKED" else "TRACKED"
    clean_both = clean_display_label(both_live_gesture)

    # Final safeguard: START, STOP, and two-handed dynamic phrases strictly require both hands!
    if detected_hand != "both":
        two_hand_guards = ("START", "STOP", "HOW ARE YOU", "NICE TO MEET YOU")
        if pred_label and any(str(pred_label).upper().startswith(g) for g in two_hand_guards):
            pred_label = None
            confidence = 0.0
            pred_status = "ambiguous"
        if clean_pred and any(str(clean_pred).upper().startswith(g) for g in two_hand_guards):
            clean_pred = None
            confidence = 0.0
            pred_status = "ambiguous"
        if any(str(right_live_gesture).upper().startswith(g) for g in two_hand_guards):
            right_live_gesture = "—"
            right_live_conf = 0.0

    is_neutral_signal = bool(clean_pred == "NEUTRAL" or pred_label == "NEUTRAL")
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
        "status": pred_status,
        "top_label": top_class_name,
        "second_label": second_class_name,
        "margin": float(round(margin_value, 2)),
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
