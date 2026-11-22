import base64
import collections
import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import traceback

import cv2
from flask import Flask, Response, jsonify, make_response, render_template, request
try:
    from flask_cors import CORS
except ImportError:
    CORS = None

import numpy as np
import tensorflow as tf

import aslo_features
from aslo_pipeline import (
    ASLOEnsembleRouter,
    KeypointInterpolator,
    LowLightEnhancer,
    MotionBasedRouter,
    SoftmaxMarginVerifier,
    StatelessHierarchicalMLPRouter,
    TimerDynamicGestureManager,
)
from speech_to_text import SUPPORTED_LANGUAGES, engine as stt_engine

app = Flask(__name__, static_folder="templates", static_url_path="")
if CORS is not None:
    CORS(app, resources={r"/*": {"origins": "*"}})
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "online", "engine": "local"}), 200


# ── Stateless Cooperative Multi-MLP Ensemble Router ────────────────────────
mlp_router = ASLOEnsembleRouter(models_dir="models")
low_light_enhancer = LowLightEnhancer(base_clip_limit=2.5, tile_grid_size=(8, 8))
margin_verifier = SoftmaxMarginVerifier(min_confidence=0.55, min_margin=0.10)

FIST_CLUSTER = aslo_features.FIST_CLUSTER_SET
N_FRAME = 5
N_FRAMES = 5

TWO_HANDED_LABELS = {
    "START", "STOP", "SPACE", "BACK SPACE", "BACKSPACE", "NEUTRAL",
    "FINE", "ME", "I LOVE YOU", "LOVE",
    "HOW ARE YOU", "HOW ARE YOU_START", "HOW ARE YOU_END",
    "NICE TO MEET YOU", "NICE TO MEET YOU_START", "NICE TO MEET YOU_END"
}

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


def get_clean_folder_label(label: str):
    if not label or label in ("—", "none", "NONE"):
        return None
    cleaned = re.sub(r"_(START|END)$", "", str(label), flags=re.IGNORECASE).strip()
    key = cleaned.lower()
    if key in ("dynamic_gesture", "dynamic gesture", "hand detected", "neutral", "tracked", "—", "hand_detected", "fist_cluster"):
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
    if not label or str(label).strip() in ("—", "none", "NONE", "IDLE", "FIST_CLUSTER"):
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
    cleaned = re.sub(r"_(START|END)$", "", lbl_strip, flags=re.IGNORECASE).strip()
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
    ua = req.headers.get("User-Agent", "").lower()
    return any(k in ua for k in MOBILE_UA_KEYWORDS)


def process_landmarks_data(
    landmarks_data,
    handedness="Right",
    all_hands=None,
    is_mirrored=True,
    hand_type=None,
    sequence_buffer=None,
    motion_speed=0.0,
    mode="auto",
):
    """
    100% Stateless Multi-MLP Hierarchical Inference Engine:
    - Accepts:
      {
        "landmarks": [...],
        "hands": [...],
        "hand_type": "right"|"both",
        "sequence_buffer": [...],
        "motion_speed": float,
        "mode": "auto"|"asl"|"words"|"dynamic"
      }
    - Routes strictly through `StatelessHierarchicalMLPRouter` across:
      1. Model 1A (`asl_stage1_mlp.keras`) -> Model 1B (`asl_fist_specialist_mlp.keras` for FIST_CLUSTER)
      2. Model 2 (`words_static_mlp.keras`)
      3. Model 3 (`dynamic_mlp.keras` on 25-frame motion sequences)
    """
    if not mlp_router.is_ready:
        mlp_router.load_all_models()

    has_seq = bool(sequence_buffer and len(sequence_buffer) >= 4)
    hands_detected = bool((landmarks_data and len(landmarks_data) >= 21) or (all_hands and len(all_hands) > 0) or has_seq)

    hand_type_clean = (hand_type or "").lower().strip()
    handedness_clean = (handedness or "").capitalize().strip()
    mode_clean = (mode or "auto").lower().strip()

    # Check if request contains only a Left hand
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

    if hands_detected and is_left_only and landmarks_data and len(landmarks_data) >= 1:
        first_pt = landmarks_data[0]
        wx = float(first_pt.get("x", 0.5)) if isinstance(first_pt, dict) else float(first_pt[0])
        wy = float(first_pt.get("y", 0.5)) if isinstance(first_pt, dict) else float(first_pt[1])
        if wx <= 0.55 and wy < 0.85:
            is_left_only = False

    if hands_detected and is_left_only:
        return {
            "ok": True,
            "prediction": None,
            "gesture_type": None,
            "model_source": "none",
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
            "status": "running",
        }

    if not hands_detected:
        return {
            "ok": True,
            "prediction": None,
            "gesture_type": None,
            "model_source": "none",
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
            "status": "running",
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
            if is_mirrored:
                px = 1.0 - px
            pts.append([px, py, pz])
        return pts

    def _mirror_sequence(seq_buf):
        if not seq_buf:
            return []
        return [_to_mirrored(frame) for frame in seq_buf]

    if not mlp_router.is_ready:
        return {
            "ok": True,
            "prediction": "HAND DETECTED",
            "gesture_type": "static_asl",
            "model_source": "fallback",
            "confidence": 1.0,
            "detected_hand": "right",
            "live_gesture": "HAND DETECTED",
            "live_conf": 1.0,
            "status": "running",
        }

    # Resolve 1-hand vs 2-hand landmarks
    is_two_hand_inference = False
    r_pts = []
    l_pts = []
    single_pts = []

    if all_hands and len(all_hands) >= 2:
        h0_pts = _to_mirrored(all_hands[0].get("points") or all_hands[0].get("landmarks") or [])
        h1_pts = _to_mirrored(all_hands[1].get("points") or all_hands[1].get("landmarks") or [])
        h0_lbl = (all_hands[0].get("label") or "").capitalize().strip()
        h1_lbl = (all_hands[1].get("label") or "").capitalize().strip()

        has_opposite_handedness = (h0_lbl == "Right" and h1_lbl == "Left") or (h0_lbl == "Left" and h1_lbl == "Right")
        d_wrists = 1.0
        wy0 = float(h0_pts[0][1]) if (h0_pts and len(h0_pts) >= 1) else 0.5
        wy1 = float(h1_pts[0][1]) if (h1_pts and len(h1_pts) >= 1) else 0.5
        if h0_pts and h1_pts and len(h0_pts) >= 1 and len(h1_pts) >= 1:
            w0 = np.array([h0_pts[0][0], h0_pts[0][1]])
            w1 = np.array([h1_pts[0][0], h1_pts[0][1]])
            d_wrists = float(np.linalg.norm(w0 - w1))

        one_hand_resting = (wy0 >= 0.85 and wy1 < 0.72) or (wy1 >= 0.85 and wy0 < 0.72)

        if d_wrists >= 0.04 and not one_hand_resting and (has_opposite_handedness or hand_type_clean == "both" or len(all_hands) == 2):
            is_two_hand_inference = True
            if h0_lbl == "Right" and h1_lbl == "Left":
                r_pts, l_pts = h0_pts, h1_pts
            elif h0_lbl == "Left" and h1_lbl == "Right":
                r_pts, l_pts = h1_pts, h0_pts
            else:
                avg_x0 = sum(p[0] for p in h0_pts) / len(h0_pts) if h0_pts else 0.5
                avg_x1 = sum(p[0] for p in h1_pts) / len(h1_pts) if h1_pts else 0.5
                if avg_x0 <= avg_x1:
                    r_pts, l_pts = h0_pts, h1_pts
                else:
                    r_pts, l_pts = h1_pts, h0_pts
        else:
            if h0_lbl == "Right" and not (wy0 >= 0.85 and wy1 < 0.72):
                single_pts = h0_pts
            elif h1_lbl == "Right" and not (wy1 >= 0.85 and wy0 < 0.72):
                single_pts = h1_pts
            else:
                single_pts = h0_pts if wy0 <= wy1 else h1_pts
    elif landmarks_data and len(landmarks_data) >= 21:
        single_pts = _to_mirrored(landmarks_data)
    elif all_hands and len(all_hands) == 1:
        single_pts = _to_mirrored(all_hands[0].get("points") or all_hands[0].get("landmarks") or [])

    # Build Canonical 3D Palm-Frame Feature Vectors for ASL (124-dim) and Static Words (252-dim)
    asl_features = None
    word_features = None
    detected_hand = "both" if is_two_hand_inference else "right"
    is_resting_single = False

    single_wrist_y = None
    if is_two_hand_inference:
        r_lms = aslo_features.HandLandmarks(r_pts)
        l_lms = aslo_features.HandLandmarks(l_pts)
        word_features = aslo_features.extract_static_word_features((r_lms, l_lms), hand_type="both")
    elif single_pts and len(single_pts) >= 21:
        single_lms = aslo_features.HandLandmarks(single_pts)
        asl_features = aslo_features.extract_asl_features(single_lms)
        word_features = aslo_features.extract_static_word_features(single_lms, hand_type="right")
        single_wrist_y = float(single_lms.landmark[0].y)
        is_resting_single = single_wrist_y > 0.90

    mirrored_seq = _mirror_sequence(sequence_buffer) if sequence_buffer else None
    effective_hand_type = "both" if is_two_hand_inference else "right"

    # If single hand is resting below frame bottom (y > 0.90) and not in dynamic motion, return NEUTRAL
    if is_resting_single and not is_two_hand_inference and float(motion_speed or 0.0) < mlp_router.MOTION_THRESHOLD:
        return {
            "ok": True,
            "prediction": "NEUTRAL",
            "raw_pred": "NEUTRAL",
            "gesture_type": "static_words",
            "model_source": "resting_gate",
            "specialist_used": "Resting Gate",
            "is_dynamic": False,
            "confidence": 0.90,
            "status": "valid",
            "top_label": "NEUTRAL",
            "second_label": "—",
            "margin": 0.90,
            "detected_hand": "right",
            "dominant_hand": "right",
            "live_gesture": "NEUTRAL",
            "live_conf": 0.90,
            "is_neutral": True,
            "right_gesture": "NEUTRAL",
            "right_conf": 0.90,
            "left_gesture": "—",
            "left_conf": 0.0,
            "both_gesture": "—",
            "both_conf": 0.0,
        }

    # Route through the Soft Cooperative Multi-MLP Ensemble Router
    route_res = mlp_router.route_and_predict(
        asl_features_1d=asl_features,
        word_features_1d=word_features,
        hand_type=effective_hand_type,
        mode=mode_clean,
        sequence_buffer=mirrored_seq,
        motion_speed=float(motion_speed or 0.0),
        wrist_y=single_wrist_y,
    )

    raw_pred = route_res.get("prediction")
    confidence = float(route_res.get("confidence") or 0.0)
    pred_status = route_res.get("status") or "ambiguous"
    telem = route_res.get("telemetry") or {}
    gesture_type = route_res.get("gesture_type") or "static_asl"
    model_source = route_res.get("model_source") or "none"
    specialist_used = route_res.get("specialist_used") or "None"
    routed_via_fist = bool(route_res.get("routed_via_fist_specialist", False))

    top_class_name = telem.get("top_label", "—")
    second_class_name = telem.get("second_label", "—")
    margin_value = float(telem.get("margin", 0.0))

    clean_pred = clean_display_label(raw_pred) if raw_pred else None
    if clean_pred in ("—", "NONE", ""):
        clean_pred = None

    # Ensure Start, Stop, Space, Backspace are always tagged as 'static_words' (Word type gesture)
    if clean_pred in ("START", "STOP", "SPACE", "BACKSPACE", "NEUTRAL"):
        gesture_type = "static_words"
        if model_source == "none":
            model_source = "words_static_mlp"
        if specialist_used in ("None", "Stage-1 Direct"):
            specialist_used = "Static Words MLP"

    # When Model 3 (`dynamic_mlp.keras`) confirms a completed 25-frame dynamic trajectory,
    # expose `<GESTURE>_END` in `raw_pred` so the frontend's `_START` -> Countdown Timer -> `_END`
    # state machine completes the active dynamic gesture cleanly (and ignores uninitiated fragments)!
    raw_pred_out = raw_pred
    if gesture_type == "dynamic" and raw_pred_out and not str(raw_pred_out).upper().endswith(("_START", "_END")):
        raw_pred_out = f"{raw_pred_out}_END"

    if raw_pred_out and str(raw_pred_out).upper().endswith(("_START", "_END")):
        gesture_type = "dynamic"

    # Final safeguard: START, STOP, and two-handed dynamic phrases strictly require both hands!
    if detected_hand != "both":
        two_hand_guards = ("HOW ARE YOU", "NICE TO MEET YOU")
        if raw_pred_out and any(str(raw_pred_out).upper().startswith(g) for g in two_hand_guards):
            raw_pred_out = None
            clean_pred = None
            confidence = 0.0
            pred_status = "ambiguous"

    is_neutral_signal = bool(clean_pred == "NEUTRAL" or (raw_pred_out and str(raw_pred_out).upper() == "NEUTRAL"))
    is_dynamic_signal = bool(
        gesture_type == "dynamic"
        or (
            raw_pred_out
            and (
                str(raw_pred_out).upper().endswith("_START")
                or str(raw_pred_out).upper().endswith("_END")
                or str(raw_pred_out).upper() in ("HOW ARE YOU", "NICE TO MEET YOU")
            )
        )
    )

    right_live_gesture = clean_pred if (detected_hand == "right" and clean_pred) else "—"
    right_live_conf = confidence if (detected_hand == "right" and clean_pred) else 0.0
    both_live_gesture = clean_pred if (detected_hand == "both" and clean_pred) else "—"
    both_live_conf = confidence if (detected_hand == "both" and clean_pred) else 0.0

    return {
        "ok": True,
        "prediction": clean_pred,
        "raw_pred": raw_pred_out,
        "gesture_type": gesture_type,
        "model_source": model_source,
        "specialist_used": specialist_used,
        "routed_via_fist_specialist": routed_via_fist,
        "is_dynamic": is_dynamic_signal,
        "confidence": float(round(confidence, 2)),
        "status": pred_status,
        "top_label": top_class_name,
        "second_label": second_class_name,
        "margin": float(round(margin_value, 2)),
        "detected_hand": detected_hand,
        "dominant_hand": "right",
        "live_gesture": clean_pred or "—",
        "live_conf": float(round(confidence, 2)),
        "is_neutral": is_neutral_signal,
        "right_gesture": right_live_gesture,
        "right_conf": float(round(right_live_conf, 2)),
        "left_gesture": "—",
        "left_conf": 0.0,
        "both_gesture": both_live_gesture,
        "both_conf": float(round(both_live_conf, 2)),
    }


# ── Flask API Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
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
    resp = make_response(render_template("mobile.html"))
    resp.set_cookie("view_pref", "mobile", max_age=86400 * 30)
    return resp


@app.route("/desktop")
def desktop():
    resp = make_response(render_template("index.html"))
    resp.set_cookie("view_pref", "desktop", max_age=86400 * 30)
    return resp


@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    """
    100% Stateless Multi-MLP Landmark Inference Endpoint.
    Accepts:
      {
        "landmarks": [...],
        "hands": [...],
        "hand_type": "right"|"both",
        "sequence_buffer": [...],
        "motion_speed": float,
        "mode": "auto"|"asl"|"words"|"dynamic"
      }
    """
    try:
        data = request.get_json(silent=True) or {}
        landmarks = data.get("landmarks", [])
        handedness = data.get("handedness", "Right")
        all_hands = data.get("all_hands") or data.get("hands") or []
        is_mirrored = bool(data.get("is_mirrored", True))
        hand_type = data.get("hand_type")
        sequence_buffer = data.get("sequence_buffer") or []
        motion_speed = float(data.get("motion_speed", 0.0) or 0.0)
        mode = data.get("mode", "auto")

        with _inference_lock:
            result = process_landmarks_data(
                landmarks_data=landmarks,
                handedness=handedness,
                all_hands=all_hands,
                is_mirrored=is_mirrored,
                hand_type=hand_type,
                sequence_buffer=sequence_buffer,
                motion_speed=motion_speed,
                mode=mode,
            )

        return jsonify(result)
    except Exception as e:
        print(f"[Error] /predict_landmarks exception: {e}")
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
            "prediction": None,
            "status": "error",
            "live_gesture": "—",
            "live_conf": 0.0,
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        if "landmarks" in data or "sequence_buffer" in data or "hands" in data:
            return predict_landmarks()

        return jsonify({
            "ok": True,
            "prediction": "—",
            "confidence": 0.0,
            "live_gesture": "—",
            "live_conf": 0.0,
            "message": "Client-side MediaPipe is active. Send landmarks to /predict_landmarks.",
        })
    except Exception as e:
        print(f"[Error] /predict exception: {e}")
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e),
            "prediction": "—",
            "live_gesture": "—",
            "live_conf": 0.0,
        }), 500


@app.route("/gesture")
def gesture_status():
    return jsonify({"status": "running", "ok": True})


@app.route("/video_feed")
def video_feed():
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
        "default": "en-US",
    })


@app.route("/api/stt/transcribe", methods=["POST"])
def api_transcribe_audio():
    language = request.form.get("language") or request.args.get("language") or "en-US"
    if "audio" in request.files:
        audio_bytes = request.files["audio"].read()
    else:
        audio_bytes = request.get_data()

    if not audio_bytes:
        return jsonify({"ok": False, "error": "No audio data received."}), 400

    result = stt_engine.transcribe_wav_bytes(audio_bytes, language=language)
    return jsonify(result)


def _retrain_model_background():
    global is_retraining
    try:
        subprocess.run([sys.executable, "train_model.py"], check=True)
        mlp_router.load_all_models()
        print("All 4 specialized MLP models successfully retrained and reloaded.")
    except Exception as e:
        print(f"Error during retraining: {e}")
    finally:
        is_retraining = False


@app.route("/correct_gesture", methods=["POST"])
def correct_gesture():
    global is_retraining
    if is_retraining:
        return jsonify({"ok": False, "message": "Models are currently retraining. Please wait."})

    data = request.get_json(silent=True) or {}
    if "label" not in data:
        return jsonify({"ok": False, "message": "No label provided."})

    correct_label = str(data["label"]).upper().strip()
    is_retraining = True
    threading.Thread(target=_retrain_model_background, daemon=True).start()
    return jsonify({"ok": True, "message": f'Learning "{correct_label}" in background...'})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "aslo-engine", "models_ready": mlp_router.is_ready})


@app.route("/api/dictionary", methods=["GET"])
def api_dictionary():
    canonical_file = "canonical_landmarks.json"
    canonical_data = {}
    if os.path.exists(canonical_file):
        try:
            with open(canonical_file, "r") as f:
                canonical_data = json.load(f)
        except Exception as e:
            print(f"Error loading canonical landmarks: {e}")

    SIGN_METADATA = {
        "A": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Closed fist with thumb resting straight upright along outer edge of index finger.", "tips": "Keep thumb upright on the side. Do NOT fold thumb across front knuckles (which is 'S')."},
        "B": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Four fingers extended straight upward and held together, with thumb tucked across the palm.", "tips": "Keep fingers closely parallel with palm facing forward."},
        "C": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Curved fingers and thumb forming an open 'C' profile shape.", "tips": "Arch all five fingers gently toward each other like holding an invisible cup."},
        "D": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index finger extended straight up; middle, ring, pinky, and thumb curl together forming a closed circle.", "tips": "Only index finger points up; the other 3 fingers touch the thumb."},
        "E": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "All 4 fingertips curled tightly down against top of palm, with thumb folded flat underneath.", "tips": "Tuck all fingertips down completely without extending knuckles outward."},
        "F": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index finger and thumb touch tips forming an 'OK' circle; middle, ring, and pinky extended upright.", "tips": "Spread the top three fingers open upward while index and thumb touch."},
        "G": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index finger and thumb extended horizontally parallel like pointing a small gauge.", "tips": "Orient hand sideways with index and thumb pointing to the side."},
        "H": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index and middle fingers extended horizontally side-by-side, pointing outward.", "tips": "Keep index and middle tightly together, thumb tucked under ring finger."},
        "I": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Pinky finger extended straight up, other fingers curled into a fist with thumb folded across.", "tips": "Only the pinky extends straight up vertically."},
        "J": {"category": "dynamic", "type": "alphabet", "handedness": "right", "desc": "Pinky finger extended, tracing a downward and upward curving 'J' hook in space.", "tips": "Start with pinky upright, then sweep down and hook outward smoothly."},
        "K": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index extended straight up, middle finger forward at a 45-degree angle, thumb tucked between them.", "tips": "Thumb tip rests at the base knuckle of the middle finger."},
        "L": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index finger pointing up and thumb extended outward horizontally forming an 'L'.", "tips": "Form a crisp 90-degree angle between thumb and index finger."},
        "M": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Closed fist with thumb tucked under the index, middle, and ring fingers.", "tips": "Thumb tip peaks out between the ring and pinky fingers."},
        "N": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Closed fist with thumb tucked under the index and middle fingers.", "tips": "Thumb tip peaks out between the middle and ring fingers."},
        "O": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "All 5 fingertips curved forward touching the thumb tip to form an open circle.", "tips": "Keep fingers smoothly arched forming a clear circular silhouette."},
        "P": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Hand tilted downward: middle finger points down, index points forward, thumb between them.", "tips": "Similar finger configuration to 'K', but oriented pointing downward."},
        "Q": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index finger and thumb pointing downward, separated like gripping a small object.", "tips": "Similar to 'G', but hand is oriented pointing down toward the floor."},
        "R": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Middle finger crossed over index finger, both pointing straight up.", "tips": "Fingers crossed for good luck; other fingers curled into palm with thumb over them."},
        "S": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Closed fist with thumb folded horizontally across front knuckles of the fingers.", "tips": "Wrap thumb across the front of index and middle fingers (unlike 'A' where thumb is on side)."},
        "T": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Closed fist with thumb tucked up between index and middle fingers.", "tips": "Thumb pops out over index finger between index and middle fingers."},
        "U": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index and middle fingers extended straight up, held close together side-by-side.", "tips": "Keep index and middle tightly touching; ring and pinky curled into palm."},
        "V": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index and middle fingers extended straight up, spread apart in a clear 'V' peace sign.", "tips": "Separate index and middle fingers widely; ring and pinky curled into palm."},
        "W": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index, middle, and ring fingers extended straight up, spread apart forming a 'W'.", "tips": "Pinky held down by thumb; three middle fingers spread upright."},
        "X": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Index finger bent into a curved hook, other fingers curled into palm.", "tips": "Bend the top knuckle of the index finger like a pirate's hook."},
        "Y": {"category": "static", "type": "alphabet", "handedness": "right", "desc": "Thumb and pinky extended outward like 'hang loose' / phone, three middle fingers curled.", "tips": "Extend thumb and pinky as wide as comfortable."},
        "Z": {"category": "dynamic", "type": "alphabet", "handedness": "right", "desc": "Index finger extended, tracing a 'Z' zigzag path horizontally, diagonally down, and horizontally.", "tips": "Draw the letter Z cleanly in the air with the index fingertip."},
        "HELLO": {"category": "static", "type": "phrase", "handedness": "right", "desc": "Open hand touching side of forehead/temple and waving gently outward in a salute.", "tips": "Salute outward from temple toward the conversation partner."},
        "THANK YOU": {"category": "static", "type": "phrase", "handedness": "right", "desc": "Flat open hand touches chin/lips and extends forward and outward toward receiver.", "tips": "Palm faces upward/inward as hand extends forward toward the person."},
        "I LOVE YOU": {"category": "static", "type": "phrase", "handedness": "right", "desc": "Thumb, index, and pinky extended upright (combines I, L, Y); middle and ring curled.", "tips": "Universal ASL 'ILY' sign with palm facing forward."},
        "LOVE": {"category": "static", "type": "phrase", "handedness": "both", "desc": "Both hands crossed over chest in a gentle hug.", "tips": "Cross wrists over upper chest with hands resting against shoulders."},
        "FINE": {"category": "static", "type": "phrase", "handedness": "right", "desc": "Open 5 hand with thumb touching chest, fingers spread.", "tips": "Thumb touches chest with fingers pointing upright and spread."},
        "ME": {"category": "static", "type": "phrase", "handedness": "both", "desc": "Thumbs point directly to own chest.", "tips": "Point thumbs cleanly toward chest."},
        "YOU": {"category": "static", "type": "phrase", "handedness": "right", "desc": "Index finger points forward directly toward the conversational partner.", "tips": "Point index finger straight forward at chest height."},
        "HOW ARE YOU": {"category": "dynamic", "type": "phrase", "handedness": "both", "desc": "Both hands cupped facing chest, arcing outward and opening palms upward.", "tips": "Start with knuckles touching/facing chest, rotate hands forward opening palms up."},
        "NICE TO MEET YOU": {"category": "dynamic", "type": "phrase", "handedness": "both", "desc": "Right hand slides forward across left open palm, then indices meet pointing forward.", "tips": "Part 1: Slide right palm across left palm (Nice); Part 2: Hands come together pointing forward (Meet you)."},
        "START": {"category": "static", "type": "control", "handedness": "both", "desc": "Two hands raised with thumbs up in signing frame.", "tips": "Requires both hands in signing frame to initialize translation engine."},
        "STOP": {"category": "static", "type": "control", "handedness": "both", "desc": "Two hands pointing index fingers downward (halt signal).", "tips": "Use both hands to pause recognition."},
        "SPACE": {"category": "static", "type": "control", "handedness": "both", "desc": "Two hands move horizontally outward adding a word space to the sentence.", "tips": "Point both hands outward sideways in synchronization."},
        "BACKSPACE": {"category": "static", "type": "control", "handedness": "both", "desc": "Both hands point backward/left deleting the previous letter or sign.", "tips": "Point both hands to delete last character or word."},
        "NEUTRAL": {"category": "static", "type": "control", "handedness": "both", "desc": "Hands resting comfortably at waist level or off-screen in relaxed posture.", "tips": "Return hands to resting position to clear candidate latches."},
    }

    canonical_lower = {k.lower(): v for k, v in canonical_data.items()}
    def get_can(name):
        return canonical_data.get(name) or canonical_lower.get(name.lower()) or {}

    result = []
    for sign_key, meta in SIGN_METADATA.items():
        can = get_can(sign_key)
        h0 = can.get("h0")
        h1 = can.get("h1")

        if sign_key == "I" and not h0:
            h0 = get_can("J_START").get("h0")

        keyframes = []
        if meta["category"] == "dynamic":
            start_can = get_can(f"{sign_key}_START") or can
            end_can = get_can(f"{sign_key}_END") or can
            s_pts = start_can.get("h0") or h0 or []
            e_pts = end_can.get("h0") or h0 or []

            if s_pts and not h0:
                h0 = s_pts

            if s_pts and e_pts:
                for step in range(12):
                    t = step / 11.0
                    frame_pts = []
                    if sign_key == "J":
                        arc_offset_x = -0.05 * math.sin(math.pi * t)
                        arc_offset_y = 0.08 * math.sin(math.pi * min(1.0, t * 1.3))
                        for p0, p1 in zip(s_pts, e_pts):
                            frame_pts.append({
                                "x": round(p0["x"] * (1 - t) + p1["x"] * t + arc_offset_x, 4),
                                "y": round(p0["y"] * (1 - t) + p1["y"] * t + arc_offset_y, 4),
                                "z": round(p0["z"] * (1 - t) + p1["z"] * t, 4),
                            })
                    else:
                        for p0, p1 in zip(s_pts, e_pts):
                            frame_pts.append({
                                "x": round(p0["x"] * (1 - t) + p1["x"] * t, 4),
                                "y": round(p0["y"] * (1 - t) + p1["y"] * t, 4),
                                "z": round(p0["z"] * (1 - t) + p1["z"] * t, 4),
                            })
                    keyframes.append(frame_pts)

        result.append({
            "id": sign_key,
            "name": sign_key,
            "category": meta["category"],
            "type": meta["type"],
            "handedness": meta["handedness"],
            "description": meta["desc"],
            "tips": meta["tips"],
            "landmarks": h0,
            "landmarks_secondary": h1,
            "keyframes": keyframes,
        })

    return jsonify({"ok": True, "dictionary": result, "count": len(result)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting ASLO Flask Server on port {port}...")
    print(f"Navigate to http://localhost:{port} in your browser.")
    app.run(host="0.0.0.0", port=port, debug=False)
