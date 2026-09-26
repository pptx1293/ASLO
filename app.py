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
from flask import Flask, Response, jsonify, render_template, request
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

# ── ASLO High-Performance Pipeline Modules ─────────────────────────────────
low_light_enhancer = LowLightEnhancer(base_clip_limit=2.5, tile_grid_size=(8, 8))
keypoint_interpolator = KeypointInterpolator(alpha=0.65, max_missing_frames=6)
motion_router = MotionBasedRouter(
    window_size=5, motion_threshold=0.055, stationary_threshold=0.022
)

MIN_DYNAMIC_DURATION = 0.30
MAX_DYNAMIC_DURATION = 2.80
dynamic_manager = TimerDynamicGestureManager(
    min_duration=MIN_DYNAMIC_DURATION,
    max_duration=MAX_DYNAMIC_DURATION,
    pause_speed_threshold=0.020,
    pause_required_frames=4,
    max_buffer_frames=32,
)

current_sentence = []
current_word = ""
N_FRAMES = 10
CONFIRM_THRESHOLD = 0.40
buffer_predictions = collections.deque(maxlen=N_FRAMES)
is_recording = True  # Active by default: translates signs immediately without requiring manual start
DYNAMIC_GESTURES = {"how are you", "j", "nice to meet you", "z"}
last_dynamic_trigger_time = 0.0

_SMOOTH_WINDOW = 3
_prob_buffer_both = collections.deque(maxlen=_SMOOTH_WINDOW)
_prob_buffer_single = collections.deque(maxlen=_SMOOTH_WINDOW)
_prob_buffer_right = collections.deque(maxlen=_SMOOTH_WINDOW)
_prob_buffer_left = collections.deque(maxlen=_SMOOTH_WINDOW)

TWO_HANDED_LABELS = {
    "START", "STOP", "SPACE", "BACK SPACE", "BACKSPACE", "NEUTRAL",
    "FINE",
    "HOW ARE YOU_START", "HOW ARE YOU_END",
    "NICE TO MEET YOU_START", "NICE TO MEET YOU_END"
}
LEFT_HAND_ALLOWED_LABELS = {
    "START", "STOP", "SPACE", "BACK SPACE", "BACKSPACE", "NEUTRAL"
}

dominant_hand = "right"
locked_hand = "none"
locked_counter = 0

live_gesture = "—"
live_conf = 0.0
detected_hand = "none"
right_live_gesture = "—"
right_live_conf = 0.0
left_live_gesture = "—"
left_live_conf = 0.0
both_live_gesture = "—"
both_live_conf = 0.0
agreeing = False
latest_features = None
is_retraining = False
route_mode = "STATIC"
motion_speed = 0.0
speak_alert = None

pending_dynamic_gesture = None
dynamic_timer_start = 0.0
dynamic_status_msg = ""
dynamic_status_time = 0.0
dynamic_start_counter = 0

last_trigger_time = 0.0
REPEAT_DELAY = 1.3
prev_active_hand = "none"
consecutive_no_hand = 0
start_trigger_count = 0
stop_trigger_count = 0

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
    if not label or label in ("—", "none", "NONE", "IDLE"):
        return "—"
    folder = get_clean_folder_label(label)
    if folder:
        return folder
    cleaned = re.sub(r'_(START|END)$', '', str(label), flags=re.IGNORECASE).strip()
    return cleaned if cleaned else "—"


def apply_clahe_preprocessing(frame_bgr):
    return low_light_enhancer.process(frame_bgr)


def process_landmarks_data(landmarks_data, handedness="Right", all_hands=None, is_mirrored=True, hand_type=None):
    """
    Core cloud inference engine. Processes 21-point coordinates sent from client browser:
    1. Zero-latency: uses client MediaPipe landmarks, saving 100% video bandwidth & decoding CPU
    2. Enforces dataset validation: Single Left-hand gestures are rejected (dataset only has Right and Both hands)
    3. Routes single Right-hand data to single-hand extractor, and Two-hand data to two-hand extractor
    4. Evaluates static and dynamic ASL gestures
    5. Returns full telemetry state for client UI
    """
    global current_sentence, current_word, buffer_predictions, is_recording
    global live_gesture, live_conf, agreeing, latest_features, detected_hand
    global dominant_hand, locked_hand, locked_counter
    global right_live_gesture, right_live_conf, left_live_gesture, left_live_conf
    global both_live_gesture, both_live_conf
    global route_mode, motion_speed, speak_alert
    global pending_dynamic_gesture, dynamic_timer_start, dynamic_status_msg, dynamic_status_time
    global last_dynamic_trigger_time, dynamic_start_counter
    global last_trigger_time, prev_active_hand, consecutive_no_hand
    global start_trigger_count, stop_trigger_count

    speak_alert = None
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
        print("[Validation] Rejected Left-Hand-Only sign: ASLO dataset only supports Right Hand and Both Hands gestures.")
        buffer_predictions.clear()
        detected_hand = "left_ignored"
        live_gesture = "—"
        live_conf = 0.0
        return {
            "ok": False,
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
            "word": "".join(current_sentence),
            "active": is_recording
        }

    if not hands_detected:
        consecutive_no_hand += 1
        keypoint_interpolator.update(None, "Right")
        keypoint_interpolator.update(None, "Left")
        motion_router.update(None)
        dynamic_manager.pause_timer()
    else:
        consecutive_no_hand = 0
        if dynamic_manager.is_paused:
            dynamic_manager.resume_timer()
            dynamic_timer_start = dynamic_manager.start_time

    TRACKING_FAILURE_THRESHOLD = 8
    tracking_failed = consecutive_no_hand >= TRACKING_FAILURE_THRESHOLD

    if tracking_failed and dynamic_manager.is_active:
        dynamic_manager.cancel("Tracking lost completely")
        pending_dynamic_gesture = None
        dynamic_timer_start = 0.0
        dynamic_status_msg = "CANCELLED: Hand tracking lost"
        dynamic_status_time = time.time()

    route_mode = "STATIC"
    motion_speed = 0.0
    coords_r, coords_l = None, None
    active_coords = None
    metrics_r, metrics_l = {}, {}
    landmarks_out = []

    if hands_detected:
        right_hand_lms = None
        left_hand_lms = None

        def _to_mirrored(raw_pts):
            if not raw_pts:
                return []
            return [
                [(1.0 - p[0]) if is_mirrored else p[0], p[1], p[2] if len(p) > 2 else 0.0]
                for p in raw_pts
            ]

        is_single_mode = True
        single_hand_lms = None

        if all_hands and len(all_hands) >= 2:
            # 2 hands detected: Route ONLY to two-hand feature extractors
            h0_pts = all_hands[0].get("points", [])
            h1_pts = all_hands[1].get("points", [])
            h0_label = all_hands[0].get("label", "Right")
            h1_label = all_hands[1].get("label", "Left")

            if h0_label == "Right" and h1_label == "Left":
                r_pts, l_pts = h0_pts, h1_pts
            elif h0_label == "Left" and h1_label == "Right":
                r_pts, l_pts = h1_pts, h0_pts
            else:
                # Spatial fallback in mirrored space: higher X is user's right hand
                avg_x0 = sum(p[0] for p in h0_pts) / len(h0_pts) if h0_pts else 0.5
                avg_x1 = sum(p[0] for p in h1_pts) / len(h1_pts) if h1_pts else 0.5
                if avg_x0 <= avg_x1:
                    r_pts, l_pts = h0_pts, h1_pts
                else:
                    r_pts, l_pts = h1_pts, h0_pts

            right_hand_lms = aslo_features.HandLandmarks(_to_mirrored(r_pts))
            left_hand_lms = aslo_features.HandLandmarks(_to_mirrored(l_pts))
            single_hand_lms = right_hand_lms
            is_single_mode = False
        else:
            # Single Right hand detected: Route ONLY to single right-hand feature extractors
            single_pts = _to_mirrored(landmarks_data)
            single_lms = aslo_features.HandLandmarks(single_pts)
            single_hand_lms = single_lms
            right_hand_lms = single_lms
            left_hand_lms = None
            is_single_mode = True

        coords_r, interp_r, metrics_r = keypoint_interpolator.update(right_hand_lms, "Right")
        coords_l, interp_l, metrics_l = keypoint_interpolator.update(left_hand_lms, "Left")

        active_coords = coords_r if coords_r is not None else coords_l
        active_scale = metrics_r.get("hand_scale", 0.15) if coords_r is not None else metrics_l.get("hand_scale", 0.15)
        route_mode, motion_speed = motion_router.update(active_coords, hand_scale=active_scale)

        raw_pred = "—"
        if model is None:
            raw_pred = "HAND DETECTED"
            raw_label = "HAND DETECTED"
            confidence = 1.0
            threshold = 0.5
            detected_hand = "both" if (right_hand_lms and left_hand_lms) else ("left" if left_hand_lms else "right")
        elif not is_single_mode and right_hand_lms is not None and left_hand_lms is not None:
            # ── CASE 1: BOTH HANDS DETECTED ──
            features_both = aslo_features.extract_two_hand_features(right_hand_lms, left_hand_lms)
            latest_features = features_both
            p_both = np.array(model(np.array([features_both], dtype=np.float32), training=False))[0]
            _prob_buffer_both.append(p_both)
            avg_both = np.mean(_prob_buffer_both, axis=0)
            idx_both = int(np.argmax(avg_both))
            conf_both = float(avg_both[idx_both])
            label_both = str(label_classes[idx_both])
            both_live_gesture = label_both.upper()
            both_live_conf = conf_both

            feats_r = aslo_features.extract_single_hand_features(right_hand_lms, is_left_hand=False)
            p_r = np.array(model(np.array([feats_r], dtype=np.float32), training=False))[0]
            _prob_buffer_right.append(p_r)
            avg_r = np.mean(_prob_buffer_right, axis=0)
            idx_r = int(np.argmax(avg_r))
            conf_r = float(avg_r[idx_r])
            label_r = str(label_classes[idx_r])
            heur_r = aslo_features.apply_heuristics(right_hand_lms, label_r, is_left_hand=False, confidence=conf_r, route_mode=route_mode)
            right_live_gesture = heur_r
            right_live_conf = conf_r

            feats_l = aslo_features.extract_single_hand_features(left_hand_lms, is_left_hand=True)
            p_l = np.array(model(np.array([feats_l], dtype=np.float32), training=False))[0]
            _prob_buffer_left.append(p_l)
            avg_l = np.mean(_prob_buffer_left, axis=0)
            idx_l = int(np.argmax(avg_l))
            conf_l = float(avg_l[idx_l])
            label_l = str(label_classes[idx_l])
            left_live_gesture = label_l
            left_live_conf = conf_l

            y_r = right_hand_lms.landmark[0].y
            y_l = left_hand_lms.landmark[0].y
            both_raised = (y_r < 0.85 and y_l < 0.85 and abs(y_r - y_l) < 0.40)
            is_two_handed = (
                both_raised and (
                    label_both.upper() in TWO_HANDED_LABELS
                    or label_both.upper() in ("HOW ARE YOU_START", "HOW ARE YOU_END", "NICE TO MEET YOU_START", "NICE TO MEET YOU_END")
                )
            )
            is_control_sign = label_both.upper() in ("START", "STOP", "SPACE", "BACKSPACE", "BACK SPACE")
            req_conf = 0.30 if is_control_sign else 0.40

            if is_two_handed and conf_both >= req_conf:
                raw_pred = label_both
                raw_label = label_both
                confidence = conf_both
                threshold = req_conf
                detected_hand = "both"
            else:
                raw_pred = label_r
                threshold = CONFIRM_THRESHOLD
                raw_label = heur_r
                confidence = conf_r
                detected_hand = "right"

        else:
            # ── CASE 2: SINGLE ACTIVE SIGNING HAND ──
            # Translates full ASL alphabet (A-Z) and single-hand signs.
            # Strictly routed to Right hand features without mirroring or Left-hand confusion.
            _prob_buffer_both.clear()
            detected_hand = "right"

            feats_single = aslo_features.extract_single_hand_features(single_hand_lms, is_left_hand=False)
            latest_features = feats_single
            input_tensor = np.array([feats_single], dtype=np.float32)
            probs = np.array(model(input_tensor, training=False))[0]

            _prob_buffer_single.append(probs)
            avg_s = np.mean(_prob_buffer_single, axis=0)
            final_idx = int(np.argmax(avg_s))
            raw_pred = str(label_classes[final_idx])
            confidence = float(avg_s[final_idx])
            threshold = 0.30 if raw_pred.upper() in ("START", "STOP", "SPACE", "BACKSPACE") else CONFIRM_THRESHOLD

            raw_label = aslo_features.apply_heuristics(
                single_hand_lms, raw_pred, is_left_hand=False, confidence=confidence, route_mode=route_mode
            )

            # Suppress two-handed phrase signs if only one hand is raised
            cand_clean = clean_display_label(raw_label).lower()
            if cand_clean in ("how are you", "nice to meet you", "fine"):
                raw_label = "—"
                confidence = 0.0

            right_live_gesture = raw_label
            right_live_conf = confidence
            left_live_gesture = "—"
            left_live_conf = 0.0
            both_live_gesture = "—"
            both_live_conf = 0.0

        live_gesture = raw_label
        live_conf = confidence
    else:
        _prob_buffer_both.clear()
        _prob_buffer_single.clear()
        _prob_buffer_right.clear()
        _prob_buffer_left.clear()
        raw_pred = "—"
        raw_label = "—"
        confidence = 0.0
        threshold = CONFIRM_THRESHOLD
        detected_hand = "none"
        right_live_gesture = "—"
        right_live_conf = 0.0
        left_live_gesture = "—"
        left_live_conf = 0.0
        both_live_gesture = "—"
        both_live_conf = 0.0
        live_gesture = "—"
        live_conf = 0.0

    # 3. Routing Mechanism & Dynamic Sequence Logic
    current_time = time.time()
    agreeing = False
    is_dynamic_label = live_gesture.upper().endswith("_START") or live_gesture.upper().endswith("_END")

    if detected_hand != prev_active_hand and detected_hand != "none" and prev_active_hand != "none":
        is_both_right_fluctuation = (set([detected_hand, prev_active_hand]) == set(["both", "right"]))
        if not is_both_right_fluctuation:
            buffer_predictions.clear()
    prev_active_hand = detected_hand
    buffer_predictions.append(live_gesture)

    clean_target = get_clean_folder_label(live_gesture)
    cand_lower = clean_target.lower() if clean_target else ""
    is_first_frame = live_gesture.upper().endswith("_START") and (cand_lower in DYNAMIC_GESTURES)

    # A: If Dynamic Sequence is Active -> Process frame buffer
    if dynamic_manager.is_active:
        is_neutral = (live_gesture.upper() == "NEUTRAL")
        is_end = live_gesture.upper().endswith("_END")

        flat_coords, completed_name, dyn_msg = dynamic_manager.update(
            coords=active_coords if hands_detected else None,
            motion_speed=motion_speed,
            is_neutral=is_neutral,
            is_end_label=is_end
        )
        pending_dynamic_gesture = dynamic_manager.gesture_name
        dynamic_timer_start = dynamic_manager.start_time

        if dyn_msg:
            dynamic_status_msg = dyn_msg
            dynamic_status_time = current_time

        if completed_name and flat_coords is not None:
            folder_label = get_clean_folder_label(completed_name)
            if folder_label:
                last_dynamic_trigger_time = current_time
                if is_recording:
                    current_sentence.append(folder_label)
                    current_word = folder_label
                    last_trigger_time = current_time
                    dynamic_status_msg = f"Translated: {folder_label}"
                    dynamic_status_time = current_time
                    print(f"[Dynamic Gesture] Translated '{completed_name}' as '{folder_label}'")
                else:
                    dynamic_status_msg = f"{folder_label} (ignored: paused)"
                    dynamic_status_time = current_time
            dynamic_manager.reset()
            pending_dynamic_gesture = None
            dynamic_timer_start = 0.0
            buffer_predictions.clear()

    # B: Initiate new dynamic gesture sequence
    elif is_first_frame and (current_time - last_dynamic_trigger_time) > 1.2:
        dynamic_start_counter += 1
        if dynamic_start_counter >= 2:
            dynamic_start_counter = 0
            candidate_gesture = cand_lower
            dynamic_manager.start_gesture(candidate_gesture, initial_coords=active_coords)
            pending_dynamic_gesture = candidate_gesture
            dynamic_timer_start = dynamic_manager.start_time
            dynamic_status_msg = f"Started '{candidate_gesture}'"
            dynamic_status_time = current_time
            buffer_predictions.clear()

    # C: Route to Static Gesture or Control Signs
    elif not dynamic_manager.is_active:
        dynamic_start_counter = 0

        if is_dynamic_label:
            pass
        else:
            is_start = (
                cand_lower == "start"
                or (both_live_gesture.lower() == "start" and both_live_conf >= 0.30)
                or (right_live_gesture.lower() == "start" and right_live_conf >= 0.50)
            )
            is_stop = (
                cand_lower == "stop"
                or (both_live_gesture.lower() == "stop" and both_live_conf >= 0.30)
                or (right_live_gesture.lower() == "stop" and right_live_conf >= 0.50)
            )

            if is_start:
                start_trigger_count += 1
                stop_trigger_count = 0
                if start_trigger_count >= 2:
                    if not is_recording:
                        is_recording = True
                        speak_alert = "Recording start"
                        dynamic_status_msg = "Recording started"
                        dynamic_status_time = current_time
                        print("[Gesture Control] >>> RECORDING STARTED (Active)")
                    buffer_predictions.clear()
            elif is_stop:
                stop_trigger_count += 1
                start_trigger_count = 0
                if stop_trigger_count >= 2:
                    if is_recording:
                        is_recording = False
                        speak_alert = "Recording stop"
                        dynamic_status_msg = "Recording stopped"
                        dynamic_status_time = current_time
                        print("[Gesture Control] >>> RECORDING STOPPED (Standby)")
                    buffer_predictions.clear()
            else:
                start_trigger_count = 0
                stop_trigger_count = 0

            if confidence > threshold:
                count = list(buffer_predictions).count(live_gesture)
                agreeing = len(buffer_predictions) == N_FRAMES and count >= 3

                if agreeing:
                    clean_lbl = clean_target or get_clean_folder_label(live_gesture)
                    if clean_lbl:
                        clean_upper = clean_lbl.upper()
                        if clean_upper in ("FINE", "HOW ARE YOU", "NICE TO MEET YOU") and detected_hand != "both":
                            agreeing = False

                    if agreeing and clean_lbl and clean_lbl.lower() not in ("start", "stop"):
                        predicted_label = clean_lbl
                        lbl_lower = predicted_label.lower()

                        if lbl_lower in ("backspace", "back space"):
                            if (current_time - last_trigger_time) > REPEAT_DELAY:
                                if len(current_sentence) > 0:
                                    current_sentence.pop()
                                last_trigger_time = current_time
                                buffer_predictions.clear()
                        elif lbl_lower == "space":
                            if (current_time - last_trigger_time) > REPEAT_DELAY:
                                if is_recording:
                                    current_sentence.append(" ")
                                last_trigger_time = current_time
                                buffer_predictions.clear()
                        else:
                            should_trigger = (predicted_label != current_word) or (
                                predicted_label != "—"
                                and (current_time - last_trigger_time) > REPEAT_DELAY
                            )

                            if should_trigger:
                                current_word = predicted_label
                                last_trigger_time = current_time
                                if is_recording:
                                    current_sentence.append(predicted_label)
                                    print(f"[Static Sign] >>> RECORDED TO SENTENCE: '{predicted_label}' | Current Sentence: '{''.join(current_sentence)}'")
                                else:
                                    print(f"[Static Sign] Ignored '{predicted_label}' (Recording is PAUSED / STANDBY)")
                                buffer_predictions.clear()
            else:
                agreeing = False
                if list(buffer_predictions).count("—") >= 4:
                    current_word = "—"

        if hands_detected:
            buf_match = list(buffer_predictions).count(live_gesture)
            print(f"[Inference] Hand={detected_hand} | Raw={raw_pred} ({confidence:.2f}) -> Live={live_gesture} | Agree={agreeing} ({buf_match}/{len(buffer_predictions)}) | Rec={is_recording}")

    agreeing = bool(dynamic_manager.is_active or agreeing)

    if dynamic_manager.is_active:
        time_left = round(max(0.0, dynamic_manager.get_remaining_time()), 1)
        active_dyn = dynamic_manager.gesture_name
    elif pending_dynamic_gesture:
        time_left = round(max(0.0, MAX_DYNAMIC_DURATION - (current_time - dynamic_timer_start)), 1)
        active_dyn = pending_dynamic_gesture
    else:
        time_left = 0.0
        active_dyn = None

    alert_to_send = speak_alert
    speak_alert = None

    clean_live = clean_display_label(live_gesture)
    clean_r = clean_display_label(right_live_gesture) if right_live_gesture != "TRACKED" else "TRACKED"
    clean_l = clean_display_label(left_live_gesture) if left_live_gesture != "TRACKED" else "TRACKED"
    clean_both = clean_display_label(both_live_gesture)
    clean_dyn = clean_display_label(active_dyn) if active_dyn else None

    return {
        "ok": True,
        "prediction": clean_live,
        "confidence": float(round(live_conf, 2)),
        "status": "running",
        "detected_hand": detected_hand,
        "dominant_hand": dominant_hand,
        "route_mode": route_mode,
        "motion_speed": round(motion_speed, 3),
        "live_gesture": clean_live,
        "live_conf": round(live_conf, 2),
        "right_gesture": clean_r,
        "right_conf": round(right_live_conf, 2),
        "left_gesture": clean_l,
        "left_conf": round(left_live_conf, 2),
        "both_gesture": clean_both,
        "both_conf": round(both_live_conf, 2),
        "buffer_fill": len(buffer_predictions),
        "buffer_max": N_FRAMES,
        "word": "".join(current_sentence),
        "active": is_recording,
        "agreeing": agreeing,
        "dynamic_gesture": clean_dyn,
        "dynamic_time_left": time_left,
        "max_dynamic_duration": MAX_DYNAMIC_DURATION,
        "speak_alert": alert_to_send,
        "landmarks": landmarks_out,
    }


# ── Flask API Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/mobile")
def mobile():
    return render_template("mobile.html")


@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    """Client-side landmark inference endpoint.
    Accepts lightweight 21-point hand landmark coordinates JSON.
    Zero image decoding, zero server-side MediaPipe, maximum speed on Render."""
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
            "live_conf": 0.0,
            "word": "".join(current_sentence),
            "active": is_recording
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
            "active": is_recording,
            "word": "".join(current_sentence),
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
            "live_conf": 0.0,
            "word": "".join(current_sentence),
            "active": is_recording
        }), 500


@app.route("/gesture")
def gesture_status():
    """Telemetry status poll endpoint."""
    current_time = time.time()
    if dynamic_manager.is_active:
        time_left = round(max(0.0, dynamic_manager.get_remaining_time()), 1)
        active_dyn = dynamic_manager.gesture_name
    elif pending_dynamic_gesture:
        time_left = round(max(0.0, MAX_DYNAMIC_DURATION - (current_time - dynamic_timer_start)), 1)
        active_dyn = pending_dynamic_gesture
    else:
        time_left = 0.0
        active_dyn = None

    clean_live = clean_display_label(live_gesture)
    clean_r = clean_display_label(right_live_gesture) if right_live_gesture != "TRACKED" else "TRACKED"
    clean_l = clean_display_label(left_live_gesture) if left_live_gesture != "TRACKED" else "TRACKED"
    clean_both = clean_display_label(both_live_gesture)
    clean_dyn = clean_display_label(active_dyn) if active_dyn else None

    return jsonify({
        "status": "running",
        "camera_index": 0,
        "available_cameras": [0],
        "live_gesture": clean_live,
        "live_conf": live_conf,
        "detected_hand": detected_hand,
        "dominant_hand": dominant_hand,
        "route_mode": route_mode,
        "motion_speed": round(motion_speed, 3),
        "right_gesture": clean_r,
        "right_conf": right_live_conf,
        "left_gesture": clean_l,
        "left_conf": left_live_conf,
        "both_gesture": clean_both,
        "both_conf": both_live_conf,
        "buffer_fill": len(buffer_predictions),
        "buffer_max": N_FRAMES,
        "agreeing": agreeing,
        "active": is_recording,
        "word": "".join(current_sentence),
        "dynamic_gesture": clean_dyn,
        "dynamic_time_left": time_left,
        "max_dynamic_duration": MAX_DYNAMIC_DURATION,
        "speak_alert": None,
    })


@app.route("/video_feed")
def video_feed():
    """Dummy endpoint for backwards compatibility."""
    return Response(b"", mimetype="text/plain")


@app.route("/backspace", methods=["POST"])
def backspace():
    global current_sentence, current_word
    if current_sentence:
        current_sentence.pop()
    current_word = current_sentence[-1] if current_sentence else ""
    return jsonify({"ok": True, "word": "".join(current_sentence)})


@app.route("/clear", methods=["POST"])
def clear():
    global current_sentence, current_word, buffer_predictions
    current_sentence = []
    current_word = ""
    buffer_predictions.clear()
    return jsonify({"status": "cleared"})


@app.route("/set_dominant_hand", methods=["POST"])
def set_dominant_hand():
    global dominant_hand, locked_hand, locked_counter
    global _prob_buffer_both, _prob_buffer_right, _prob_buffer_left, buffer_predictions
    data = request.get_json(silent=True) or {}
    if "hand" in data:
        hand = str(data["hand"]).lower().strip()
        if hand in ("right", "left", "auto"):
            dominant_hand = hand
            locked_hand = "none"
            locked_counter = 0
            _prob_buffer_both.clear()
            _prob_buffer_right.clear()
            _prob_buffer_left.clear()
            buffer_predictions.clear()
            return jsonify({"ok": True, "dominant_hand": dominant_hand})
    return jsonify({"ok": False, "error": "Invalid hand selection"}), 400


@app.route("/get_dominant_hand", methods=["GET"])
def get_dominant_hand():
    return jsonify({"ok": True, "dominant_hand": dominant_hand})


@app.route("/toggle_active", methods=["POST"])
def toggle_active():
    global is_recording
    data = request.get_json(silent=True) or {}
    if "active" in data:
        is_recording = bool(data["active"])
    else:
        is_recording = not is_recording
    return jsonify({"ok": True, "active": is_recording})


@app.route("/append_text", methods=["POST"])
def append_text():
    global current_sentence
    data = request.get_json(silent=True) or {}
    if "text" in data:
        current_sentence.append(" " + str(data["text"]).strip() + " ")
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
    global current_sentence
    language = request.form.get("language") or request.args.get("language") or "en-US"
    append = (request.form.get("append", "false")).lower() in ("true", "1", "yes")

    audio_bytes = None
    if "audio" in request.files:
        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()
    else:
        audio_bytes = request.get_data()

    if not audio_bytes:
        return jsonify({"ok": False, "error": "No audio data received."}), 400

    result = stt_engine.transcribe_wav_bytes(audio_bytes, language=language)
    if result.get("ok") and append and result.get("text"):
        current_sentence.append(" " + result["text"] + " ")

    return jsonify(result)


def _retrain_model_background():
    global is_retraining, model, label_classes
    global _prob_buffer_both, _prob_buffer_right, _prob_buffer_left, buffer_predictions
    try:
        subprocess.run([sys.executable, "train_model.py"], check=True)
        model = models.load_model(model_path)
        label_classes = np.load(labels_path, allow_pickle=True)
        _prob_buffer_both.clear()
        _prob_buffer_right.clear()
        _prob_buffer_left.clear()
        buffer_predictions.clear()
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
