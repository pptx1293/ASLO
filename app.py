import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from keras import models
import collections
import csv
import subprocess
import sys
import traceback

import aslo_features
from aslo_pipeline import (
    LowLightEnhancer,
    KeypointInterpolator,
    MotionBasedRouter,
    TimerDynamicGestureManager,
)
from speech_to_text import SUPPORTED_LANGUAGES, engine as stt_engine

app = Flask(__name__, static_folder="templates", static_url_path="")
app.config['TEMPLATES_AUTO_RELOAD'] = True

model_path = "gesture_model.keras"
labels_path = "label_classes.npy"

model = None
label_classes = None
camera_running = False
cap = None
camera_index = 0
_switch_camera_requested = False
_target_camera_index = 0
_camera_switch_lock = threading.Lock()
_cached_cameras = [0]
_cameras_scanned = False


def scan_available_cameras(max_check=4):
    """Safely scan available cameras once before streaming starts to prevent DirectShow COM collisions."""
    global _cached_cameras, _cameras_scanned
    if _cameras_scanned:
        return _cached_cameras

    found = []
    for idx in range(max_check):
        try:
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                found.append(idx)
                test_cap.release()
        except Exception:
            pass

    if not found:
        found = [0]
    _cached_cameras = found
    _cameras_scanned = True
    print(f"[Camera] Detected available camera indices: {_cached_cameras}")
    return _cached_cameras


def get_available_camera_indices(max_check=4):
    """Returns cached camera list to avoid concurrent VideoCapture probing during active streaming."""
    global _cached_cameras, _cameras_scanned
    if not _cameras_scanned:
        return scan_available_cameras(max_check)
    return _cached_cameras

current_sentence = []
current_word = ""
N_FRAMES = 8
CONFIRM_THRESHOLD = 0.40
buffer_predictions = collections.deque(maxlen=N_FRAMES)
is_recording = False
DYNAMIC_GESTURES = {"how are you", "j", "nice to meet you", "z"}
last_dynamic_trigger_time = 0.0

_SMOOTH_WINDOW = 5
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

# ── Clean Dataset Folder Mapping ───────────────────────────────────────────
import re
DATASET_DIR = "picture_file"
FOLDER_NAMES_MAP = {}
if os.path.exists(DATASET_DIR):
    for fname in os.listdir(DATASET_DIR):
        if os.path.isdir(os.path.join(DATASET_DIR, fname)):
            FOLDER_NAMES_MAP[fname.lower().strip()] = fname
            FOLDER_NAMES_MAP[fname.lower().replace(" ", "_").strip()] = fname


def get_clean_folder_label(label: str):
    """
    Strips internal suffixes like _START, _END and returns the exact folder name
    from the dataset. Returns None for internal states like DYNAMIC_GESTURE,
    HAND DETECTED, NEUTRAL, or unmapped tokens so they are never translated.
    """
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
    """
    Strips internal suffixes like _START, _END and returns a clean, human-readable
    folder label for the UI (e.g. 'How are you', 'Nice to meet you', 'J', 'Z').
    Guarantees that _START and _END are NEVER shown to the user in any window or badge.
    """
    if not label or label in ("—", "none", "NONE", "IDLE"):
        return "—"
    folder = get_clean_folder_label(label)
    if folder:
        return folder
    cleaned = re.sub(r'_(START|END)$', '', str(label), flags=re.IGNORECASE).strip()
    return cleaned if cleaned else "—"


speak_alert = None

# ── Dynamic Gesture Watchdog & State Machine Variables ────────────────────
pending_dynamic_gesture = None
dynamic_timer_start = 0.0
dynamic_status_msg = ""
dynamic_status_time = 0.0
dynamic_start_counter = 0     # Requires 2 consecutive frames of the first frame (_START)
MIN_DYNAMIC_DURATION = 0.30   # Minimum seconds to avoid instantaneous triggers
MAX_DYNAMIC_DURATION = 2.80   # Watchdog timeout: if user stops early or freezes, cancel

# Timer pause state for tracking failures
dynamic_timer_paused = False
dynamic_timer_pause_accumulator = 0.0  # accumulated paused time
dynamic_timer_pause_start = 0.0        # when pause began

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

if os.path.exists(model_path) and os.path.exists(labels_path):
    print("Loading model and labels...")
    model = models.load_model(model_path)
    label_classes = np.load(labels_path, allow_pickle=True)
    print(f"Loaded {len(label_classes)} classes.")
else:
    print("Warning: gesture_model.keras or label_classes.npy not found.")
    print(
        "Please run collect_data.py and train_model.py first. Inference will not work."
    )

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# ── ASLO High-Performance Pipeline Modules ─────────────────────────────
low_light_enhancer = LowLightEnhancer(base_clip_limit=2.5, tile_grid_size=(8, 8))
keypoint_interpolator = KeypointInterpolator(alpha=0.65, max_missing_frames=6)
motion_router = MotionBasedRouter(window_size=5, motion_threshold=0.055, stationary_threshold=0.022)
dynamic_manager = TimerDynamicGestureManager(
    min_duration=MIN_DYNAMIC_DURATION,
    max_duration=MAX_DYNAMIC_DURATION,
    pause_speed_threshold=0.020,
    pause_required_frames=4,
    max_buffer_frames=32,
)


def apply_clahe_preprocessing(frame_bgr):
    """Apply adaptive CLAHE to the L-channel of LAB color space for low-light enhancement (<100 Lux).
    Returns enhanced BGR frame, mean luminance, and low-light flag."""
    return low_light_enhancer.process(frame_bgr)


_latest_frame_bytes = None
_frame_lock = threading.Lock()
_camera_thread = None


def start_camera_thread():
    global _camera_thread, camera_running
    if _camera_thread is None or not _camera_thread.is_alive():
        camera_running = True
        _camera_thread = threading.Thread(target=camera_capture_worker, daemon=True)
        _camera_thread.start()
        print("[Camera] Dedicated background capture thread started.")


def camera_capture_worker():
    global \
        cap, \
        camera_running, \
        current_sentence, \
        current_word, \
        buffer_predictions, \
        is_recording
    global live_gesture, live_conf, agreeing, latest_features, detected_hand
    global dominant_hand, locked_hand, locked_counter
    global right_live_gesture, right_live_conf, left_live_gesture, left_live_conf, both_live_gesture, both_live_conf
    global route_mode, motion_speed
    global _latest_frame_bytes
    global camera_index, _switch_camera_requested, _target_camera_index
    global pending_dynamic_gesture, dynamic_timer_start, dynamic_status_msg, dynamic_status_time
    global dynamic_timer_paused, dynamic_timer_pause_accumulator, dynamic_timer_pause_start
    global speak_alert, last_dynamic_trigger_time, dynamic_start_counter

    print(f"[Camera] Initializing VideoCapture({camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera_running = True

    last_trigger_time = 0
    REPEAT_DELAY = 1.3
    prev_active_hand = "none"
    consecutive_no_hand = 0
    start_trigger_count = 0
    stop_trigger_count = 0
    dynamic_start_counter = 0

    consecutive_read_failures = 0

    while camera_running:
        if _switch_camera_requested:
            with _camera_switch_lock:
                target_idx = _target_camera_index
                _switch_camera_requested = False
            print(f"[Camera] Switching hardware from index {camera_index} to {target_idx}...")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            camera_index = target_idx
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap is None or not cap.isOpened():
            print(f"[Camera] VideoCapture({camera_index}) not ready, attempting open...")
            cap = cv2.VideoCapture(camera_index)
            time.sleep(0.5)
            continue

        success, frame = cap.read()
        if not success or frame is None:
            consecutive_read_failures += 1
            if consecutive_read_failures >= 15:
                print(f"[Camera] Frame read failed repeatedly on camera {camera_index}, resetting device...")
                time.sleep(0.5)
                try:
                    cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(camera_index)
                consecutive_read_failures = 0
            else:
                time.sleep(0.02)
            continue
        consecutive_read_failures = 0

        try:

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # ── 1. Adaptive Lighting Preprocessing (Low-Light & High-Light / Glare)
            frame_enhanced, mean_luminance, lighting_state = apply_clahe_preprocessing(frame)

            # Process with MediaPipe on enhanced feed
            image_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # ── 2. MediaPipe Tracking Failsafe & Keypoint Interpolation ──────────
            hands_detected = bool(results and results.multi_hand_landmarks)

            if not hands_detected:
                consecutive_no_hand += 1
                keypoint_interpolator.update(None, "Right")
                keypoint_interpolator.update(None, "Left")
                motion_router.update(None)
                dynamic_manager.pause_timer()
                dynamic_timer_paused = dynamic_manager.is_paused
                dynamic_timer_pause_accumulator = dynamic_manager.pause_accumulator
            else:
                consecutive_no_hand = 0
                if dynamic_manager.is_paused:
                    dynamic_manager.resume_timer()
                    dynamic_timer_paused = False
                    dynamic_timer_start = dynamic_manager.start_time

            TRACKING_FAILURE_THRESHOLD = 8  # frames at ~30fps = ~266ms; at 15fps = ~533ms
            tracking_failed = consecutive_no_hand >= TRACKING_FAILURE_THRESHOLD

            if tracking_failed and dynamic_manager.is_active:
                # If hands disappear entirely, safely pause timer and clear buffer so empty vectors avoid MLP
                dynamic_manager.cancel("Tracking lost completely")
                pending_dynamic_gesture = None
                dynamic_timer_start = 0.0
                dynamic_timer_paused = False
                dynamic_status_msg = "CANCELLED: Hand tracking lost"
                dynamic_status_time = time.time()

            route_mode = "STATIC"
            motion_speed = 0.0
            coords_r, coords_l = None, None
            active_coords = None
            metrics_r, metrics_l = {}, {}

            if hands_detected:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                try:
                    # In horizontally mirrored selfie view:
                    # "Right" = user's physical right hand (from viewpoint)
                    # "Left"  = user's physical left hand (from viewpoint)
                    right_hand_lms = None
                    left_hand_lms = None
                    if results and results.multi_hand_landmarks:
                        num_hands = len(results.multi_hand_landmarks)
                        if num_hands >= 2:
                            # ── SPATIAL GEOMETRIC ANCHORING FOR DUAL HANDS ──
                            # In horizontally mirrored selfie view (cv2.flip(frame, 1)):
                            # User's physical Right hand is on the RIGHT side of the image (larger x)
                            # User's physical Left hand is on the LEFT side of the image (smaller x)
                            h0 = results.multi_hand_landmarks[0]
                            h1 = results.multi_hand_landmarks[1]
                            x0 = h0.landmark[0].x
                            x1 = h1.landmark[0].x
                            if x0 >= x1:
                                right_hand_lms = h0
                                left_hand_lms = h1
                            else:
                                right_hand_lms = h1
                                left_hand_lms = h0
                        elif num_hands == 1:
                            single_lms = results.multi_hand_landmarks[0]
                            # In single-hand mode:
                            # If dominant_hand == "left", user explicitly designated Left hand.
                            # Otherwise (default "right"), the single hand is the user's primary signing hand!
                            if dominant_hand == "left":
                                left_hand_lms = single_lms
                            else:
                                right_hand_lms = single_lms

                    # Keypoint filtering & interpolation (self-occlusion & >1.5m checks)
                    coords_r, interp_r, metrics_r = keypoint_interpolator.update(right_hand_lms, "Right")
                    coords_l, interp_l, metrics_l = keypoint_interpolator.update(left_hand_lms, "Left")

                    active_coords = coords_r if coords_r is not None else coords_l
                    active_scale = metrics_r.get("hand_scale", 0.15) if coords_r is not None else metrics_l.get("hand_scale", 0.15)
                    route_mode, motion_speed = motion_router.update(active_coords, hand_scale=active_scale)

                    if model is None:
                        # Model not loaded yet, but tracking is fully active and drawing landmarks
                        raw_label = "HAND DETECTED"
                        confidence = 1.0
                        threshold = 0.5
                        if right_hand_lms is not None and left_hand_lms is not None:
                            detected_hand = "both"
                            right_live_gesture = "TRACKED"
                            left_live_gesture = "TRACKED"
                            both_live_gesture = "TRACKED"
                        elif right_hand_lms is not None:
                            detected_hand = "right"
                            right_live_gesture = "TRACKED"
                            left_live_gesture = "—"
                            both_live_gesture = "—"
                        elif left_hand_lms is not None:
                            detected_hand = "left"
                            right_live_gesture = "—"
                            left_live_gesture = "TRACKED"
                            both_live_gesture = "—"
                        else:
                            detected_hand = "none"
                            right_live_gesture = "—"
                            left_live_gesture = "—"
                            both_live_gesture = "—"
                    elif right_hand_lms is not None and left_hand_lms is not None:
                        # ── CASE 1: BOTH HANDS DETECTED ──────────────────────────
                        # 1. Deterministic Two-Handed model feature extraction:
                        # Slot 0 = Right hand, Slot 1 = Left hand!
                        features_both = aslo_features.extract_two_hand_features(right_hand_lms, left_hand_lms)
                        latest_features = features_both
                        p_both = np.array(model(np.array([features_both]), training=False))[0]
                        _prob_buffer_both.append(p_both)
                        avg_both = np.mean(_prob_buffer_both, axis=0)
                        idx_both = np.argmax(avg_both)
                        conf_both = float(avg_both[idx_both])
                        label_both = str(label_classes[idx_both])
                        both_live_gesture = label_both.upper()
                        both_live_conf = conf_both

                        # 2. Evaluate Right Hand alone for single-hand signs (Slot 0 unmirrored)
                        feats_r = aslo_features.extract_single_hand_features(right_hand_lms, is_left_hand=False)
                        p_r = np.array(model(np.array([feats_r]), training=False))[0]
                        _prob_buffer_right.append(p_r)
                        avg_r = np.mean(_prob_buffer_right, axis=0)
                        idx_r = np.argmax(avg_r)
                        conf_r = float(avg_r[idx_r])
                        label_r = str(label_classes[idx_r])
                        heur_r = aslo_features.apply_heuristics(right_hand_lms, label_r, is_left_hand=False, confidence=conf_r, route_mode=route_mode)

                        right_live_gesture = heur_r
                        right_live_conf = conf_r

                        # 3. Evaluate Left Hand ONLY for control signs (START, STOP, SPACE, BACKSPACE).
                        # The Left Hand must NEVER predict or display alphabet words!
                        feats_l_raw = aslo_features.extract_single_hand_features(left_hand_lms, is_left_hand=False)
                        p_l_ctl = np.array(model(np.array([feats_l_raw]), training=False))[0]
                        idx_l = np.argmax(p_l_ctl)
                        label_l = str(label_classes[idx_l]).upper()
                        conf_l = float(p_l_ctl[idx_l])

                        if label_l in LEFT_HAND_ALLOWED_LABELS and conf_l >= 0.35:
                            left_live_gesture = label_l
                            left_live_conf = conf_l
                        else:
                            left_live_gesture = "TRACKED"
                            left_live_conf = 0.0

                        # Check if user is signing a true two-handed gesture:
                        # Both hands raised in active signing space
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
                            raw_label = label_both
                            confidence = conf_both
                            threshold = req_conf
                            detected_hand = "both"
                            active_hand = None
                            is_left = False
                        else:
                            # When both hands are in the frame, but NOT performing a 2-hand gesture:
                            # The RIGHT HAND is the active signer!
                            threshold = CONFIRM_THRESHOLD
                            raw_label = heur_r
                            confidence = conf_r
                            detected_hand = "right"
                            active_hand = right_hand_lms
                            is_left = False

                        locked_hand = detected_hand
                        locked_counter = 0

                    elif right_hand_lms is not None:
                        # ── CASE 2: DOMINANT RIGHT HAND DETECTED ──
                        detected_hand = "right"
                        single_hand = right_hand_lms

                        feats_std = aslo_features.extract_single_hand_features(single_hand, is_left_hand=False)
                        latest_features = feats_std
                        probs = np.array(model(np.array([feats_std]), training=False))[0]
                        _prob_buffer_right.append(probs)
                        avg_s = np.mean(_prob_buffer_right, axis=0)

                        final_idx = int(np.argmax(avg_s))
                        raw_pred = str(label_classes[final_idx])
                        confidence = float(avg_s[final_idx])
                        threshold = 0.30 if raw_pred.upper() in ("START", "STOP", "SPACE", "BACKSPACE") else CONFIRM_THRESHOLD

                        # Apply heuristics guarded by model confidence and route_mode
                        raw_label = aslo_features.apply_heuristics(
                            single_hand, raw_pred, is_left_hand=False, confidence=confidence, route_mode=route_mode
                        )
                        active_hand = single_hand
                        is_left = False

                        # Gating: A single right hand cannot perform two-handed dynamic gestures
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
                        locked_hand = detected_hand
                        locked_counter = 0

                    elif left_hand_lms is not None:
                        # ── CASE 3: EXPLICIT DOMINANT LEFT HAND DETECTED ──
                        _prob_buffer_both.clear()
                        _prob_buffer_right.clear()

                        detected_hand = "left"
                        single_hand = left_hand_lms

                        if dominant_hand == "left":
                            # Explicit user override: user designated Left Hand as dominant signing hand
                            feats_mirr = aslo_features.extract_single_hand_features(single_hand, is_left_hand=True)
                            latest_features = feats_mirr
                            probs = np.array(model(np.array([feats_mirr]), training=False))[0]
                            _prob_buffer_single.append(probs)
                            avg_s = np.mean(_prob_buffer_single, axis=0)

                            final_idx = int(np.argmax(avg_s))
                            raw_pred = str(label_classes[final_idx])
                            confidence = float(avg_s[final_idx])
                            threshold = CONFIRM_THRESHOLD
                            raw_label = aslo_features.apply_heuristics(
                                single_hand, raw_pred, is_left_hand=True, confidence=confidence, route_mode=route_mode
                            )
                            active_hand = single_hand
                            is_left = True

                            cand_clean = clean_display_label(raw_label).lower()
                            if cand_clean in ("how are you", "nice to meet you", "fine"):
                                raw_label = "—"
                                confidence = 0.0

                            left_live_gesture = raw_label
                            left_live_conf = confidence
                            right_live_gesture = "—"
                            right_live_conf = 0.0
                        else:
                            # Standard mode: Left hand alone is evaluated ONLY for universal control signs (START, STOP, SPACE, BACKSPACE)
                            feats_ctl = aslo_features.extract_single_hand_features(single_hand, is_left_hand=False)
                            latest_features = feats_ctl
                            probs = np.array(model(np.array([feats_ctl]), training=False))[0]
                            final_idx = int(np.argmax(probs))
                            cand_pred = str(label_classes[final_idx]).upper()
                            cand_conf = float(probs[final_idx])

                            if cand_pred in LEFT_HAND_ALLOWED_LABELS and cand_conf >= 0.40:
                                raw_label = cand_pred
                                confidence = cand_conf
                                left_live_gesture = cand_pred
                                left_live_conf = cand_conf
                            else:
                                raw_label = "—"
                                confidence = 0.0
                                left_live_gesture = "TRACKED"
                                left_live_conf = 0.0

                            active_hand = single_hand
                            is_left = True
                            threshold = CONFIRM_THRESHOLD
                            right_live_gesture = "—"
                            right_live_conf = 0.0

                        both_live_gesture = "—"
                        both_live_conf = 0.0
                        locked_hand = detected_hand
                        locked_counter = 0

                    else:
                        _prob_buffer_both.clear()
                        _prob_buffer_single.clear()
                        _prob_buffer_right.clear()
                        _prob_buffer_left.clear()
                        raw_label = "—"
                        confidence = 0.0
                        threshold = CONFIRM_THRESHOLD
                        detected_hand = "none"
                        active_hand = None
                        is_left = False
                        right_live_gesture = "—"
                        right_live_conf = 0.0
                        left_live_gesture = "—"
                        left_live_conf = 0.0
                        both_live_gesture = "—"
                        both_live_conf = 0.0
                        locked_hand = "none"
                        locked_counter = 0

                    live_gesture = raw_label
                    live_conf = confidence

                    # Draw real-time HUD badges above wrists on camera feed
                    h, w, _ = frame.shape
                    if right_hand_lms is not None:
                        wx = int(right_hand_lms.landmark[0].x * w)
                        wy = int(right_hand_lms.landmark[0].y * h)
                        disp_r = clean_display_label(right_live_gesture)
                        tag_r = f"R: {disp_r} ({int(right_live_conf * 100)}%)" if disp_r != "—" else "R: IDLE"
                        is_act_r = (detected_hand == "right" or detected_hand == "both")
                        box_c = (0, 220, 130) if is_act_r else ((120, 120, 140) if disp_r == "—" else (235, 180, 0))
                        cv2.rectangle(frame, (max(5, wx - 60), max(5, wy - 30)), (min(w - 5, wx + 60), max(30, wy - 6)), (18, 22, 30), -1)
                        cv2.rectangle(frame, (max(5, wx - 60), max(5, wy - 30)), (min(w - 5, wx + 60), max(30, wy - 6)), box_c, 1)
                        cv2.putText(frame, tag_r, (max(8, wx - 56), max(22, wy - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                    if left_hand_lms is not None:
                        wx = int(left_hand_lms.landmark[0].x * w)
                        wy = int(left_hand_lms.landmark[0].y * h)
                        disp_l = clean_display_label(left_live_gesture)
                        tag_l = f"L: {disp_l} ({int(left_live_conf * 100)}%)" if disp_l != "—" else "L: IDLE"
                        is_act_l = (detected_hand == "left" or detected_hand == "both")
                        box_c = (0, 220, 130) if is_act_l else ((120, 120, 140) if disp_l == "—" else (220, 80, 230))
                        cv2.rectangle(frame, (max(5, wx - 60), max(5, wy - 30)), (min(w - 5, wx + 60), max(30, wy - 6)), (18, 22, 30), -1)
                        cv2.rectangle(frame, (max(5, wx - 60), max(5, wy - 30)), (min(w - 5, wx + 60), max(30, wy - 6)), box_c, 1)
                        cv2.putText(frame, tag_l, (max(8, wx - 56), max(22, wy - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                except Exception as e:
                    print(f"Ignored frame error (likely model swapping): {e}")
                    _prob_buffer_both.clear()
                    _prob_buffer_single.clear()
                    _prob_buffer_right.clear()
                    _prob_buffer_left.clear()
                    live_gesture = "—"
                    live_conf = 0.0
                    confidence = 0.0
                    threshold = CONFIRM_THRESHOLD
                    detected_hand = "none"
            else:
                # ── MEDIAPIPE TRACKING FAILURE / NO HANDS DETECTED ─────────────────
                # Clear probability buffers to prevent stale predictions
                _prob_buffer_both.clear()
                _prob_buffer_single.clear()
                _prob_buffer_right.clear()
                _prob_buffer_left.clear()
                live_gesture = "—"
                live_conf = 0.0
                confidence = 0.0
                threshold = CONFIRM_THRESHOLD
                detected_hand = "none"
                right_live_gesture = "—"
                right_live_conf = 0.0
                left_live_gesture = "—"
                left_live_conf = 0.0
                both_live_gesture = "—"
                both_live_conf = 0.0
                locked_hand = "none"
                locked_counter = 0

            # ── 3. Routing Mechanism & Dynamic Sequence Logic ─────────────────────
            current_time = time.time()
            agreeing = False
            is_dynamic_label = live_gesture.upper().endswith("_START") or live_gesture.upper().endswith("_END")

            # Prevent cross-hand contamination in stability buffer, but keep buffer stable during two-handed signs
            if detected_hand != prev_active_hand and detected_hand != "none" and prev_active_hand != "none":
                # Do not flush buffer if transition is simply between 'both' and 'right' when hands are in frame
                is_both_right_fluctuation = (set([detected_hand, prev_active_hand]) == set(["both", "right"]))
                if not is_both_right_fluctuation:
                    buffer_predictions.clear()
            prev_active_hand = detected_hand
            buffer_predictions.append(live_gesture)

            # Dynamic qualification: ONLY genuine dynamic sequence classes can route to dynamic_manager
            clean_target = get_clean_folder_label(live_gesture)
            cand_lower = clean_target.lower() if clean_target else ""
            is_dyn_class = (cand_lower in DYNAMIC_GESTURES) or is_dynamic_label

            # CRITICAL: Dynamic gestures can ONLY start when the user performs the FIRST frame (_START)
            is_first_frame = live_gesture.upper().endswith("_START") and (cand_lower in DYNAMIC_GESTURES)
            time_since_dynamic = current_time - last_dynamic_trigger_time

            # A: If Dynamic Sequence is Active -> Process frame buffer and check completion
            if dynamic_manager.is_active:
                is_neutral = (live_gesture.upper() == "NEUTRAL")
                is_end = live_gesture.upper().endswith("_END")

                # Update controller: buffers coords, monitors velocity/stillness, detects terminal completion
                flat_coords, completed_name, dyn_msg = dynamic_manager.update(
                    coords=active_coords if hands_detected else None,
                    motion_speed=motion_speed,
                    is_neutral=is_neutral,
                    is_end_label=is_end
                )

                # Keep legacy globals in sync for REST endpoint compatibility
                pending_dynamic_gesture = dynamic_manager.gesture_name
                dynamic_timer_start = dynamic_manager.start_time
                dynamic_timer_paused = dynamic_manager.is_paused

                if dyn_msg:
                    dynamic_status_msg = dyn_msg
                    dynamic_status_time = current_time

                if completed_name and flat_coords is not None:
                    # Successfully completed dynamic sequence: user paused or returned to neutral at last frame
                    folder_label = get_clean_folder_label(completed_name)
                    if folder_label:
                        last_dynamic_trigger_time = current_time
                        if is_recording:
                            current_sentence.append(folder_label)
                            current_word = folder_label
                            last_trigger_time = current_time
                            dynamic_status_msg = f"Translated: {folder_label}"
                            dynamic_status_time = current_time
                            print(f"[Dynamic Gesture] SUCCESS! Translated '{completed_name}' as '{folder_label}'")
                        else:
                            dynamic_status_msg = f"{folder_label} (ignored: paused)"
                            dynamic_status_time = current_time
                            print(f"[Dynamic Gesture] Completed '{folder_label}' but ignored because recording is PAUSED")
                    buffer_predictions.clear()
                    dynamic_start_counter = 0
                elif "CANCELLED" in dyn_msg:
                    # Stopped on a middle frame, performed wrong sequence, or timed out -> Cancel translation
                    print(f"[Dynamic Gesture] {dyn_msg}")
                    pending_dynamic_gesture = None
                    dynamic_timer_start = 0.0
                    dynamic_start_counter = 0

            # B: Route to Dynamic Gesture: ONLY when user performs the FIRST frame (_START) WITH MOTION
            elif is_first_frame and hands_detected and time_since_dynamic > 1.5 and confidence >= 0.45:
                # Two-handed dynamic gestures (How are you, Nice to meet you) MUST have BOTH hands present!
                is_two_handed_dyn = clean_target.lower() in ("how are you", "nice to meet you")
                # Dynamic gestures require physical hand motion (motion speed threshold)
                has_motion = (route_mode == "DYNAMIC" or motion_speed >= 0.035)

                if is_two_handed_dyn and detected_hand != "both":
                    dynamic_start_counter = 0
                # Single-handed dynamic gestures (J, Z) must be signed with the designated signing hand!
                elif not is_two_handed_dyn and detected_hand == "left" and dominant_hand != "left":
                    dynamic_start_counter = 0
                elif not has_motion:
                    # Stationary hand: do NOT hijack as dynamic gesture!
                    dynamic_start_counter = 0
                else:
                    dynamic_start_counter += 1
                    req_start_frames = 1 if cand_lower == "j" else 2
                    if dynamic_start_counter >= req_start_frames:
                        dynamic_manager.start_gesture(clean_target, initial_coords=active_coords)
                        pending_dynamic_gesture = clean_target
                        dynamic_timer_start = dynamic_manager.start_time
                        dynamic_timer_paused = False
                        dynamic_status_msg = f"Tracking dynamic: {clean_target}..."
                        dynamic_status_time = current_time
                        print(f"[Dynamic Gesture] ROUTED TO DYNAMIC sequence for '{clean_target}' from FIRST FRAME (motion={motion_speed:.3f})")
                        buffer_predictions.clear()
                        dynamic_start_counter = 0

            # C: Route to Static Gesture or Control Signs
            else:
                dynamic_start_counter = 0

                # CRITICAL: Dynamic gesture intermediate or end frames must NEVER be translated as static signs!
                if is_dyn_class and (is_dynamic_label or route_mode == "DYNAMIC"):
                    # Intermediate frames or end frames without starting from frame 1 must be ignored
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

                    # ── FAST CONTROL GESTURE TRIGGER (START / STOP) ──
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
                        agreeing = len(buffer_predictions) == N_FRAMES and count >= (N_FRAMES * 0.5)

                        if agreeing:
                            clean_lbl = clean_target or get_clean_folder_label(live_gesture)
                            if clean_lbl:
                                clean_upper = clean_lbl.upper()
                                # Guard 1: Two-handed static signs (e.g. Fine) require BOTH hands to be present
                                if clean_upper in ("FINE", "HOW ARE YOU", "NICE TO MEET YOU") and detected_hand != "both":
                                    agreeing = False
                                # Guard 2: Single-hand alphabet gestures (A-Z) cannot be recorded from Left hand unless dominant_hand == "left"
                                elif detected_hand == "left" and dominant_hand != "left" and clean_upper not in LEFT_HAND_ALLOWED_LABELS:
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
                                            print(f"[Static Sign] Recorded: '{predicted_label}'")
                                        else:
                                            print(f"[Static Sign] Ignored '{predicted_label}' (Recording is PAUSED / STANDBY)")

                                        buffer_predictions.clear()
                    else:
                        agreeing = False
                        if list(buffer_predictions).count("—") == N_FRAMES:
                            current_word = "—"

            agreeing = (dynamic_manager.is_active or agreeing)

            # ── 4. HUD Dynamic Gesture Progress Bar (if active) ──────────────────
            if dynamic_manager.is_active:
                time_left = dynamic_manager.get_remaining_time()
                ratio = max(0.0, min(1.0, time_left / MAX_DYNAMIC_DURATION))
                disp_dyn = clean_display_label(dynamic_manager.gesture_name)
                hud_txt = f"{disp_dyn} ({time_left:.1f}s)"
                cv2.putText(frame, hud_txt, (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 235, 255), 1, cv2.LINE_AA)
                bar_w = int(220 * ratio)
                cv2.rectangle(frame, (18, 60), (18 + bar_w, 64), (0, 220, 130), -1)

            # Environmental Diagnostics Badge at bottom of frame
            diag_items = []
            if lighting_state == "LOW_LIGHT":
                diag_items.append(f"Low-Light ({int(mean_luminance)}L) CLAHE ON")
            elif lighting_state == "HIGH_LIGHT":
                diag_items.append(f"High-Light ({int(mean_luminance)}L) Anti-Glare ON")
            else:
                diag_items.append(f"Optimal Light ({int(mean_luminance)}L)")

            if (coords_r is not None and metrics_r.get("is_far")) or (coords_l is not None and metrics_l.get("is_far")):
                diag_items.append(">1.5m Dist Filter")
            diag_items.append(f"Router: {route_mode} (v={motion_speed:.2f})")

            diag_str = " | ".join(diag_items)
            cv2.putText(frame, diag_str, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 230, 255), 1, cv2.LINE_AA)


            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                with _frame_lock:
                    _latest_frame_bytes = buffer.tobytes()

        except Exception as e:
            print(f"[Camera Worker Exception]: {e}")
            traceback.print_exc()

        time.sleep(0.01)

    if cap:
        cap.release()
        cap = None


def generate_frames():
    start_camera_thread()
    last_sent = None
    try:
        while camera_running:
            with _frame_lock:
                fb = _latest_frame_bytes
            if fb is not None and fb is not last_sent:
                last_sent = fb
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n")
            time.sleep(0.02)
    except (GeneratorExit, ConnectionResetError, BrokenPipeError, OSError):
        pass
    except Exception as e:
        print(f"[Video Feed Stream Error]: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/mobile")
def mobile():
    return render_template("mobile.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/gesture")
def gesture_status():
    global pending_dynamic_gesture, dynamic_timer_start, speak_alert, route_mode, motion_speed
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

    alert_to_send = speak_alert
    speak_alert = None  # Clear once sent

    # Clean display labels without _START or _END suffixes
    clean_live = clean_display_label(live_gesture)
    clean_r = clean_display_label(right_live_gesture) if right_live_gesture != "TRACKED" else "TRACKED"
    clean_l = clean_display_label(left_live_gesture) if left_live_gesture != "TRACKED" else "TRACKED"
    clean_both = clean_display_label(both_live_gesture)
    clean_dyn = clean_display_label(active_dyn) if active_dyn else None

    return jsonify(
        {
            "status": "running" if camera_running else "stopped",
            "camera_index": camera_index,
            "available_cameras": get_available_camera_indices(),
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
            "speak_alert": alert_to_send,
        }
    )


@app.route("/backspace", methods=["POST"])
def backspace():
    global current_sentence, current_word
    if current_sentence:
        current_sentence.pop()
    current_word = current_sentence[-1] if current_sentence else ""
    return jsonify({"ok": True, "word": "".join(current_sentence)})


@app.route("/api/camera/switch", methods=["POST"])
def switch_camera():
    global camera_index, _switch_camera_requested, _target_camera_index
    data = request.get_json(silent=True) or {}

    if "index" in data:
        try:
            target_idx = int(data["index"])
        except (ValueError, TypeError):
            return jsonify({"ok": False, "error": "Invalid camera index"}), 400
    else:
        avail = get_available_camera_indices()
        if not avail:
            avail = [0, 1]
        if camera_index in avail:
            curr_pos = avail.index(camera_index)
            target_idx = avail[(curr_pos + 1) % len(avail)]
        else:
            target_idx = avail[0]

    with _camera_switch_lock:
        _target_camera_index = target_idx
        _switch_camera_requested = True

    return jsonify({
        "ok": True,
        "camera_index": target_idx,
        "available_cameras": get_available_camera_indices(),
        "message": f"Switching to Camera {target_idx}..."
    })


@app.route("/api/camera/list", methods=["GET"])
def list_cameras():
    return jsonify({
        "ok": True,
        "camera_index": camera_index,
        "available_cameras": get_available_camera_indices()
    })


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


@app.route("/clear", methods=["POST"])
def clear():

    global current_sentence, current_word, buffer_predictions
    current_sentence = []
    current_word = ""
    buffer_predictions.clear()
    return jsonify({"status": "cleared"})


@app.route("/shutdown", methods=["POST"])
def shutdown():

    global camera_running
    camera_running = False

    os._exit(0)
    return jsonify({"status": "shutting down"})


def _retrain_model_background():
    global \
        is_retraining, \
        model, \
        label_classes, \
        _prob_buffer_both, \
        _prob_buffer_right, \
        _prob_buffer_left, \
        buffer_predictions
    try:
        import subprocess
        import sys

        subprocess.run([sys.executable, "train_model.py"], check=True)

        model = models.load_model(model_path)
        label_classes = np.load(labels_path, allow_pickle=True)

        _prob_buffer_both.clear()
        _prob_buffer_right.clear()
        _prob_buffer_left.clear()
        buffer_predictions.clear()
        print(
            f"Model successfully retrained and reloaded. Now has {len(label_classes)} classes."
        )
    except Exception as e:
        print(f"Error during retraining: {e}")
    finally:
        is_retraining = False


@app.route("/correct_gesture", methods=["POST"])
def correct_gesture():
    global latest_features, is_retraining

    if is_retraining:
        return jsonify(
            {"ok": False, "message": "Model is currently retraining. Please wait."}
        )

    data = request.get_json(silent=True) or {}
    if "label" not in data:
        return jsonify({"ok": False, "message": "No label provided."})

    correct_label = str(data["label"]).upper().strip()

    if latest_features is None:
        return jsonify({"ok": False, "message": "No gesture detected to correct."})

    import csv

    with open("gesture_data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        row = list(latest_features) + [correct_label]
        for _ in range(20):
            writer.writerow(row)

    is_retraining = True
    threading.Thread(target=_retrain_model_background, daemon=True).start()

    return jsonify(
        {"ok": True, "message": f'Learning "{correct_label}" in background...'}
    )


if __name__ == "__main__":
    scan_available_cameras()
    start_camera_thread()
    print("Starting Flask Server...")
    print("Navigate to http://localhost:5000 in your browser.")

    import webbrowser

    threading.Timer(1.25, lambda: webbrowser.open("http://localhost:5000")).start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
