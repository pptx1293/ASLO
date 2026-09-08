import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from keras import models
import collections

import aslo_features

app = Flask(__name__)

model_path = "gesture_model.keras"
labels_path = "label_classes.npy"

model = None
label_classes = None
camera_running = False
cap = None

current_sentence = []
current_word = ""
N_FRAMES = 15
buffer_predictions = collections.deque(maxlen=N_FRAMES)
is_recording = False

_SMOOTH_WINDOW = 8
_prob_buffer_both = collections.deque(maxlen=_SMOOTH_WINDOW)
_prob_buffer_single = collections.deque(maxlen=_SMOOTH_WINDOW)
TWO_HANDED_LABELS = {"START", "STOP", "SPACE", "BACK SPACE", "BACKSPACE", "NEUTRAL"}

live_gesture = "—"
live_conf = 0.0
agreeing = False
latest_features = None
is_retraining = False

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
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils


def generate_frames():
    global \
        cap, \
        camera_running, \
        current_sentence, \
        current_word, \
        buffer_predictions, \
        is_recording
    global live_gesture, live_conf, agreeing, latest_features

    cap = cv2.VideoCapture(0)
    camera_running = True

    last_trigger_time = 0
    REPEAT_DELAY = 1.3

    while camera_running:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks and model is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            try:
                features_both = aslo_features.extract_features(results)
                features_single = aslo_features.extract_primary_hand_features(results)

                latest_features = features_both

                inputs = np.array([features_both, features_single])
                probs = np.array(model(inputs, training=False))
                probs_both = probs[0]
                probs_single = probs[1]

                if (
                    _prob_buffer_both
                    and _prob_buffer_both[-1].shape != probs_both.shape
                ):
                    _prob_buffer_both.clear()
                    _prob_buffer_single.clear()

                _prob_buffer_both.append(probs_both)
                _prob_buffer_single.append(probs_single)

                avg_probs_both = np.mean(_prob_buffer_both, axis=0)
                avg_probs_single = np.mean(_prob_buffer_single, axis=0)

                idx_both = np.argmax(avg_probs_both)
                idx_single = np.argmax(avg_probs_single)

                label_both = str(label_classes[idx_both])
                label_single = str(label_classes[idx_single])

                conf_both = float(avg_probs_both[idx_both])
                conf_single = float(avg_probs_single[idx_single])

                if label_both.upper() in TWO_HANDED_LABELS and (
                    conf_both > 0.75 or conf_both > conf_single
                ):
                    raw_label = label_both
                    confidence = conf_both
                else:
                    raw_label = label_single
                    confidence = conf_single

                live_gesture = aslo_features.apply_heuristics(results, raw_label)
                live_conf = confidence

                left_present = False
                right_present = False
                for handed in results.multi_handedness:
                    if handed.classification[0].label == "Left":
                        right_present = True
                    else:
                        left_present = True

                if left_present and not right_present:
                    threshold = 0.85
                else:
                    threshold = 0.25
            except Exception as e:
                print(f"Ignored frame error (likely model swapping): {e}")
                _prob_buffer_both.clear()
                _prob_buffer_single.clear()
                live_gesture = "—"
                live_conf = 0.0
                confidence = 0.0
                threshold = 0.85
        else:
            _prob_buffer_both.clear()
            _prob_buffer_single.clear()
            live_gesture = "—"
            live_conf = 0.0
            confidence = 0.0
            threshold = 0.85

        buffer_predictions.append(live_gesture)

        if confidence > threshold:
            count = list(buffer_predictions).count(live_gesture)

            agreeing = len(buffer_predictions) == N_FRAMES and count >= (N_FRAMES * 0.6)

            if agreeing:
                predicted_label = live_gesture
                current_time = time.time()

                should_trigger = (predicted_label != current_word) or (
                    predicted_label != "—"
                    and (current_time - last_trigger_time) > REPEAT_DELAY
                )

                if should_trigger:
                    current_word = predicted_label
                    last_trigger_time = current_time

                    if current_word.upper() != "NEUTRAL":
                        cw_lower = current_word.lower()
                        if cw_lower == "start":
                            is_recording = True
                        elif cw_lower == "stop":
                            is_recording = False
                        elif cw_lower == "backspace" or cw_lower == "back space":
                            if len(current_sentence) > 0:
                                current_sentence.pop()
                        elif cw_lower == "space":
                            if is_recording:
                                current_sentence.append(" ")
                        else:
                            if is_recording:
                                current_sentence.append(current_word)

                    buffer_predictions.clear()
        else:
            agreeing = False

            if list(buffer_predictions).count("—") == N_FRAMES:
                current_word = "—"

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    if cap:
        cap.release()


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/gesture")
def gesture_status():

    return jsonify(
        {
            "status": "running" if camera_running else "stopped",
            "live_gesture": live_gesture,
            "live_conf": live_conf,
            "buffer_fill": len(buffer_predictions),
            "buffer_max": N_FRAMES,
            "agreeing": agreeing,
            "active": is_recording,
            "word": "".join(current_sentence),
        }
    )


@app.route("/toggle_active", methods=["POST"])
def toggle_active():
    global is_recording
    data = request.get_json()
    if data and "active" in data:
        is_recording = data["active"]
    return jsonify({"ok": True})


@app.route("/append_text", methods=["POST"])
def append_text():
    global current_sentence
    data = request.get_json()
    if data and "text" in data:
        current_sentence.append(" " + data["text"] + " ")
    return jsonify({"ok": True})


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
        _prob_buffer_single, \
        buffer_predictions
    try:
        import subprocess
        import sys

        subprocess.run([sys.executable, "train_model.py"], check=True)

        model = models.load_model(model_path)
        label_classes = np.load(labels_path, allow_pickle=True)

        _prob_buffer_both.clear()
        _prob_buffer_single.clear()
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

    data = request.get_json()
    if not data or "label" not in data:
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
    print("Starting Flask Server...")
    print("Navigate to http://localhost:5000 in your browser.")

    import webbrowser

    threading.Timer(1.25, lambda: webbrowser.open("http://localhost:5000")).start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
