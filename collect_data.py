import argparse
import os
import re
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from aslo_features import extract_features
from aslo_pipeline import LowLightEnhancer

low_light_enhancer = LowLightEnhancer()


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally (e.g., 1.jpg, 2.jpg, 10.jpg)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def process_frame(frame, hands):
    """Process a single frame (BGR) with horizontal flip, adaptive lighting (anti-glare & low-light), and extract landmarks."""
    flipped = cv2.flip(frame, 1)
    enhanced, _, _ = low_light_enhancer.process(flipped)
    image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not (results and results.multi_hand_landmarks):
        raw_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        results = hands.process(raw_rgb)
    if results and results.multi_hand_landmarks:
        feats = extract_features(results)
        return feats
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from images and dynamic videos.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="picture_file",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--output", type=str, default="gesture_data.csv", help="Output CSV file name."
    )
    args = parser.parse_args()

    dataset_dir = args.dataset
    output_file = args.output

    if not os.path.exists(dataset_dir):
        print(
            f"Directory '{dataset_dir}' not found. Please create it, organize your images/videos by gesture label, and try again."
        )
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )

    data = []
    labels = []

    print(f"Reading images and videos from '{dataset_dir}'...")

    for label_name in sorted(os.listdir(dataset_dir)):
        label_dir = os.path.join(dataset_dir, label_name)
        if not os.path.isdir(label_dir):
            continue

        print(f"\nProcessing gesture folder '{label_name}'...")

        all_entries = os.listdir(label_dir)
        video_files = [f for f in all_entries if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        seq_dirs = [d for d in all_entries if os.path.isdir(os.path.join(label_dir, d))]
        image_files = [f for f in all_entries if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        is_dynamic = len(video_files) > 0 or len(seq_dirs) > 0

        # Case 1: Folder has video files (e.g., How are you, Nice to meet you, J)
        if video_files:
            print(f"  -> Found {len(video_files)} video(s) for dynamic gesture '{label_name}'.")
            start_count = 0
            end_count = 0

            for v_name in sorted(video_files):
                v_path = os.path.join(label_dir, v_name)
                cap = cv2.VideoCapture(v_path)
                valid_entries = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    flipped = cv2.flip(frame, 1)
                    enhanced, _, _ = low_light_enhancer.process(flipped)
                    image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                    if results.multi_hand_landmarks:
                        feats = extract_features(results)
                        avg_wy = np.mean([lm.landmark[0].y for lm in results.multi_hand_landmarks])
                        valid_entries.append((feats, avg_wy))

                cap.release()

                if len(valid_entries) >= 6:
                    # Filter for active frames where hands are raised into signing space (wrist_y < 0.75)
                    raised = [e[0] for e in valid_entries if e[1] < 0.75]
                    if len(raised) >= 6:
                        target_feats = raised
                    else:
                        # Fallback: slice out resting setup (< 20%) and resting teardown (> 80%)
                        n_tot = len(valid_entries)
                        target_feats = [e[0] for e in valid_entries[int(n_tot * 0.20) : int(n_tot * 0.80)]]

                    n_active = len(target_feats)
                    if n_active >= 4:
                        # First half of active gesture is START, second half is END
                        half = n_active // 2
                        start_feats = target_feats[:half]
                        end_feats = target_feats[half:]

                        # Take up to 20 samples from each phase
                        step_s = max(1, len(start_feats) // 20)
                        step_e = max(1, len(end_feats) // 20)

                        for f in start_feats[::step_s][:20]:
                            data.append(f)
                            labels.append(f"{label_name}_START")
                            start_count += 1

                        for f in end_feats[::step_e][:20]:
                            data.append(f)
                            labels.append(f"{label_name}_END")
                            end_count += 1

            print(f"  Extracted {start_count} START frames and {end_count} END frames from videos.")

        # Case 2: Folder has sequence subdirectories (e.g., Z/SEQ1, Z/SEQ2, Z/SEQ3)
        elif seq_dirs:
            print(f"  -> Found {len(seq_dirs)} sequence folder(s) for dynamic gesture '{label_name}'.")
            start_count = 0
            end_count = 0

            for s_name in sorted(seq_dirs, key=natural_sort_key):
                s_path = os.path.join(label_dir, s_name)
                seq_images = [f for f in os.listdir(s_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                seq_images.sort(key=natural_sort_key)

                valid_feats = []
                for img_name in seq_images:
                    img_path = os.path.join(s_path, img_name)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    feats = process_frame(image, hands)
                    if feats is not None:
                        valid_feats.append(feats)

                if len(valid_feats) >= 4:
                    n_take = max(2, min(15, int(len(valid_feats) * 0.25)))
                    start_feats = valid_feats[:n_take]
                    end_feats = valid_feats[-n_take:]

                    for f in start_feats:
                        data.append(f)
                        labels.append(f"{label_name}_START")
                        start_count += 1

                    for f in end_feats:
                        data.append(f)
                        labels.append(f"{label_name}_END")
                        end_count += 1

            print(f"  Extracted {start_count} START frames and {end_count} END frames from sequences.")

        # Case 3: Static gesture folder (regular image files)
        else:
            img_count = 0
            for img_name in sorted(image_files, key=natural_sort_key):
                img_path = os.path.join(label_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                feats = process_frame(image, hands)
                if feats is not None:
                    data.append(feats)
                    labels.append(label_name)
                    img_count += 1

            print(f"  Extracted {img_count} valid static frames.")

    if not data:
        print("No valid images or videos found / no hands detected.")
        return

    df = pd.DataFrame(data)
    df["label"] = labels

    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully processed dataset. Total samples: {len(data)}")
    print(f"Extracted landmarks saved to '{output_file}'. Shape: {df.shape}")
    print("\nClass distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
