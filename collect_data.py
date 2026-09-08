import argparse
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from aslo_features import extract_features


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from images.")
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
            f"Directory '{dataset_dir}' not found. Please create it, organize your images into folders by gesture label, and try again."
        )
        return

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    )

    data = []
    labels = []

    print(f"Reading images from '{dataset_dir}'...")

    for label_name in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label_name)
        if not os.path.isdir(label_dir):
            continue

        print(f"Collecting gesture '{label_name}'...")

        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)

            if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            landmarks_flat = np.array(extract_features(results))

            data.append(landmarks_flat)
            labels.append(label_name)

    if not data:
        print("No valid images found or no data extracted.")
        return

    df = pd.DataFrame(data)
    df["label"] = labels

    df.to_csv(output_file, index=False)
    print(f"Successfully processed {len(data)} images.")
    print(f"Extracted landmarks saved to '{output_file}'. Shape: {df.shape}")


if __name__ == "__main__":
    main()
