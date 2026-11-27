"""
Shared Utilities for ASLO Separate-Training & Merged Multi-MLP Pipeline (`train_utils.py`)
==========================================================================================
Provides common dataset loading, auto-categorization (`models/gesture_registry.json`),
3D landmark rotation, and 100% pure Keras MLP construction/training with `StandardScaler`.
"""

import json
import os
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from keras import callbacks, layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from aslo_features import (
    ASL_SHAPE_FAMILIES,
    EXPECTED_ASL_DIM,
    EXPECTED_DYNAMIC_DIM,
    EXPECTED_SHAPE_DIM,
    EXPECTED_WORD_DIM,
    SEQ_LEN,
)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MODELS_DIR = "models"
DATASET_DIR = "picture_file"

DYN_KEYFRAME_LABELS = {
    "how are you_start",
    "how are you_end",
    "nice to meet you_start",
    "nice to meet you_end",
    "hello_start",
    "hello_end",
    "j_start",
    "j_end",
    "z_start",
    "z_end",
}

STRICT_ONE_HAND_WORDS = {"fine", "i love you", "thank you", "you"}
SINGLE_HAND_VIEW_CONTROL_SET = set()  # Control words are strictly 2-handed so they NEVER override 1-hand ASL letters!
STRICT_TWO_HAND_WORDS = {"me", "love", "start", "stop", "space", "backspace", "back space", "neutral"}

STRICT_ONE_HAND_DYN_KEYFRAMES = {"hello_start", "hello_end", "j_start", "j_end", "z_start", "z_end"}
STRICT_TWO_HAND_DYN_KEYFRAMES = {
    "how are you_start",
    "how are you_end",
    "nice to meet you_start",
    "nice to meet you_end",
}

EXPLICIT_DYNAMIC_GESTURES = {"hello", "j", "z", "how are you", "nice to meet you"}


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def load_dataset_df(data_path: str = "gesture_data.csv") -> pd.DataFrame:
    if not os.path.exists(data_path):
        if os.path.exists("gesture_data_original_backup.csv"):
            print("Loading dataset from 'gesture_data_original_backup.csv'...")
            return pd.read_csv("gesture_data_original_backup.csv")
        raise FileNotFoundError(f"'{data_path}' not found. Please run collect_data.py first.")
    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def _rotate_pts_3d(pts: np.ndarray, rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    return (pts @ R.T).astype(np.float32)


def build_keras_mlp(input_dim: int, num_classes: int, model_name: str) -> models.Model:
    """
    Constructs a 100% pure Keras MLP architecture:
    Input -> GaussianNoise(0.05) -> Dense(256) -> BN -> Dropout -> Dense(128) -> BN -> Dropout -> Dense(64) -> BN -> Softmax
    """
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)
    inp = layers.Input(shape=(input_dim,))
    x = layers.GaussianNoise(0.05)(inp)

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.0004))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.0004))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.0004))(x)
    x = layers.BatchNormalization()(x)

    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out, name=safe_name)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def auto_categorize_dataset(dataset_dir: str, df: pd.DataFrame) -> Dict:
    """
    Scans `picture_file` and `gesture_data.csv` to build `models/gesture_registry.json`,
    including active anatomical shape families for Stage-1 / Stage-2 cooperative routing.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    dynamic_folders = set()
    static_asl_folders = set()
    static_words_folders = set()

    if os.path.exists(dataset_dir):
        for folder_name in sorted(os.listdir(dataset_dir)):
            folder_path = os.path.join(dataset_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            entries = os.listdir(folder_path)
            has_video = any(f.lower().endswith(VIDEO_EXTENSIONS) for f in entries)
            has_seq_dir = any(
                os.path.isdir(os.path.join(folder_path, d)) and d.upper().startswith("SEQ")
                for d in entries
            )

            clean_name = folder_name.strip()
            if has_video or has_seq_dir or clean_name.lower() in EXPLICIT_DYNAMIC_GESTURES:
                dynamic_folders.add(clean_name)
            elif len(clean_name) == 1 and clean_name.isalnum():
                static_asl_folders.add(clean_name.upper())
            else:
                static_words_folders.add(clean_name)

    csv_labels = sorted(df["label"].unique()) if (df is not None and not df.empty) else []
    label_to_type = {}
    dynamic_base_upper = {d.upper() for d in dynamic_folders} | {d.upper() for d in EXPLICIT_DYNAMIC_GESTURES}
    dynamic_keyframes = set()

    for lbl in csv_labels:
        lbl_str = str(lbl).strip()
        base_lbl = re.sub(r"_(START|END)$", "", lbl_str, flags=re.IGNORECASE).strip()
        if (
            lbl_str.upper().endswith("_START")
            or lbl_str.upper().endswith("_END")
            or base_lbl.upper() in dynamic_base_upper
        ):
            label_to_type[lbl_str] = "dynamic_gestures"
            dynamic_folders.add(base_lbl)
            if lbl_str.upper().endswith("_START") or lbl_str.upper().endswith("_END"):
                dynamic_keyframes.add(lbl_str)
        elif len(lbl_str) == 1 and lbl_str.isalnum():
            label_to_type[lbl_str] = "static_asl"
            static_asl_folders.add(lbl_str.upper())
        else:
            label_to_type[lbl_str] = "static_words"
            static_words_folders.add(lbl_str)

    active_families = {}
    sign_to_family = {}
    for cluster_name, members in ASL_SHAPE_FAMILIES.items():
        present_members = sorted([m for m in members if m in static_asl_folders])
        if len(present_members) >= 2:
            active_families[cluster_name] = present_members
            for m in present_members:
                sign_to_family[m] = cluster_name

    registry = {
        "static_asl": sorted(list(static_asl_folders)),
        "shape_families": active_families,
        "sign_to_family": sign_to_family,
        "static_words": sorted(list(static_words_folders)),
        "dynamic_gestures": sorted(list(dynamic_folders)),
        "dynamic_keyframes": sorted(list(dynamic_keyframes)),
        "label_to_type": label_to_type,
        "expected_shape_dim": EXPECTED_SHAPE_DIM,
        "expected_asl_dim": EXPECTED_ASL_DIM,
        "expected_word_dim": EXPECTED_WORD_DIM,
        "expected_dynamic_dim": EXPECTED_DYNAMIC_DIM,
        "seq_len": SEQ_LEN,
    }

    registry_path = os.path.join(MODELS_DIR, "gesture_registry.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"\n[Registry] Saved gesture registry to '{registry_path}':")
    print(f"  - Static ASL ({len(registry['static_asl'])}): {registry['static_asl']}")
    for c_name, c_members in active_families.items():
        print(f"    * {c_name} ({len(c_members)}): {c_members}")
    print(f"  - Static Words ({len(registry['static_words'])}): {registry['static_words']}")
    print(f"  - Dynamic Gestures ({len(registry['dynamic_gestures'])}): {registry['dynamic_gestures']}")
    print(f"  - Dynamic Keyframes ({len(registry['dynamic_keyframes'])}): {registry['dynamic_keyframes']}")
    return registry


def train_and_evaluate_mlp_with_scaler(
    X: np.ndarray,
    y: np.ndarray,
    model_prefix: str,
    model_title: str,
    epochs: int = 90,
    batch_size: int = 64,
) -> Tuple[models.Model, LabelEncoder, StandardScaler, Dict]:
    """
    Fits a `StandardScaler` on `X_train`, trains a Keras MLP with `compute_class_weight('balanced', ...)`,
    `EarlyStopping`, and `ReduceLROnPlateau`, and saves:
      - `models/<model_prefix>_mlp.keras`
      - `models/<model_prefix>_labels.npy`
      - `models/<model_prefix>_scaler.npz` (`mean` and `scale`)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    # Variance-floor robust scaling: floor scale at 0.15 so low-variance / Z-damped coordinates
    # are NEVER amplified 20x-65x by live webcam MediaPipe noise, and clip outliers to [-5, +5].
    safe_mean = scaler.mean_.astype(np.float32)
    safe_scale = np.where(scaler.scale_ < 1e-7, 1.0, np.maximum(scaler.scale_, 0.15)).astype(np.float32)
    scaler.scale_ = safe_scale

    X_train_scaled = np.clip((X_train - safe_mean) / safe_scale, -5.0, 5.0).astype(np.float32)
    X_test_scaled = np.clip((X_test - safe_mean) / safe_scale, -5.0, 5.0).astype(np.float32)

    scaler_path = os.path.join(MODELS_DIR, f"{model_prefix}_scaler.npz")
    np.savez(
        scaler_path,
        mean=safe_mean,
        scale=safe_scale,
    )

    labels_path = os.path.join(MODELS_DIR, f"{model_prefix}_labels.npy")
    np.save(labels_path, le.classes_)

    unique_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}

    model = build_keras_mlp(
        input_dim=X.shape[1],
        num_classes=num_classes,
        model_name=model_prefix,
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=22, restore_best_weights=True, verbose=0
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=0
    )

    print(
        f"\n--- Training {model_title} ({num_classes} classes, "
        f"{X_train.shape[0]} train / {X_test.shape[0]} val, dim={X.shape[1]}) ---"
    )
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_scaled, y_test),
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )

    train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    model_path = os.path.join(MODELS_DIR, f"{model_prefix}_mlp.keras")
    model.save(model_path)

    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    target_names = [str(c) for c in le.classes_]
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    hist_clean = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }
    val_losses = hist_clean.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else len(hist_clean.get("loss", []))

    metrics_payload = {
        "model_prefix": model_prefix,
        "model_title": model_title,
        "input_dim": int(X.shape[1]),
        "num_classes": int(num_classes),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_test.shape[0]),
        "epochs_ran": int(len(hist_clean.get("loss", []))),
        "best_epoch": int(best_epoch),
        "train_acc": float(train_acc),
        "train_loss": float(train_loss),
        "val_acc": float(val_acc),
        "val_loss": float(val_loss),
        "classes": target_names,
        "report": report,
        "cm": cm.tolist(),
        "history": hist_clean,
    }
    metrics_path = os.path.join(MODELS_DIR, f"{model_prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics_payload, mf, indent=2)

    print(f"[{model_title}] Saved '{model_path}', '{labels_path}', '{scaler_path}', & '{metrics_path}'")
    print(
        f"[{model_title}] Train Acc: {train_acc * 100:.2f}% | "
        f"Val Acc: {val_acc * 100:.2f}% (Val Loss: {val_loss:.4f}, Best Epoch: {best_epoch}/{metrics_payload['epochs_ran']})"
    )

    return model, le, scaler, {
        "train_acc": float(train_acc),
        "train_loss": float(train_loss),
        "val_acc": float(val_acc),
        "val_loss": float(val_loss),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(metrics_payload["epochs_ran"]),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_test.shape[0]),
        "report": report,
        "cm": cm,
        "classes": target_names,
        "history": hist_clean,
    }


def print_detailed_report(title: str, stats: Dict):
    print("\n" + "=" * 68)
    print(f"{title} (Val Acc: {stats['val_acc']*100:.2f}%)")
    print("=" * 68)
    print(f"{'Class':<24} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support':<9}")
    print("-" * 68)
    for cls in stats["classes"]:
        if cls in stats["report"]:
            r = stats["report"][cls]
            print(
                f"{cls:<24} {r['precision']:<11.3f} {r['recall']:<11.3f} "
                f"{r['f1-score']:<11.3f} {int(r['support']):<9}"
            )
    print("-" * 68)
    print("Confusion Matrix (Rows=True, Cols=Predicted):", stats["classes"])
    print(stats["cm"])
    print("=" * 68)
