"""
Standalone Trainer 2: Pure Static Words MLP (`train_words_model.py`)
====================================================================
Trains ONLY Model Group 2 (`models/words_static_mlp.keras`) purely on the 12 Static Words
and Control Gestures (`Fine`, `I love you`, `Love`, `ME`, `Thank you`, `YOU`, `backspace`,
`hello`, `neutral`, `space`, `start`, `stop`) with ZERO ASL letters or dynamic `_START`/`_END`
classes mixed in.
All models are 100% pure Keras MLPs (`Dense`, `BatchNormalization`, `Dropout`).
"""

import os
from typing import Dict, Tuple

import numpy as np

from aslo_features import EXPECTED_WORD_DIM, extract_static_word_features
from train_utils import (
    DATASET_DIR,
    MODELS_DIR,
    SINGLE_HAND_VIEW_CONTROL_SET,
    STRICT_ONE_HAND_WORDS,
    STRICT_TWO_HAND_WORDS,
    _rotate_pts_3d,
    auto_categorize_dataset,
    load_dataset_df,
    print_detailed_report,
    train_and_evaluate_mlp_with_scaler,
)


def _synthesize_live_thumbs_up_fists(rng, n_samples: int = 90) -> list:
    """
    Generates realistic non-foreshortened live-webcam two-hand thumbs-up (👍 👍) landmarks
    for the `start` control gesture, completely independent of ASL training data.
    """
    # Canonical upright closed-fist palm (wrist at origin, MCPs along y ~ -0.95, curled fingertips)
    base_fist = np.array(
        [
            [0.00, 0.00, 0.00],    # 0 wrist
            [-0.28, -0.25, -0.05], # 1 thumb cmc
            [-0.34, -0.55, -0.08], # 2 thumb mcp
            [-0.36, -1.05, -0.10], # 3 thumb ip
            [-0.34, -1.68, -0.14], # 4 thumb tip (pointing UP)
            [-0.25, -0.92, -0.02], # 5 index mcp
            [-0.22, -1.12, -0.15], # 6 index pip
            [-0.18, -0.82, -0.18], # 7 index dip
            [-0.16, -0.62, -0.12], # 8 index tip (curled)
            [-0.02, -0.98, 0.00],  # 9 middle mcp
            [-0.01, -1.16, -0.14], # 10 middle pip
            [0.00, -0.84, -0.17],  # 11 middle dip
            [0.01, -0.64, -0.11],  # 12 middle tip (curled)
            [0.18, -0.92, 0.03],   # 13 ring mcp
            [0.18, -1.08, -0.12],  # 14 ring pip
            [0.16, -0.78, -0.15],  # 15 ring dip
            [0.15, -0.60, -0.10],  # 16 ring tip (curled)
            [0.36, -0.82, 0.06],   # 17 pinky mcp
            [0.35, -0.96, -0.08],  # 18 pinky pip
            [0.32, -0.72, -0.11],  # 19 pinky dip
            [0.30, -0.56, -0.07],  # 20 pinky tip (curled)
        ],
        dtype=np.float32,
    )
    samples = []
    for _ in range(n_samples):
        r_tu = base_fist.copy()
        r_tu[4, 0] = float(rng.uniform(-0.46, -0.18))
        r_tu[4, 1] = float(rng.uniform(-1.95, -1.42))
        r_tu[4, 2] = float(rng.uniform(-0.25, -0.04))
        noise = rng.normal(0.0, 0.035, size=r_tu.shape).astype(np.float32)
        noise[0] = 0.0
        r_tu = _rotate_pts_3d(
            r_tu + noise,
            float(rng.uniform(-12.0, 12.0)),
            float(rng.uniform(-12.0, 12.0)),
            float(rng.uniform(-14.0, 14.0)),
        )
        r_tu[0] = 0.0
        l_tu = r_tu.copy()
        l_tu[:, 0] = -l_tu[:, 0]
        l_noise = rng.normal(0.0, 0.025, size=l_tu.shape).astype(np.float32)
        l_noise[0] = 0.0
        l_tu = l_tu + l_noise
        l_tu[0] = 0.0
        samples.append((r_tu, l_tu))
    return samples


def build_augmented_words_dataset(
    X_words_raw: np.ndarray,
    y_words: np.ndarray,
    num_augs: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the 252-dim Static Words feature matrix purely from Static Words & Control Gestures.
    - Shifts single-hand slot-1 rows (`only_slot1 -> slot0`) before any masking so no all-zero vectors are created.
    - Strips contaminated 2nd-hand landmarks from strictly 1-hand words (`YOU`, `Fine`, `I love you`, `Thank you`, `hello`).
    - Keeps strictly 2-hand words (`ME`) strictly 2-handed.
    - Adds swapped slot order `(l, r)` for 2-hand gestures and single-hand views for control words.
    """
    rng = np.random.default_rng(42)
    feats_list = []
    labels_list = []

    for row, lbl in zip(X_words_raw, y_words):
        r_base = np.asarray(row[:63], dtype=np.float32).reshape(21, 3)
        l_base = np.asarray(row[63:126], dtype=np.float32).reshape(21, 3)
        lbl_lower = str(lbl).strip().lower()

        has_r = not np.all(np.abs(r_base) < 1e-7)
        has_l = not np.all(np.abs(l_base) < 1e-7)
        if not has_r and not has_l:
            continue

        # Shift single-hand slot-1 rows into slot-0 first!
        if (not has_r) and has_l:
            r_base = l_base.copy()
            l_base = np.zeros((21, 3), dtype=np.float32)
            has_r = True
            has_l = False

        if lbl_lower in STRICT_ONE_HAND_WORDS:
            l_base = np.zeros((21, 3), dtype=np.float32)
            has_l = False

        if lbl_lower in STRICT_TWO_HAND_WORDS and not has_l:
            l_base = r_base.copy()
            l_base[:, 0] = -l_base[:, 0]
            has_l = True

        htype = "both" if (has_r and has_l) else "right"

        # 1. Original slot order
        feats_list.append(extract_static_word_features((r_base, l_base), hand_type=htype))
        labels_list.append(lbl)

        # 2. Swapped slot order for 2-hand gestures
        if has_r and has_l:
            feats_list.append(extract_static_word_features((l_base, r_base), hand_type="both"))
            labels_list.append(lbl)

        augs_for_word = (
            (num_augs + 3)
            if lbl_lower in ("start", "stop", "me", "you")
            else (num_augs + (2 if not has_l else 0))
        )
        for _ in range(augs_for_word):
            r_pts = r_base.copy()
            l_pts = l_base.copy()
            if lbl_lower in ("start", "stop", "me"):
                palm_foreshorten = float(rng.uniform(0.48, 1.85))
                if has_r:
                    r_pts[5:18, 1] *= palm_foreshorten
                if has_l:
                    l_pts[5:18, 1] *= palm_foreshorten
            if has_r:
                r_pts[:, 2] *= float(rng.uniform(0.30, 1.65))
                r_noise = rng.normal(0.0, 0.026, size=r_pts.shape).astype(np.float32)
                r_noise[0] = 0.0
                r_pts = _rotate_pts_3d(
                    r_pts + r_noise,
                    float(rng.uniform(-12.0, 12.0)),
                    float(rng.uniform(-12.0, 12.0)),
                    float(rng.uniform(-14.0, 14.0)),
                )
                r_pts[0] = 0.0
            if has_l:
                l_pts[:, 2] *= float(rng.uniform(0.30, 1.65))
                l_noise = rng.normal(0.0, 0.026, size=l_pts.shape).astype(np.float32)
                l_noise[0] = 0.0
                l_pts = _rotate_pts_3d(
                    l_pts + l_noise,
                    float(rng.uniform(-12.0, 12.0)),
                    float(rng.uniform(-12.0, 12.0)),
                    float(rng.uniform(-14.0, 14.0)),
                )
                l_pts[0] = 0.0
            feats_list.append(extract_static_word_features((r_pts, l_pts), hand_type=htype))
            labels_list.append(lbl)
            if has_r and has_l:
                feats_list.append(extract_static_word_features((l_pts, r_pts), hand_type="both"))
                labels_list.append(lbl)

    # 4. Synthesize natural live-webcam two-hand thumbs-up ('start') poses
    start_mask = np.array([str(y).strip().lower() == "start" for y in y_words])
    if np.any(start_mask):
        start_label_str = str(y_words[np.where(start_mask)[0][0]])
        for r_tu, l_tu in _synthesize_live_thumbs_up_fists(rng, n_samples=90):
            feats_list.append(extract_static_word_features((r_tu, l_tu), hand_type="both"))
            labels_list.append(start_label_str)

    return np.asarray(feats_list, dtype=np.float32), np.asarray(labels_list)


def main() -> Dict:
    print("\n" + "=" * 72)
    print("STANDALONE TRAINING 2/3: PURE STATIC WORDS MODEL GROUP")
    print("=" * 72)
    df = load_dataset_df()
    X_raw_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values

    registry = auto_categorize_dataset(DATASET_DIR, df)
    label_to_type = registry["label_to_type"]

    # Pure Static Words ONLY — zero ASL letters, zero dynamic _START/_END poses!
    words_mask = np.array([label_to_type.get(str(lbl), "") == "static_words" for lbl in y_all])
    X_words_raw = X_raw_all[words_mask]
    y_words_base = np.array([str(lbl).strip() for lbl in y_all[words_mask]])

    print(
        f"\n[Words Feature Engine] Computing {EXPECTED_WORD_DIM}-dim dual-hand features "
        f"(pure Static Words only, {len(X_words_raw)} base samples)..."
    )
    X_words, y_words = build_augmented_words_dataset(X_words_raw, y_words_base, num_augs=3)
    assert X_words.shape[1] == EXPECTED_WORD_DIM

    _, _, _, stats_words = train_and_evaluate_mlp_with_scaler(
        X=X_words,
        y=y_words,
        model_prefix="words_static",
        model_title="Model Group 2: Pure Static Words MLP",
        epochs=95,
    )

    for src_ext, dst_name in [("_labels.npy", "words_labels.npy"), ("_scaler.npz", "words_scaler.npz")]:
        src_p = os.path.join(MODELS_DIR, f"words_static{src_ext}")
        dst_p = os.path.join(MODELS_DIR, dst_name)
        if os.path.exists(src_p):
            with open(src_p, "rb") as rf, open(dst_p, "wb") as wf:
                wf.write(rf.read())

    print_detailed_report("MODEL GROUP 2: PURE STATIC WORDS MLP REPORT", stats_words)
    print("\n[DONE] Static Words Model Group trained independently and saved to 'models/'.")
    return {"words": stats_words}


if __name__ == "__main__":
    main()
