"""
Standalone Trainer 3: Pure Dynamic Gestures Model Group (`train_dynamic_model.py`)
==================================================================================
Trains ONLY Model Group 3 (Dynamic Gestures: `How are you`, `J`, `Nice to meet you`, `Z`):
  - Model 3A: 25-Frame Trajectory MLP (`models/dynamic_mlp.keras`, 393 dims)
    including `STATIC_HOLD` background sequences so static poses never false-trigger dynamics.
  - Model 3B: Dynamic Keyframe `_START` / `_END` MLP (`models/dynamic_keyframe_mlp.keras`, 252 dims)
    trained purely on the 8 `_START`/`_END` keyframe poses so the multi-stage
    `_START -> Countdown Timer -> _END` algorithm works cleanly without contaminating Static Words.
All models are 100% pure Keras MLPs (`Dense`, `BatchNormalization`, `Dropout`).
"""

import os
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from aslo_features import (
    EXPECTED_DYNAMIC_DIM,
    EXPECTED_WORD_DIM,
    SEQ_LEN,
    extract_dynamic_mlp_features,
    extract_static_word_features,
    normalize_sequence_length,
)
from aslo_pipeline import LowLightEnhancer
from train_utils import (
    DATASET_DIR,
    DYN_KEYFRAME_LABELS,
    IMAGE_EXTENSIONS,
    MODELS_DIR,
    STRICT_ONE_HAND_DYN_KEYFRAMES,
    STRICT_TWO_HAND_DYN_KEYFRAMES,
    VIDEO_EXTENSIONS,
    _rotate_pts_3d,
    auto_categorize_dataset,
    load_dataset_df,
    natural_sort_key,
    print_detailed_report,
    train_and_evaluate_mlp_with_scaler,
)


def _extract_dominant_hand_21x3(results) -> np.ndarray:
    if not (results and results.multi_hand_landmarks):
        return None
    if len(results.multi_hand_landmarks) == 1:
        lms = results.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

    best_lms = None
    best_score = 999.0
    for lms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
        mp_label = handed.classification[0].label
        wy = lms.landmark[0].y
        score = wy + (0.0 if mp_label == "Right" else 0.25) + (1.0 if wy >= 0.82 else 0.0)
        if score < best_score:
            best_score = score
            best_lms = lms.landmark
    if best_lms is None:
        best_lms = results.multi_hand_landmarks[0].landmark
    return np.array([[lm.x, lm.y, lm.z] for lm in best_lms], dtype=np.float32)


def _augment_sequence_windows(raw_seq: List[np.ndarray], num_augs: int = 12, rng=None) -> List[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(raw_seq)
    if n < 3:
        return []

    base_25 = normalize_sequence_length(raw_seq, target_len=SEQ_LEN)
    sequences = [base_25]

    for _ in range(num_augs):
        if n >= 8:
            start_idx = int(rng.integers(0, max(1, int(n * 0.20))))
            end_idx = int(rng.integers(min(n, max(start_idx + 4, int(n * 0.80))), n + 1))
            sub = raw_seq[start_idx:end_idx]
        else:
            sub = raw_seq

        seq_25 = normalize_sequence_length(sub, target_len=SEQ_LEN)
        scale_factor = float(rng.uniform(0.92, 1.08))
        shift = rng.normal(0.0, 0.008, size=(1, 1, 3)).astype(np.float32)
        jitter = rng.normal(0.0, 0.0025, size=seq_25.shape).astype(np.float32)

        center = seq_25[0:1, 0:1, :]
        aug_seq = (seq_25 - center) * scale_factor + center + shift + jitter
        sequences.append(aug_seq.astype(np.float32))

    return sequences


def _build_static_hold_dynamic_samples(df_csv: pd.DataFrame, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates negative/background 25-frame sequences labeled `'STATIC_HOLD'` across 3 modes:
    Mode 0 (stationary hold), Mode 1 (hand-raising entry + hold), Mode 2 (pose-to-pose transition + hold).
    """
    if df_csv is None or df_csv.empty:
        return np.zeros((0, EXPECTED_DYNAMIC_DIM), dtype=np.float32), np.array([])

    X_hold = []
    y_hold = []
    all_labels = sorted(df_csv["label"].unique())
    all_static_poses = []
    for lbl in all_labels:
        lbl_str = str(lbl).strip()
        if (
            lbl_str.upper().endswith("_START")
            or lbl_str.upper().endswith("_END")
            or lbl_str.upper() in ("J", "Z", "HELLO", "HOW ARE YOU", "NICE TO MEET YOU")
        ):
            continue
        r_vals = df_csv[df_csv["label"] == lbl].iloc[:4, 0:63].values.astype(np.float32)
        r_vals = r_vals[np.any(r_vals != 0, axis=1)]
        for r63 in r_vals:
            all_static_poses.append(r63.reshape(21, 3))

    for lbl in all_labels:
        lbl_str = str(lbl).strip()
        if (
            lbl_str.upper().endswith("_START")
            or lbl_str.upper().endswith("_END")
            or lbl_str.upper() in ("J", "Z", "HELLO", "HOW ARE YOU", "NICE TO MEET YOU")
        ):
            continue
        rows = df_csv[df_csv["label"] == lbl].iloc[:12, 0:63].values.astype(np.float32)
        rows = rows[np.any(rows != 0, axis=1)]
        n_take = (
            12
            if lbl_str.upper()
            in ("B", "F", "W", "X", "Q", "D", "G", "P", "L", "I", "Y", "YOU", "FINE", "THANK YOU", "START", "STOP", "ME", "LOVE")
            else 6
        )
        for r_idx, r63 in enumerate(rows[:n_take]):
            base_norm = r63.reshape(21, 3)
            sc = float(rng.uniform(0.08, 0.12))
            w0 = np.array(
                [0.50 + rng.uniform(-0.05, 0.05), 0.50 + rng.uniform(-0.05, 0.05), 0.0],
                dtype=np.float32,
            )
            mode_type = r_idx % 3
            if mode_type == 0:
                drift_x, drift_y = 0.0, 0.0
                start_pose = base_norm
            elif mode_type == 1:
                drift_x = float(rng.uniform(-0.65, 0.65)) * sc
                drift_y = float(rng.uniform(-1.55, 0.65)) * sc
                start_pose = base_norm
            else:
                drift_x = float(rng.uniform(-0.75, 0.75)) * sc
                drift_y = float(rng.uniform(-1.60, 0.65)) * sc
                start_pose = (
                    all_static_poses[int(rng.integers(0, len(all_static_poses)))]
                    if len(all_static_poses) > 0
                    else base_norm
                )

            seq_25 = []
            for step_i in range(SEQ_LEN):
                t = min(1.0, step_i / float(max(1, int(SEQ_LEN * 0.55))))
                cur_pose = (1.0 - t) * start_pose + t * base_norm
                w_t = w0 + np.array([drift_x * t, drift_y * t, 0.0], dtype=np.float32)
                jitter = rng.normal(0.0, 0.006, size=(21, 3)).astype(np.float32)
                jitter[0] = 0.0
                frame_pts = w_t + (cur_pose + jitter) * sc
                seq_25.append(frame_pts.astype(np.float32))
            X_hold.append(extract_dynamic_mlp_features(seq_25))
            y_hold.append("STATIC_HOLD")

    return np.asarray(X_hold, dtype=np.float32), np.asarray(y_hold)


def build_dynamic_dataset(
    dataset_dir: str,
    dynamic_classes: List[str],
    df_csv: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the 25-frame Dynamic Gestures dataset (`EXPECTED_DYNAMIC_DIM = 393`) for `models/dynamic_mlp.keras`.
    """
    cache_path = os.path.join(MODELS_DIR, "dynamic_sequences_v6_cache.npz")
    rng = np.random.default_rng(42)
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            X_dyn = cached["X"].astype(np.float32)
            y_dyn = cached["y"]
            if X_dyn.shape[1] == EXPECTED_DYNAMIC_DIM:
                pos_mask = np.array([str(y) != "STATIC_HOLD" for y in y_dyn])
                X_pos = X_dyn[pos_mask]
                y_pos = y_dyn[pos_mask]
                if len(np.unique(y_pos)) == len(dynamic_classes):
                    X_hold, y_hold = _build_static_hold_dynamic_samples(df_csv, rng)
                    X_all = np.vstack([X_pos, X_hold]).astype(np.float32) if len(X_hold) > 0 else X_pos
                    y_all = np.concatenate([y_pos, y_hold]) if len(y_hold) > 0 else y_pos
                    np.savez_compressed(cache_path, X=X_all, y=y_all)
                    v2_cache_path = os.path.join(MODELS_DIR, "dynamic_sequences_v2_cache.npz")
                    np.savez_compressed(v2_cache_path, X=X_all, y=y_all)
                    print(
                        f"[Dynamic Dataset] Loaded {len(X_pos)} dynamic + {len(X_hold)} STATIC_HOLD "
                        f"25-frame vectors (total={len(X_all)}, dim={EXPECTED_DYNAMIC_DIM})."
                    )
                    return X_all, y_all
        except Exception as e:
            print(f"[Dynamic Dataset] Cache reload note: {e}")

    print(
        f"\n[Dynamic Dataset] Extracting 25-frame sequences (dim={EXPECTED_DYNAMIC_DIM}) "
        f"for dynamic classes: {dynamic_classes}..."
    )
    enhancer = LowLightEnhancer()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    rng = np.random.default_rng(42)
    X_dyn_list = []
    y_dyn_list = []

    for dyn_label in dynamic_classes:
        class_dir = os.path.join(dataset_dir, dyn_label)
        raw_trajectories = []

        if os.path.exists(class_dir) and os.path.isdir(class_dir):
            entries = sorted(os.listdir(class_dir), key=natural_sort_key)
            video_files = [f for f in entries if f.lower().endswith(VIDEO_EXTENSIONS)]
            seq_dirs = [d for d in entries if os.path.isdir(os.path.join(class_dir, d))]
            image_files = [f for f in entries if f.lower().endswith(IMAGE_EXTENSIONS)]

            for vf in video_files:
                v_path = os.path.join(class_dir, vf)
                cap = cv2.VideoCapture(v_path)
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                step = max(1, total_f // 55) if total_f > 70 else 1
                traj = []
                f_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if f_idx % step == 0:
                        flipped = cv2.flip(frame, 1)
                        enh, _, _ = enhancer.process(flipped)
                        res = hands.process(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB))
                        pts = _extract_dominant_hand_21x3(res)
                        if pts is not None and pts[0, 1] < 0.78:
                            traj.append(pts)
                    f_idx += 1
                cap.release()
                if len(traj) >= 6:
                    n_t = len(traj)
                    core_traj = traj[int(n_t * 0.14) : max(int(n_t * 0.14) + 5, int(n_t * 0.88))]
                    if len(core_traj) >= 5:
                        raw_trajectories.append(core_traj)
                    if n_t >= 18:
                        for w_frac in (0.45, 0.60, 0.75):
                            w_len = max(10, int(n_t * w_frac))
                            for s_pos in (
                                int(n_t * 0.10),
                                int(n_t * 0.20),
                                max(0, n_t - w_len - int(n_t * 0.08)),
                            ):
                                sub_t = traj[s_pos : s_pos + w_len]
                                if len(sub_t) >= 8:
                                    raw_trajectories.append(sub_t)
                    else:
                        raw_trajectories.append(traj)

            for sd in seq_dirs:
                s_path = os.path.join(class_dir, sd)
                s_imgs = sorted(
                    [f for f in os.listdir(s_path) if f.lower().endswith(IMAGE_EXTENSIONS)],
                    key=natural_sort_key,
                )
                traj = []
                for im_name in s_imgs:
                    img = cv2.imread(os.path.join(s_path, im_name))
                    if img is None:
                        continue
                    flipped = cv2.flip(img, 1)
                    enh, _, _ = enhancer.process(flipped)
                    res = hands.process(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB))
                    pts = _extract_dominant_hand_21x3(res)
                    if pts is not None:
                        traj.append(pts)
                if len(traj) >= 4:
                    raw_trajectories.append(traj)

            if len(raw_trajectories) < 4 and len(image_files) >= 6:
                import re as _re
                def _img_ts_sec(fname: str) -> int:
                    m = _re.search(r"_(\d{2})_(\d{2})_(\d{2})_Pro", fname)
                    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else -1

                # Group consecutive burst captures (gap <= 5s) as genuine temporal motion sequences!
                burst_chunks = []
                if _img_ts_sec(image_files[0]) >= 0:
                    cur_b = [image_files[0]]
                    for im_f in image_files[1:]:
                        t_prev = _img_ts_sec(cur_b[-1])
                        t_cur = _img_ts_sec(im_f)
                        if t_prev >= 0 and t_cur >= 0 and (t_cur - t_prev) > 5:
                            if len(cur_b) >= 6:
                                burst_chunks.append(cur_b)
                            cur_b = [im_f]
                        else:
                            cur_b.append(im_f)
                    if len(cur_b) >= 6:
                        burst_chunks.append(cur_b)

                if not burst_chunks:
                    chunk_size = max(6, min(25, len(image_files) // 4))
                    burst_chunks = [
                        image_files[c_start : c_start + chunk_size]
                        for c_start in range(0, len(image_files), chunk_size)
                        if len(image_files[c_start : c_start + chunk_size]) >= 4
                    ]

                for chunk in burst_chunks:
                    traj = []
                    for im_name in chunk:
                        img = cv2.imread(os.path.join(class_dir, im_name))
                        if img is None:
                            continue
                        flipped = cv2.flip(img, 1)
                        enh, _, _ = enhancer.process(flipped)
                        res = hands.process(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB))
                        pts = _extract_dominant_hand_21x3(res)
                        if pts is not None:
                            traj.append(pts)
                    if len(traj) >= 6:
                        raw_trajectories.append(traj)
                        if len(traj) >= 24:
                            n_tr = len(traj)
                            raw_trajectories.append(traj[int(n_tr * 0.05) : int(n_tr * 0.85)])
                            raw_trajectories.append(traj[int(n_tr * 0.15) : n_tr])

        start_col_lbl = f"{dyn_label}_START"
        end_col_lbl = f"{dyn_label}_END"
        if (
            df_csv is not None
            and start_col_lbl in df_csv["label"].values
            and end_col_lbl in df_csv["label"].values
        ):
            s_rows = df_csv[df_csv["label"] == start_col_lbl].iloc[:, 0:63].values
            e_rows = df_csv[df_csv["label"] == end_col_lbl].iloc[:, 0:63].values
            s_rows = s_rows[np.any(s_rows != 0, axis=1)]
            e_rows = e_rows[np.any(e_rows != 0, axis=1)]

            if len(s_rows) > 0 and len(e_rows) > 0:
                n_pairs = min(30, max(len(s_rows), len(e_rows)))
                for p_i in range(n_pairs):
                    s_norm = s_rows[p_i % len(s_rows)].reshape(21, 3)
                    e_norm = e_rows[p_i % len(e_rows)].reshape(21, 3)
                    sc = float(rng.uniform(0.07, 0.11))
                    amp = float(rng.uniform(0.75, 1.65))
                    w0 = np.array(
                        [0.50 + rng.uniform(-0.04, 0.04), 0.45 + rng.uniform(-0.04, 0.04), 0.0],
                        dtype=np.float32,
                    )

                    synth_traj = []
                    for step_idx in range(SEQ_LEN):
                        t = step_idx / float(SEQ_LEN - 1)
                        interp_shape = (1.0 - t) * s_norm + t * e_norm
                        dyn_upper = dyn_label.upper()
                        if dyn_upper == "J":
                            dx = -0.65 * amp * sc * np.sin(np.pi * t)
                            dy = 1.15 * amp * sc * np.sin(np.pi * min(1.0, t * 1.25))
                        elif dyn_upper == "Z":
                            z_dir = 1.0 if (p_i % 2 == 0) else -1.0
                            if t < 0.33:
                                u = t / 0.33
                                dx, dy = z_dir * 1.15 * amp * sc * u, 0.10 * amp * sc * u
                            elif t < 0.66:
                                u = (t - 0.33) / 0.33
                                dx, dy = (
                                    z_dir * 1.15 * amp * sc * (1.0 - u),
                                    (0.10 + 0.95 * u) * amp * sc,
                                )
                            else:
                                u = (t - 0.66) / 0.34
                                dx, dy = (
                                    z_dir * 1.15 * amp * sc * u,
                                    (1.05 + 0.15 * u) * amp * sc,
                                )
                        elif dyn_upper == "HELLO":
                            # Outward temple/forehead salute or wave sweep (+X outward, gentle arc in Y)
                            dx = 0.85 * amp * sc * t
                            dy = 0.12 * amp * sc * np.sin(np.pi * t)
                        elif "HOW ARE YOU" in dyn_upper:
                            dx = 0.45 * amp * sc * np.sin(np.pi * 0.5 * t)
                            dy = 0.95 * amp * sc * t
                        else:
                            dx = -0.95 * amp * sc * np.sin(np.pi * 0.5 * t)
                            dy = -0.65 * amp * sc * t

                        w_t = w0 + np.array([dx, dy, 0.0], dtype=np.float32)
                        frame_pts = w_t + interp_shape * sc
                        synth_traj.append(frame_pts.astype(np.float32))
                    raw_trajectories.append(synth_traj)

        augs_per_traj = max(4, min(20, 140 // max(1, len(raw_trajectories))))
        count_for_class = 0
        for traj in raw_trajectories:
            aug_seqs = _augment_sequence_windows(traj, num_augs=augs_per_traj, rng=rng)
            for seq_25 in aug_seqs:
                feats_dyn = extract_dynamic_mlp_features(seq_25)
                X_dyn_list.append(feats_dyn)
                y_dyn_list.append(dyn_label)
                count_for_class += 1

        print(
            f"  -> Dynamic class '{dyn_label}': generated {count_for_class} 25-frame sequence vectors "
            f"from {len(raw_trajectories)} base trajectories."
        )

    hands.close()
    X_hold, y_hold = _build_static_hold_dynamic_samples(df_csv, rng)
    X_pos_arr = np.asarray(X_dyn_list, dtype=np.float32)
    y_pos_arr = np.asarray(y_dyn_list)
    X_dyn_arr = np.vstack([X_pos_arr, X_hold]).astype(np.float32) if len(X_hold) > 0 else X_pos_arr
    y_dyn_arr = np.concatenate([y_pos_arr, y_hold]) if len(y_hold) > 0 else y_pos_arr

    np.savez_compressed(cache_path, X=X_dyn_arr, y=y_dyn_arr)
    v2_cache_path = os.path.join(MODELS_DIR, "dynamic_sequences_v2_cache.npz")
    np.savez_compressed(v2_cache_path, X=X_dyn_arr, y=y_dyn_arr)
    print(
        f"[Dynamic Dataset] Saved {len(X_pos_arr)} dynamic + {len(X_hold)} STATIC_HOLD "
        f"sequence vectors to '{cache_path}'."
    )
    return X_dyn_arr, y_dyn_arr


def build_augmented_dynamic_keyframes_dataset(
    X_kf_raw: np.ndarray,
    y_kf: np.ndarray,
    num_augs: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the 252-dim Dynamic Keyframe (`_START` / `_END`) dataset purely from the
    8 dynamic keyframe classes:
      - Strictly 1-hand: `J_START`, `J_END`, `Z_START`, `Z_END`
      - Strictly 2-hand: `How are you_START`, `How are you_END`, `Nice to meet you_START`, `Nice to meet you_END`
    """
    rng = np.random.default_rng(42)
    feats_list = []
    labels_list = []

    for row, lbl in zip(X_kf_raw, y_kf):
        r_base = np.asarray(row[:63], dtype=np.float32).reshape(21, 3)
        l_base = np.asarray(row[63:126], dtype=np.float32).reshape(21, 3)
        lbl_lower = str(lbl).strip().lower()

        has_r = not np.all(np.abs(r_base) < 1e-7)
        has_l = not np.all(np.abs(l_base) < 1e-7)
        if not has_r and not has_l:
            continue

        if (not has_r) and has_l:
            r_base = l_base.copy()
            l_base = np.zeros((21, 3), dtype=np.float32)
            has_r = True
            has_l = False

        if lbl_lower in STRICT_ONE_HAND_DYN_KEYFRAMES:
            l_base = np.zeros((21, 3), dtype=np.float32)
            has_l = False

        if lbl_lower in STRICT_TWO_HAND_DYN_KEYFRAMES and not has_l:
            continue

        htype = "both" if (has_r and has_l) else "right"

        feats_list.append(extract_static_word_features((r_base, l_base), hand_type=htype))
        labels_list.append(lbl)
        if has_r and has_l:
            feats_list.append(extract_static_word_features((l_base, r_base), hand_type="both"))
            labels_list.append(lbl)

        n_kf_augs = (
            (num_augs + 6)
            if lbl_lower == "hello_end"
            else (num_augs + (2 if not has_l else 0))
        )
        rz_range = (-5.0, 5.0) if lbl_lower in ("hello_start", "hello_end") else (-13.0, 13.0)
        if lbl_lower == "hello_end":
            rz_range = (-2.0, 14.0)  # Keep hello_END tilted outward (>= +9 deg, never rotating into upright B!)
        elif lbl_lower == "hello_start":
            rz_range = (-12.0, 2.0)  # Keep hello_START tilted inward (<= -16 deg, never rotating into upright B!)

        for aug_i in range(n_kf_augs):
            r_pts = r_base.copy()
            l_pts = l_base.copy()
            if has_r:
                # Also synthesize open-thumb 5-finger variations for hello_START / hello_END so waving with an open thumb works too
                if lbl_lower in ("hello_start", "hello_end") and (aug_i % 2 == 1):
                    v_out = (r_pts[5] - r_pts[17]) * float(rng.uniform(0.35, 0.70)) + (r_pts[9] - r_pts[0]) * 0.25
                    r_pts[3] = r_pts[2] + v_out * 0.55
                    r_pts[4] = r_pts[2] + v_out * 1.10
                r_pts[:, 2] *= float(rng.uniform(0.30, 1.65))
                r_noise = rng.normal(0.0, 0.024, size=r_pts.shape).astype(np.float32)
                r_noise[0] = 0.0
                r_pts = _rotate_pts_3d(
                    r_pts + r_noise,
                    float(rng.uniform(-11.0, 11.0)),
                    float(rng.uniform(-11.0, 11.0)),
                    float(rng.uniform(rz_range[0], rz_range[1])),
                )
                r_pts[0] = 0.0
            if has_l:
                l_pts[:, 2] *= float(rng.uniform(0.30, 1.65))
                l_noise = rng.normal(0.0, 0.024, size=l_pts.shape).astype(np.float32)
                l_noise[0] = 0.0
                l_pts = _rotate_pts_3d(
                    l_pts + l_noise,
                    float(rng.uniform(-11.0, 11.0)),
                    float(rng.uniform(-11.0, 11.0)),
                    float(rng.uniform(-13.0, 13.0)),
                )
                l_pts[0] = 0.0
            feats_list.append(extract_static_word_features((r_pts, l_pts), hand_type=htype))
            labels_list.append(lbl)
            if has_r and has_l:
                feats_list.append(extract_static_word_features((l_pts, r_pts), hand_type="both"))
                labels_list.append(lbl)

    return np.asarray(feats_list, dtype=np.float32), np.asarray(labels_list)


def main() -> Dict:
    print("\n" + "=" * 72)
    print("STANDALONE TRAINING 3/3: PURE DYNAMIC GESTURES MODEL GROUP")
    print("=" * 72)
    df = load_dataset_df()
    X_raw_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values

    registry = auto_categorize_dataset(DATASET_DIR, df)

    # ── Model 3A: 25-Frame Dynamic Trajectory MLP ──────────────────────────
    X_dyn, y_dyn = build_dynamic_dataset(
        dataset_dir=DATASET_DIR,
        dynamic_classes=registry["dynamic_gestures"],
        df_csv=df,
    )
    assert X_dyn.shape[1] == EXPECTED_DYNAMIC_DIM

    _, _, _, stats_dyn = train_and_evaluate_mlp_with_scaler(
        X=X_dyn,
        y=y_dyn,
        model_prefix="dynamic",
        model_title="Model Group 3A: Dynamic Gestures 25-Frame Trajectory MLP",
        epochs=120,
    )
    print_detailed_report("MODEL GROUP 3A: DYNAMIC 25-FRAME TRAJECTORY MLP REPORT", stats_dyn)

    # ── Model 3B: Dynamic Keyframe (_START / _END) MLP ─────────────────────
    kf_mask = np.array([str(lbl).strip().lower() in DYN_KEYFRAME_LABELS for lbl in y_all])
    X_kf_raw = X_raw_all[kf_mask]
    y_kf_base = np.array([str(lbl).strip() for lbl in y_all[kf_mask]])

    print(
        f"\n[Dynamic Keyframes Engine] Computing {EXPECTED_WORD_DIM}-dim features "
        f"for {len(X_kf_raw)} base _START/_END samples..."
    )
    X_kf, y_kf = build_augmented_dynamic_keyframes_dataset(X_kf_raw, y_kf_base, num_augs=3)
    assert X_kf.shape[1] == EXPECTED_WORD_DIM

    _, _, _, stats_kf = train_and_evaluate_mlp_with_scaler(
        X=X_kf,
        y=y_kf,
        model_prefix="dynamic_keyframe",
        model_title="Model Group 3B: Dynamic Keyframe (_START/_END) MLP",
        epochs=95,
    )
    print_detailed_report("MODEL GROUP 3B: DYNAMIC KEYFRAME (_START/_END) MLP REPORT", stats_kf)

    print("\n[DONE] Dynamic Gestures Model Group (3A + 3B) trained independently and saved to 'models/'.")
    return {"dynamic_trajectory": stats_dyn, "dynamic_keyframe": stats_kf}


if __name__ == "__main__":
    main()
