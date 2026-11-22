"""
Standalone Trainer 1: Static ASL Two-Stage Cooperative MLPs (`train_asl_model.py`)
==================================================================================
Trains ONLY Model Group 1 (Static ASL alphabet letters, purely on Static ASL data
with ZERO word or dynamic classes mixed in):
  - Stage 1 Shape-Family Router: `models/asl_stage1_router_mlp.keras`
  - Stage 2 Shape-Family Specialists:
      * `models/asl_specialist_cluster_fist_mlp.keras`
      * `models/asl_specialist_cluster_upright_mlp.keras`
      * `models/asl_specialist_cluster_curve_pinch_mlp.keras`
      * `models/asl_specialist_cluster_open_mlp.keras`
All models are 100% pure Keras MLPs (`Dense`, `BatchNormalization`, `Dropout`).
"""

import os
from typing import Dict, Tuple

import numpy as np

from aslo_features import EXPECTED_ASL_DIM, extract_asl_features
from train_utils import (
    DATASET_DIR,
    MODELS_DIR,
    _rotate_pts_3d,
    auto_categorize_dataset,
    load_dataset_df,
    print_detailed_report,
    train_and_evaluate_mlp_with_scaler,
)


def build_augmented_asl_dataset(
    X_asl_raw: np.ndarray,
    y_asl: np.ndarray,
    num_augs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs 3D landmark-space augmentation BEFORE calling `extract_asl_features` so the
    StandardScaler and Stage-1/Stage-2 MLPs are invariant to live webcam MediaPipe depth noise,
    camera tilt, and natural user finger-spread variation (especially separating U, V, R, K, X, Q).
    """
    rng = np.random.default_rng(42)
    feats_list = []
    labels_list = []

    for row, lbl in zip(X_asl_raw, y_asl):
        base_pts = np.asarray(row[:63], dtype=np.float32).reshape(21, 3)
        if np.all(np.abs(base_pts) < 1e-7) and len(row) >= 126:
            alt_pts = np.asarray(row[63:126], dtype=np.float32).reshape(21, 3)
            if np.any(np.abs(alt_pts) > 1e-7):
                base_pts = alt_pts
        if np.all(np.abs(base_pts) < 1e-7):
            continue
        feats_list.append(extract_asl_features(base_pts))
        labels_list.append(lbl)

        augs_for_row = 3 if lbl in ("U", "V", "R", "K", "P", "Q", "X") else num_augs
        for _ in range(augs_for_row):
            pts = base_pts.copy()
            z_scale = float(rng.uniform(0.30, 1.65))
            pts[:, 2] *= z_scale
            noise = rng.normal(0.0, 0.018, size=pts.shape).astype(np.float32)
            noise[0] = 0.0

            if lbl in ("U", "V", "R", "K"):
                for j_other in (1, 2, 3, 13, 14, 15, 16, 17, 18, 19, 20):
                    noise[j_other] = rng.normal(0.0, 0.032, size=3).astype(np.float32)
                pts += noise

                dx_mcp = float(base_pts[5, 0] - base_pts[9, 0])
                sign_x = 1.0 if dx_mcp >= 0.0 else -1.0
                w_mcp = abs(dx_mcp) if abs(dx_mcp) >= 0.06 else 0.11
                mid_tip_x = 0.5 * float(base_pts[8, 0] + base_pts[12, 0])
                mid_dip_x = 0.5 * float(base_pts[7, 0] + base_pts[11, 0])
                mid_pip_x = 0.5 * float(base_pts[6, 0] + base_pts[10, 0])

                if lbl == "U":
                    target_ratio = float(rng.uniform(0.52, 1.22))
                    sep_tip = target_ratio * w_mcp * sign_x
                    sep_dip = (0.85 * target_ratio + 0.12) * w_mcp * sign_x
                    sep_pip = 0.92 * w_mcp * sign_x
                    pts[8, 0], pts[12, 0] = mid_tip_x + 0.5 * sep_tip, mid_tip_x - 0.5 * sep_tip
                    pts[7, 0], pts[11, 0] = mid_dip_x + 0.5 * sep_dip, mid_dip_x - 0.5 * sep_dip
                    pts[6, 0], pts[10, 0] = mid_pip_x + 0.5 * sep_pip, mid_pip_x - 0.5 * sep_pip
                    pts[4, 1] = max(float(pts[4, 1]), -1.16)
                elif lbl == "V":
                    target_ratio = float(rng.uniform(1.55, 2.85))
                    sep_tip = target_ratio * w_mcp * sign_x
                    sep_dip = (0.72 * target_ratio + 0.25) * w_mcp * sign_x
                    sep_pip = (0.45 * target_ratio + 0.45) * w_mcp * sign_x
                    pts[8, 0], pts[12, 0] = mid_tip_x + 0.5 * sep_tip, mid_tip_x - 0.5 * sep_tip
                    pts[7, 0], pts[11, 0] = mid_dip_x + 0.5 * sep_dip, mid_dip_x - 0.5 * sep_dip
                    pts[6, 0], pts[10, 0] = mid_pip_x + 0.5 * sep_pip, mid_pip_x - 0.5 * sep_pip
                    pts[4, 1] = max(float(pts[4, 1]), -1.14)
                elif lbl == "R":
                    target_ratio = float(rng.uniform(-0.75, 0.32))
                    sep_tip = target_ratio * w_mcp * sign_x
                    sep_dip = (0.55 * target_ratio + 0.15) * w_mcp * sign_x
                    pts[8, 0], pts[12, 0] = mid_tip_x + 0.5 * sep_tip, mid_tip_x - 0.5 * sep_tip
                    pts[7, 0], pts[11, 0] = mid_dip_x + 0.5 * sep_dip, mid_dip_x - 0.5 * sep_dip
                elif lbl == "K":
                    pts[4, 1] = min(float(pts[4, 1]), -1.24)
            elif lbl == "X":
                pts += noise
                hook_drop = float(rng.uniform(0.0, 0.22))
                pts[8, 1] = min(float(pts[8, 1]) + hook_drop, -0.95)
                pts[8, 2] = float(pts[8, 2]) - float(rng.uniform(0.0, 0.12))
                pts[4, 1] = min(float(pts[4, 1]), -0.25)
            elif lbl == "Q":
                pts += noise
                pts[8, 1] = max(float(pts[8, 1]), 0.15)
                pts[4, 1] = max(float(pts[4, 1]), 0.35)
            elif lbl == "P":
                pts += noise
                pts[8, 1] = float(np.clip(pts[8, 1], -0.98, 0.05))
                pts[4, 1] = min(float(pts[4, 1]), 0.10)
            else:
                pts += noise

            rz_lim = 3.5 if lbl == "B" else 11.0
            pts = _rotate_pts_3d(
                pts,
                float(rng.uniform(-9.0, 9.0)),
                float(rng.uniform(-9.0, 9.0)),
                float(rng.uniform(-rz_lim, rz_lim)),
            )
            pts[0] = 0.0
            feats_list.append(extract_asl_features(pts))
            labels_list.append(lbl)

    return np.asarray(feats_list, dtype=np.float32), np.asarray(labels_list)


def main() -> Dict:
    print("\n" + "=" * 72)
    print("STANDALONE TRAINING 1/3: PURE STATIC ASL MODEL GROUP")
    print("=" * 72)
    df = load_dataset_df()
    X_raw_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values

    registry = auto_categorize_dataset(DATASET_DIR, df)
    label_to_type = registry["label_to_type"]
    shape_families = registry["shape_families"]
    sign_to_family = registry["sign_to_family"]

    asl_mask = np.array([label_to_type.get(str(lbl), "") == "static_asl" for lbl in y_all])
    X_asl_raw = X_raw_all[asl_mask]
    y_asl_raw = np.array([str(lbl).upper().strip() for lbl in y_all[asl_mask]])

    print(
        f"\n[ASL Feature Engine] Computing {EXPECTED_ASL_DIM}-dim palm-aligned features "
        f"(pure Static ASL only, {len(X_asl_raw)} base samples)..."
    )
    X_asl, y_asl = build_augmented_asl_dataset(X_asl_raw, y_asl_raw, num_augs=1)
    assert X_asl.shape[1] == EXPECTED_ASL_DIM

    # Stage-1 Router: PURE Static ASL shape families + singletons (NO words or dynamic classes!)
    y_stage1_asl = np.array([sign_to_family.get(lbl, lbl) for lbl in y_asl])

    _, _, _, stats_stage1 = train_and_evaluate_mlp_with_scaler(
        X=X_asl,
        y=y_stage1_asl,
        model_prefix="asl_stage1_router",
        model_title="Stage-1 Pure Static ASL Shape-Family Router MLP",
        epochs=95,
    )
    print_detailed_report("STAGE-1 PURE ASL ROUTER MLP REPORT", stats_stage1)

    for src_ext, dst_name in [
        ("_mlp.keras", "asl_stage1_mlp.keras"),
        ("_labels.npy", "asl_stage1_labels.npy"),
        ("_scaler.npz", "asl_stage1_scaler.npz"),
    ]:
        src_p = os.path.join(MODELS_DIR, f"asl_stage1_router{src_ext}")
        dst_p = os.path.join(MODELS_DIR, dst_name)
        if os.path.exists(src_p):
            with open(src_p, "rb") as rf, open(dst_p, "wb") as wf:
                wf.write(rf.read())

    # Stage-2 Specialists: one dedicated MLP per active shape family
    specialist_stats = {}
    for cluster_name, members in shape_families.items():
        short_key = cluster_name.lower()
        c_mask = np.array([lbl in members for lbl in y_asl])
        X_cluster = X_asl[c_mask]
        y_cluster = y_asl[c_mask]

        prefix = f"asl_specialist_{short_key}"
        _, _, _, c_stats = train_and_evaluate_mlp_with_scaler(
            X=X_cluster,
            y=y_cluster,
            model_prefix=prefix,
            model_title=f"Stage-2 Specialist MLP ({cluster_name}: {members})",
            epochs=100,
        )
        specialist_stats[cluster_name] = c_stats
        print_detailed_report(f"STAGE-2 SPECIALIST REPORT [{cluster_name}]", c_stats)

        if cluster_name == "CLUSTER_FIST":
            for src_ext, dst_name in [
                ("_mlp.keras", "asl_fist_specialist_mlp.keras"),
                ("_labels.npy", "asl_fist_labels.npy"),
                ("_scaler.npz", "asl_fist_scaler.npz"),
            ]:
                src_p = os.path.join(MODELS_DIR, f"{prefix}{src_ext}")
                dst_p = os.path.join(MODELS_DIR, dst_name)
                if os.path.exists(src_p):
                    with open(src_p, "rb") as rf, open(dst_p, "wb") as wf:
                        wf.write(rf.read())

    print("\n[DONE] Static ASL Model Group trained independently and saved to 'models/'.")
    return {"stage1": stats_stage1, "specialists": specialist_stats}


if __name__ == "__main__":
    main()
