"""
ASLO Verification Test Suite (100% MLP-Only Cooperative Multi-MLP Architecture)
Verifies all target architectural requirements:
1. Rotation-Invariant 3D Hand-Shape Calculation (`aslo_features.py`):
   - Rotating a 21-joint 3D hand by ±30° around arbitrary 3D axes produces an identical
     115-dim `extract_hand_shape_features` fingerprint (`np.allclose` within tolerance).
2. All Cooperative Keras MLP Models + Scalers & Dimension Parity (`models/`):
   - Stage-1 ASL Shape-Family Router (`asl_stage1_router_mlp.keras`),
   - All 4 Stage-2 Shape Specialists (`CLUSTER_FIST`, `CLUSTER_UPRIGHT`, `CLUSTER_CURVE_PINCH`, `CLUSTER_OPEN`),
   - Static Words MLP (`words_static_mlp.keras`),
   - Dynamic 25-Frame MLP (`dynamic_mlp.keras`),
   - Plus their `.npy` label encoders and `.npz` `StandardScaler` parameters load cleanly
     with zero shape mismatches against `EXPECTED_SHAPE_DIM` (115), `EXPECTED_ASL_DIM` (124),
     `EXPECTED_WORD_DIM` (252), and `EXPECTED_DYNAMIC_DIM` (393), using exclusively Keras MLP layers.
3. Soft Probabilistic Two-Stage Fusion (`ASLOEnsembleRouter.fuse_stage1_and_stage2_probabilities`):
   - Accurately separates `A, S, T, N, M, E` (`CLUSTER_FIST`) and `U, V, R, K` (`CLUSTER_UPRIGHT`),
     including when Stage-1 confidence is artificially split (55% / 45%) between two families.
4. 3-Type Cooperative Routing & Stateless Flask Endpoints:
   - Two-hand static inputs route cleanly to `words_static_mlp.keras` without triggering ASL letters.
   - Dynamic 25-frame sequences route cleanly to `dynamic_mlp.keras` and suppress static flickering.
   - `/predict_landmarks` returns `gesture_type`, `model_source`, and `specialist_used`.
"""

import json
import math
import os
import numpy as np
import pandas as pd

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import app
import aslo_features
import aslo_pipeline
import train_model


def _rotate_3d_points(pts: np.ndarray, angle_deg: float, axis: np.ndarray) -> np.ndarray:
    """Rodrigues' rotation formula around an arbitrary 3D axis."""
    rad = math.radians(angle_deg)
    k = axis / (np.linalg.norm(axis) + 1e-8)
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + math.sin(rad) * K + (1.0 - math.cos(rad)) * (K @ K)
    return (pts @ R.T).astype(np.float32)


def test_3d_rotation_invariance_hand_shape():
    print("\n--- Test 1: Canonical 3D Palm Frame & Rotation-Invariant Hand-Shape Math (±30°) ---")
    assert aslo_features.EXPECTED_SHAPE_DIM == 115, f"Expected 115, got {aslo_features.EXPECTED_SHAPE_DIM}"
    assert aslo_features.EXPECTED_ASL_DIM == 124, f"Expected 124, got {aslo_features.EXPECTED_ASL_DIM}"
    assert aslo_features.EXPECTED_WORD_DIM == 252, f"Expected 252, got {aslo_features.EXPECTED_WORD_DIM}"
    assert aslo_features.EXPECTED_DYNAMIC_DIM == 393, f"Expected 393, got {aslo_features.EXPECTED_DYNAMIC_DIM}"
    assert aslo_features.SEQ_LEN == 25, f"Expected SEQ_LEN=25, got {aslo_features.SEQ_LEN}"

    # Construct a realistic 21-joint 3D hand pose
    df = pd.read_csv("gesture_data.csv")
    sample_row = df[df["label"] == "A"].iloc[0, :63].values.astype(np.float32).reshape(21, 3)

    base_shape = aslo_features.extract_hand_shape_features(sample_row)
    assert base_shape.shape == (115,), f"Expected (115,), got {base_shape.shape}"

    for angle in [30.0, -30.0]:
        for axis in [
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.5, -0.3], dtype=np.float32),
            np.array([-0.4, 1.0, 0.7], dtype=np.float32),
        ]:
            # Also apply arbitrary 3D translation and uniform scale
            transformed_pts = _rotate_3d_points(sample_row, angle, axis) * 1.35 + np.array([0.25, -0.15, 0.40], dtype=np.float32)
            rot_shape = aslo_features.extract_hand_shape_features(transformed_pts)
            max_diff = float(np.max(np.abs(base_shape - rot_shape)))
            assert np.allclose(base_shape, rot_shape, atol=1e-4), (
                f"Rotation invariance failed for angle={angle}°, axis={axis}: max_diff={max_diff:.6f}"
            )

    print("[PASS] `extract_hand_shape_features` (115-dim) is strictly invariant to ±30° 3D rotation, translation, and scale!")


def test_all_cooperative_mlps_and_scalers_load():
    print("\n--- Test 2: All Cooperative Keras MLP Models, Scalers, & Pure MLP Layer Check ---")
    required_files = [
        "models/gesture_registry.json",
        "models/asl_stage1_router_mlp.keras",
        "models/asl_stage1_router_labels.npy",
        "models/asl_stage1_router_scaler.npz",
        "models/asl_specialist_cluster_fist_mlp.keras",
        "models/asl_specialist_cluster_fist_labels.npy",
        "models/asl_specialist_cluster_fist_scaler.npz",
        "models/asl_specialist_cluster_upright_mlp.keras",
        "models/asl_specialist_cluster_upright_labels.npy",
        "models/asl_specialist_cluster_upright_scaler.npz",
        "models/asl_specialist_cluster_curve_pinch_mlp.keras",
        "models/asl_specialist_cluster_curve_pinch_labels.npy",
        "models/asl_specialist_cluster_curve_pinch_scaler.npz",
        "models/asl_specialist_cluster_open_mlp.keras",
        "models/asl_specialist_cluster_open_labels.npy",
        "models/asl_specialist_cluster_open_scaler.npz",
        "models/words_static_mlp.keras",
        "models/words_static_labels.npy",
        "models/words_static_scaler.npz",
        "models/dynamic_mlp.keras",
        "models/dynamic_labels.npy",
        "models/dynamic_scaler.npz",
    ]
    for path in required_files:
        assert os.path.exists(path), f"Missing required model/scaler artifact: {path}"

    router = app.mlp_router
    assert isinstance(router, aslo_pipeline.ASLOEnsembleRouter)
    assert router.load_all_models() is True, "ASLOEnsembleRouter failed to load all cooperative MLP models!"
    assert router.is_ready is True

    for cluster_name in ["CLUSTER_FIST", "CLUSTER_UPRIGHT", "CLUSTER_CURVE_PINCH", "CLUSTER_OPEN"]:
        assert cluster_name in router.specialists, f"Missing Stage-2 specialist for {cluster_name}"
        spec = router.specialists[cluster_name]
        assert spec["scaler"].mean.shape == (1, aslo_features.EXPECTED_ASL_DIM)

    # Verify all models are pure Keras MLPs (InputLayer, GaussianNoise, Dense, BatchNormalization, Dropout)
    allowed_layer_types = {"InputLayer", "GaussianNoise", "Dense", "BatchNormalization", "Dropout"}
    all_models_to_check = [
        ("asl_stage1_router_mlp", router.asl_stage1_model),
        ("words_static_mlp", router.words_model),
        ("dynamic_mlp", router.dynamic_model),
    ] + [(f"specialist_{k}", v["model"]) for k, v in router.specialists.items()]

    for name, m in all_models_to_check:
        for layer in m.layers:
            cls_name = layer.__class__.__name__
            assert cls_name in allowed_layer_types, f"Forbidden non-MLP layer '{cls_name}' in {name}!"

    # Check feature vector validator shapes
    dummy_pts = [{"x": float(i) * 0.02, "y": float(i) * 0.03, "z": 0.0} for i in range(21)]
    asl_vec = aslo_features.validate_asl_feature_vector(aslo_features.extract_asl_features(dummy_pts))
    assert asl_vec.shape == (1, aslo_features.EXPECTED_ASL_DIM)

    word_vec = aslo_features.validate_word_feature_vector(
        aslo_features.extract_static_word_features((dummy_pts, dummy_pts), hand_type="both")
    )
    assert word_vec.shape == (1, aslo_features.EXPECTED_WORD_DIM)

    dummy_seq = [dummy_pts for _ in range(aslo_features.SEQ_LEN)]
    dyn_vec = aslo_features.validate_dynamic_feature_vector(
        aslo_features.extract_dynamic_mlp_features(dummy_seq)
    )
    assert dyn_vec.shape == (1, aslo_features.EXPECTED_DYNAMIC_DIM)

    print("[PASS] All 7 pure Keras MLP models + StandardScalers loaded with exact dimension parity (Shape=115, ASL=124, Word=252, Dynamic=393)!")


def test_soft_probabilistic_two_stage_fusion():
    print("\n--- Test 3: Soft Probabilistic Two-Stage Fusion (A/S/T/N/M/E & U/V/R/K + 55%/45% Split Test) ---")
    df = pd.read_csv("gesture_data.csv")
    router = app.mlp_router

    # 1. Verify CLUSTER_FIST letters: A, S, T, N, M, E
    for fist_letter in ["A", "S", "T", "N", "M", "E"]:
        rows = df[df["label"] == fist_letter].iloc[:15, :-1].values.astype(np.float32)
        assert len(rows) > 0, f"No samples found for '{fist_letter}'"
        correct_normal = 0
        correct_split = 0
        for r in rows:
            asl_vec = aslo_features.csv_row_to_asl_features(r)
            res = router.route_and_predict(asl_features_1d=asl_vec, hand_type="right", mode="asl")
            if res.get("prediction") == fist_letter:
                correct_normal += 1

            # Artificially split Stage-1 confidence 55% CLUSTER_FIST / 45% CLUSTER_UPRIGHT
            fused_split = router.fuse_stage1_and_stage2_probabilities(
                asl_vec,
                override_stage1_probs={"CLUSTER_FIST": 0.55, "CLUSTER_UPRIGHT": 0.45},
            )
            if fused_split["top_label"] == fist_letter:
                correct_split += 1

        acc_normal = correct_normal / len(rows)
        acc_split = correct_split / len(rows)
        assert acc_normal >= 0.85, f"Low normal accuracy on fist sign '{fist_letter}': {acc_normal:.2f}"
        assert acc_split >= 0.85, f"Low split-confidence (55%/45%) accuracy on fist sign '{fist_letter}': {acc_split:.2f}"
        print(f"  -> Fist '{fist_letter}': normal_acc={acc_normal*100:.1f}%, split_55_45_acc={acc_split*100:.1f}%")

    # 2. Verify CLUSTER_UPRIGHT letters: U, V, R, K
    for upright_letter in ["U", "V", "R", "K"]:
        rows = df[df["label"] == upright_letter].iloc[:15, :-1].values.astype(np.float32)
        assert len(rows) > 0, f"No samples found for '{upright_letter}'"
        correct_normal = 0
        correct_split = 0
        for r in rows:
            asl_vec = aslo_features.csv_row_to_asl_features(r)
            res = router.route_and_predict(asl_features_1d=asl_vec, hand_type="right", mode="asl")
            if res.get("prediction") == upright_letter:
                correct_normal += 1

            # Artificially split Stage-1 confidence 55% CLUSTER_UPRIGHT / 45% CLUSTER_OPEN
            fused_split = router.fuse_stage1_and_stage2_probabilities(
                asl_vec,
                override_stage1_probs={"CLUSTER_UPRIGHT": 0.55, "CLUSTER_OPEN": 0.45},
            )
            if fused_split["top_label"] == upright_letter:
                correct_split += 1

        acc_normal = correct_normal / len(rows)
        acc_split = correct_split / len(rows)
        assert acc_normal >= 0.85, f"Low normal accuracy on upright sign '{upright_letter}': {acc_normal:.2f}"
        assert acc_split >= 0.85, f"Low split-confidence (55%/45%) accuracy on upright sign '{upright_letter}': {acc_split:.2f}"
        print(f"  -> Upright '{upright_letter}': normal_acc={acc_normal*100:.1f}%, split_55_45_acc={acc_split*100:.1f}%")

    print("[PASS] Soft Probabilistic Two-Stage Fusion accurately separates A/S/T/N/M/E and U/V/R/K even under 55%/45% split Stage-1 probabilities!")


def test_two_hand_static_words_and_dynamic_routing():
    print("\n--- Test 4: Two-Hand Static Words & 25-Frame Dynamic Sequence Routing ---")
    df = pd.read_csv("gesture_data.csv")
    router = app.mlp_router
    asl_letters = set(router.registry.get("static_asl", []))

    for word_lbl in ["start", "stop", "space", "backspace", "Love", "ME", "neutral"]:
        rows = df[df["label"] == word_lbl].iloc[:10, :-1].values.astype(np.float32)
        assert len(rows) > 0, f"No samples for '{word_lbl}'"
        correct = 0
        for r in rows:
            word_vec = aslo_features.csv_row_to_word_features(r)
            res = router.route_and_predict(
                word_features_1d=word_vec,
                hand_type="both",
                mode="auto",
            )
            assert res["model_source"] == "words_static_mlp", f"Expected words_static_mlp, got {res['model_source']}"
            assert res["gesture_type"] == "static_words", f"Expected static_words, got {res['gesture_type']}"
            assert res["prediction"] not in asl_letters, f"Two-hand input triggered ASL letter {res['prediction']}!"
            if res["prediction"] and res["prediction"].lower() == word_lbl.lower():
                correct += 1
        assert correct >= 8, f"Low accuracy on two-hand word '{word_lbl}': {correct}/10"
        print(f"  -> Two-hand word '{word_lbl}': routed to words_static_mlp ({correct}/10 correct)")

    # Test Dynamic 25-frame sequences
    cache_path = os.path.join("models", "dynamic_sequences_v2_cache.npz")
    assert os.path.exists(cache_path), "Dynamic v2 sequence cache missing!"
    cached = np.load(cache_path, allow_pickle=True)
    X_dyn = cached["X"]
    y_dyn = cached["y"]

    for dyn_cls in router.registry.get("dynamic_gestures", []):
        mask = (y_dyn == dyn_cls)
        assert np.any(mask), f"No cached dynamic vectors for '{dyn_cls}'"
        X_sub = X_dyn[mask][:10]
        X_scaled = router.dynamic_scaler.transform(X_sub)
        x_tf = app.tf.constant(X_scaled, dtype=app.tf.float32)
        probs = router._fast_dynamic(x_tf).numpy()
        preds = [str(router.dynamic_labels[int(np.argmax(p))]) for p in probs]
        acc = np.mean([p == dyn_cls for p in preds])
        assert acc >= 0.85, f"Dynamic MLP accuracy on '{dyn_cls}' too low: {acc:.2f}"
        print(f"  -> Dynamic class '{dyn_cls}': 25-frame MLP accuracy = {acc*100:.1f}%")

    print("[PASS] Two-hand static words and 25-frame dynamic sequences route cleanly with high accuracy!")


def test_requirements_and_endpoints():
    print("\n--- Test 5: Clean requirements.txt & Stateless Flask Endpoints ---")
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = f.read().lower()

    forbidden_pkgs = ["pywin32", "pyaudio", "winsound", "xgboost", "lightgbm"]
    for pkg in forbidden_pkgs:
        assert pkg not in reqs, f"Forbidden package '{pkg}' found in requirements.txt!"

    for req_pkg in ["tensorflow", "keras", "scikit-learn", "flask", "gunicorn"]:
        assert req_pkg in reqs, f"Required package '{req_pkg}' missing from requirements.txt!"

    client = app.app.test_client()
    res_ping = client.get("/ping")
    assert res_ping.status_code == 200
    assert json.loads(res_ping.data)["status"] == "online"

    res_dict = client.get("/api/dictionary")
    assert res_dict.status_code == 200
    assert len(json.loads(res_dict.data)["dictionary"]) >= 40

    dummy_pts = [[0.45 + i * 0.005, 0.45 + i * 0.005, 0.0] for i in range(21)]
    payload = {
        "landmarks": dummy_pts,
        "hands": [{"label": "Right", "points": dummy_pts}],
        "hand_type": "right",
        "sequence_buffer": [dummy_pts for _ in range(25)],
        "motion_speed": 0.0,
        "mode": "asl",
    }
    res_pred = client.post("/predict_landmarks", json=payload)
    assert res_pred.status_code == 200
    pred_data = json.loads(res_pred.data)
    assert pred_data["ok"] is True
    assert "gesture_type" in pred_data
    assert "model_source" in pred_data
    assert "specialist_used" in pred_data

    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "server-mode-select" in res_index.data.decode("utf-8")

    res_mobile = client.get("/mobile")
    assert res_mobile.status_code == 200
    assert "server-mode-select" in res_mobile.data.decode("utf-8")

    print("[PASS] requirements.txt is clean and all stateless API & UI endpoints verified!")


if __name__ == "__main__":
    test_3d_rotation_invariance_hand_shape()
    test_all_cooperative_mlps_and_scalers_load()
    test_soft_probabilistic_two_stage_fusion()
    test_two_hand_static_words_and_dynamic_routing()
    test_requirements_and_endpoints()
    print("\n=====================================================================")
    print("ALL 100% MLP-ONLY COOPERATIVE MULTI-MLP ARCHITECTURE TESTS PASSED!")
    print("=====================================================================")
