"""
Dedicated Model Merger: OOD-Aware Meta-Router MLP (`merge_models.py`)
=====================================================================
Loads the 3 separately trained pure-MLP model groups from `models/`:
  1. Model Group 1 — Static ASL (`asl_stage1_router_mlp.keras` + 4 Stage-2 Specialists)
  2. Model Group 2 — Static Words (`words_static_mlp.keras`)
  3. Model Group 3 — Dynamic Gestures (`dynamic_mlp.keras` + `dynamic_keyframe_mlp.keras`)

Extracts BOTH:
  - Calibrated `(top1, top2, margin)` probabilities from each model group, AND
  - **Out-of-Distribution (OOD) Embedding Cosine Similarity** from each model's 64-dim
    penultimate MLP layer against its saved per-class centroids (`models/meta_centroids.npz`)
alongside the 252-dim hand features and hand-count flag (265 dims total), and trains a
100% pure Keras **Meta-Router MLP** (`models/meta_router_mlp.keras`) that cleanly separates
`static_asl`, `static_words`, and `dynamic_keyframe` without cross-model overrides.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from keras import models

from aslo_features import (
    EXPECTED_ASL_DIM,
    EXPECTED_DYNAMIC_DIM,
    EXPECTED_WORD_DIM,
    extract_asl_features,
    extract_static_word_features,
)
from train_utils import (
    DATASET_DIR,
    DYN_KEYFRAME_LABELS,
    MODELS_DIR,
    STRICT_ONE_HAND_DYN_KEYFRAMES,
    STRICT_ONE_HAND_WORDS,
    STRICT_TWO_HAND_DYN_KEYFRAMES,
    STRICT_TWO_HAND_WORDS,
    _rotate_pts_3d,
    auto_categorize_dataset,
    load_dataset_df,
    print_detailed_report,
    train_and_evaluate_mlp_with_scaler,
)
from train_words_model import _synthesize_live_thumbs_up_fists

# 252 (word_feats) + 1 (has_second_hand) + 3 * (p1, p2, margin, max_cos_sim) = 265 dims
EXPECTED_META_DIM = EXPECTED_WORD_DIM + 1 + 12


def _load_scaler(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    mean = np.asarray(data["mean"], dtype=np.float32)
    scale = np.asarray(data["scale"], dtype=np.float32)
    scale = np.where(np.abs(scale) < 1e-7, 1.0, scale)
    return mean, scale


def _scale_batch(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((X - mean) / scale, -5.0, 5.0).astype(np.float32)


def _make_embed_model(keras_model: models.Model) -> models.Model:
    """Returns a sub-model outputting the penultimate (pre-softmax 64-dim) MLP layer."""
    return models.Model(inputs=keras_model.input, outputs=keras_model.layers[-2].output)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-7)).astype(np.float32)


def _compute_class_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Computes L2-normalized mean embedding centroid per unique class."""
    emb_norm = _l2_normalize(embeddings)
    centroids = []
    for cls in sorted(np.unique(labels)):
        c_vec = np.mean(emb_norm[labels == cls], axis=0)
        c_norm = float(np.linalg.norm(c_vec))
        if c_norm > 1e-7:
            c_vec = c_vec / c_norm
        centroids.append(c_vec.astype(np.float32))
    return np.asarray(centroids, dtype=np.float32)


def _max_cosine_sim(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Returns (N, 1) max cosine similarity between each sample's penultimate embedding and class centroids."""
    emb_norm = _l2_normalize(embeddings)
    sims = emb_norm @ centroids.T
    return np.max(sims, axis=1, keepdims=True).astype(np.float32)


def _top2_margin_from_probs(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] == 1:
        p1 = probs[:, 0]
        p2 = np.zeros_like(p1)
    else:
        part = np.sort(probs, axis=1)
        p1 = part[:, -1]
        p2 = part[:, -2]
    return np.stack([p1, p2, p1 - p2], axis=1).astype(np.float32)


def _batch_asl_stats(
    X_asl: np.ndarray,
    has_second_hand: np.ndarray,
    s1_model,
    s1_embed_model,
    asl_centroids: np.ndarray,
    s1_labels: np.ndarray,
    s1_mean: np.ndarray,
    s1_scale: np.ndarray,
    specialists: Dict[str, Dict],
) -> np.ndarray:
    """Returns (N, 4) `[p1, p2, margin, max_cos_sim]` for Model 1 (Static ASL)."""
    n = len(X_asl)
    out = np.zeros((n, 4), dtype=np.float32)
    one_hand_idx = np.where(has_second_hand < 0.5)[0]
    if len(one_hand_idx) == 0:
        return out

    X_1h = X_asl[one_hand_idx]
    X_1h_s1 = _scale_batch(X_1h, s1_mean, s1_scale)
    s1_probs = s1_model.predict(X_1h_s1, batch_size=512, verbose=0)
    s1_embs = s1_embed_model.predict(X_1h_s1, batch_size=512, verbose=0)
    cos_sims = _max_cosine_sim(s1_embs, asl_centroids).flatten()

    spec_probs_map = {}
    for c_name, sp in specialists.items():
        spec_probs_map[c_name] = sp["model"].predict(
            _scale_batch(X_1h, sp["mean"], sp["scale"]), batch_size=512, verbose=0
        )

    s1_label_strs = [str(l) for l in s1_labels]
    for row_i, orig_i in enumerate(one_hand_idx):
        p1_row = s1_probs[row_i]
        top2_c = np.argsort(p1_row)[::-1][:2]
        scores = {}
        for c_idx in top2_c:
            c_lbl = s1_label_strs[int(c_idx)]
            p1_c = float(p1_row[int(c_idx)])
            if p1_c <= 1e-6:
                continue
            if c_lbl in specialists:
                sp_labels = specialists[c_lbl]["labels"]
                sp_p = spec_probs_map[c_lbl][row_i]
                for s_j, s_name in enumerate(sp_labels):
                    jp = p1_c * float(sp_p[s_j])
                    s_str = str(s_name)
                    if jp > scores.get(s_str, -1.0):
                        scores[s_str] = jp
            else:
                if p1_c > scores.get(c_lbl, -1.0):
                    scores[c_lbl] = p1_c
        if scores:
            vals = sorted(scores.values(), reverse=True)
            v1 = float(vals[0])
            v2 = float(vals[1]) if len(vals) > 1 else 0.0
            out[orig_i] = [v1, v2, v1 - v2, float(cos_sims[row_i])]

    return out


def _batch_masked_stats(
    X_word: np.ndarray,
    has_second_hand: np.ndarray,
    model,
    embed_model,
    centroids: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    strict_1h_set: set,
    strict_2h_set: set,
) -> np.ndarray:
    """Returns (N, 4) `[p1, p2, margin, max_cos_sim]`."""
    X_s = _scale_batch(X_word, mean, scale)
    probs = model.predict(X_s, batch_size=512, verbose=0).copy()
    embs = embed_model.predict(X_s, batch_size=512, verbose=0)
    cos_sims = _max_cosine_sim(embs, centroids)

    lbl_lows = [str(l).strip().lower() for l in labels]
    mask_1h_cols = np.array([l in strict_1h_set for l in lbl_lows], dtype=bool)
    mask_2h_cols = np.array([l in strict_2h_set for l in lbl_lows], dtype=bool)

    two_h_rows = has_second_hand >= 0.5
    one_h_rows = ~two_h_rows

    if np.any(two_h_rows) and np.any(mask_1h_cols):
        probs[np.ix_(two_h_rows, mask_1h_cols)] = 0.0
    if np.any(one_h_rows) and np.any(mask_2h_cols):
        probs[np.ix_(one_h_rows, mask_2h_cols)] = 0.0

    row_sums = np.sum(probs, axis=1, keepdims=True)
    probs = np.where(row_sums > 1e-6, probs / np.maximum(row_sums, 1e-6), probs)
    top2 = _top2_margin_from_probs(probs)
    return np.concatenate([top2, cos_sims], axis=1).astype(np.float32)


def _collect_meta_samples(df, registry: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collects paired `(X_asl(124), X_word(252), has_second_hand(1), domain_label, fine_label)`
    with SYMMETRIC live-webcam domain-shift augmentation across all 3 domains so `meta_router_mlp`
    has ZERO noise-level bias between Static ASL, Static Words, and Dynamic Keyframes.
    """
    rng = np.random.default_rng(123)
    X_raw_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values
    label_to_type = registry["label_to_type"]

    asl_list: List[np.ndarray] = []
    word_list: List[np.ndarray] = []
    two_hand_list: List[float] = []
    domain_list: List[str] = []
    fine_label_list: List[str] = []

    def _add_1h(pts: np.ndarray, domain: str, flbl: str):
        asl_list.append(extract_asl_features(pts))
        word_list.append(extract_static_word_features(pts, hand_type="right"))
        two_hand_list.append(0.0)
        domain_list.append(domain)
        fine_label_list.append(flbl)

    def _add_2h(r_pts: np.ndarray, l_pts: np.ndarray, domain: str, flbl: str):
        asl_list.append(np.zeros(EXPECTED_ASL_DIM, dtype=np.float32))
        word_list.append(extract_static_word_features((r_pts, l_pts), hand_type="both"))
        two_hand_list.append(1.0)
        domain_list.append(domain)
        fine_label_list.append(flbl)

    for row, lbl in zip(X_raw_all, y_all):
        lbl_str = str(lbl).strip()
        lbl_low = lbl_str.lower()
        r_base = np.asarray(row[:63], dtype=np.float32).reshape(21, 3)
        l_base = np.asarray(row[63:126], dtype=np.float32).reshape(21, 3)
        has_r = not np.all(np.abs(r_base) < 1e-7)
        has_l = not np.all(np.abs(l_base) < 1e-7)
        if not has_r and not has_l:
            continue
        if (not has_r) and has_l:
            r_base = l_base.copy()
            l_base = np.zeros((21, 3), dtype=np.float32)
            has_r = True
            has_l = False

        # 1. Static ASL (strictly 1-hand, symmetric webcam augmentation; keep B strictly vertical!)
        if label_to_type.get(lbl_str, "") == "static_asl":
            lbl_up = lbl_str.upper()
            _add_1h(r_base, "static_asl", lbl_up)
            rz_lim = 3.5 if lbl_up == "B" else 14.0
            for _ in range(2):
                pts = r_base.copy()
                pts[:, 2] *= float(rng.uniform(0.35, 1.60))
                noise = rng.normal(0.0, 0.026, size=pts.shape).astype(np.float32)
                noise[0] = 0.0
                pts = _rotate_pts_3d(
                    pts + noise,
                    float(rng.uniform(-14.0, 14.0)),
                    float(rng.uniform(-14.0, 14.0)),
                    float(rng.uniform(-rz_lim, rz_lim)),
                )
                pts[0] = 0.0
                _add_1h(pts, "static_asl", lbl_up)
            continue

        # 2. Static Words (STRICT 1-hand for Fine/I love you/Thank you/YOU/hello; STRICT 2-hand for start/stop/space/backspace/neutral/Love/ME)
        if label_to_type.get(lbl_str, "") == "static_words":
            if lbl_low in STRICT_ONE_HAND_WORDS:
                l_base = np.zeros((21, 3), dtype=np.float32)
                has_l = False
            if lbl_low in STRICT_TWO_HAND_WORDS and not has_l:
                l_base = r_base.copy()
                l_base[:, 0] = -l_base[:, 0]
                has_l = True

            if has_r and has_l:
                _add_2h(r_base, l_base, "static_words", lbl_str)
                _add_2h(l_base, r_base, "static_words", lbl_str)
            else:
                _add_1h(r_base, "static_words", lbl_str)

            n_augs = 3 if lbl_low in ("start", "stop", "me", "you") else 2
            for _ in range(n_augs):
                r_pts = r_base.copy()
                l_pts = l_base.copy()
                if lbl_low in ("start", "stop", "me"):
                    pf = float(rng.uniform(0.50, 1.80))
                    r_pts[5:18, 1] *= pf
                    if has_l:
                        l_pts[5:18, 1] *= pf
                r_pts[:, 2] *= float(rng.uniform(0.35, 1.60))
                r_n = rng.normal(0.0, 0.028, size=r_pts.shape).astype(np.float32)
                r_n[0] = 0.0
                r_pts = _rotate_pts_3d(
                    r_pts + r_n,
                    float(rng.uniform(-15.0, 15.0)),
                    float(rng.uniform(-15.0, 15.0)),
                    float(rng.uniform(-15.0, 15.0)),
                )
                r_pts[0] = 0.0
                if has_l:
                    l_pts[:, 2] *= float(rng.uniform(0.35, 1.60))
                    l_n = rng.normal(0.0, 0.028, size=l_pts.shape).astype(np.float32)
                    l_n[0] = 0.0
                    l_pts = _rotate_pts_3d(
                        l_pts + l_n,
                        float(rng.uniform(-15.0, 15.0)),
                        float(rng.uniform(-15.0, 15.0)),
                        float(rng.uniform(-15.0, 15.0)),
                    )
                    l_pts[0] = 0.0
                    _add_2h(r_pts, l_pts, "static_words", lbl_str)
                    _add_2h(l_pts, r_pts, "static_words", lbl_str)
                else:
                    _add_1h(r_pts, "static_words", lbl_str)
            continue

        # 3. Dynamic Keyframes (_START / _END)
        if lbl_low in DYN_KEYFRAME_LABELS:
            if lbl_low in STRICT_ONE_HAND_DYN_KEYFRAMES:
                l_base = np.zeros((21, 3), dtype=np.float32)
                has_l = False
            if lbl_low in STRICT_TWO_HAND_DYN_KEYFRAMES and not has_l:
                continue

            if has_r and has_l:
                _add_2h(r_base, l_base, "dynamic_keyframe", lbl_str)
                _add_2h(l_base, r_base, "dynamic_keyframe", lbl_str)
            else:
                _add_1h(r_base, "dynamic_keyframe", lbl_str)

            n_kf_meta = 5 if lbl_low == "hello_end" else 3
            rz_kf = (-2.0, 14.0) if lbl_low == "hello_end" else ((-12.0, 2.0) if lbl_low == "hello_start" else (-14.0, 14.0))
            for aug_k in range(n_kf_meta):
                r_pts = r_base.copy()
                l_pts = l_base.copy()
                if lbl_low in ("hello_start", "hello_end") and (aug_k % 2 == 1):
                    v_out = (r_pts[5] - r_pts[17]) * float(rng.uniform(0.35, 0.70)) + (r_pts[9] - r_pts[0]) * 0.25
                    r_pts[3] = r_pts[2] + v_out * 0.55
                    r_pts[4] = r_pts[2] + v_out * 1.10
                r_pts[:, 2] *= float(rng.uniform(0.35, 1.60))
                r_n = rng.normal(0.0, 0.026, size=r_pts.shape).astype(np.float32)
                r_n[0] = 0.0
                r_pts = _rotate_pts_3d(
                    r_pts + r_n,
                    float(rng.uniform(-14.0, 14.0)),
                    float(rng.uniform(-14.0, 14.0)),
                    float(rng.uniform(rz_kf[0], rz_kf[1])),
                )
                r_pts[0] = 0.0
                if has_l:
                    l_pts[:, 2] *= float(rng.uniform(0.35, 1.60))
                    l_n = rng.normal(0.0, 0.026, size=l_pts.shape).astype(np.float32)
                    l_n[0] = 0.0
                    l_pts = _rotate_pts_3d(
                        l_pts + l_n,
                        float(rng.uniform(-14.0, 14.0)),
                        float(rng.uniform(-14.0, 14.0)),
                        float(rng.uniform(-14.0, 14.0)),
                    )
                    l_pts[0] = 0.0
                    _add_2h(r_pts, l_pts, "dynamic_keyframe", lbl_str)
                    _add_2h(l_pts, r_pts, "dynamic_keyframe", lbl_str)
                else:
                    _add_1h(r_pts, "dynamic_keyframe", lbl_str)

    for r_tu, l_tu in _synthesize_live_thumbs_up_fists(rng, n_samples=90):
        _add_2h(r_tu, l_tu, "static_words", "start")

    return (
        np.asarray(asl_list, dtype=np.float32),
        np.asarray(word_list, dtype=np.float32),
        np.asarray(two_hand_list, dtype=np.float32),
        np.asarray(domain_list),
        np.asarray(fine_label_list),
    )


def main() -> Dict:
    print("\n" + "=" * 72)
    print("MERGING ALL 3 SEPARATELY TRAINED MODEL GROUPS -> OOD-AWARE META-ROUTER MLP")
    print("=" * 72)
    df = load_dataset_df()
    registry = auto_categorize_dataset(DATASET_DIR, df)
    shape_families = registry["shape_families"]

    # 1. Load Model Group 1 (Static ASL)
    s1_model = models.load_model(os.path.join(MODELS_DIR, "asl_stage1_router_mlp.keras"))
    s1_embed = _make_embed_model(s1_model)
    s1_labels = np.load(os.path.join(MODELS_DIR, "asl_stage1_router_labels.npy"), allow_pickle=True)
    s1_mean, s1_scale = _load_scaler(os.path.join(MODELS_DIR, "asl_stage1_router_scaler.npz"))

    specialists = {}
    for c_name in shape_families.keys():
        prefix = os.path.join(MODELS_DIR, f"asl_specialist_{c_name.lower()}")
        sp_m = models.load_model(f"{prefix}_mlp.keras")
        sp_l = np.load(f"{prefix}_labels.npy", allow_pickle=True)
        sp_mean, sp_scale = _load_scaler(f"{prefix}_scaler.npz")
        specialists[c_name] = {"model": sp_m, "labels": sp_l, "mean": sp_mean, "scale": sp_scale}

    # 2. Load Model Group 2 (Static Words)
    w_model = models.load_model(os.path.join(MODELS_DIR, "words_static_mlp.keras"))
    w_embed = _make_embed_model(w_model)
    w_labels = np.load(os.path.join(MODELS_DIR, "words_static_labels.npy"), allow_pickle=True)
    w_mean, w_scale = _load_scaler(os.path.join(MODELS_DIR, "words_static_scaler.npz"))

    # 3. Load Model Group 3 (Dynamic Trajectory + Dynamic Keyframes)
    dyn_model = models.load_model(os.path.join(MODELS_DIR, "dynamic_mlp.keras"))
    dyn_labels = np.load(os.path.join(MODELS_DIR, "dynamic_labels.npy"), allow_pickle=True)
    kf_model = models.load_model(os.path.join(MODELS_DIR, "dynamic_keyframe_mlp.keras"))
    kf_embed = _make_embed_model(kf_model)
    kf_labels = np.load(os.path.join(MODELS_DIR, "dynamic_keyframe_labels.npy"), allow_pickle=True)
    kf_mean, kf_scale = _load_scaler(os.path.join(MODELS_DIR, "dynamic_keyframe_scaler.npz"))

    print(
        f"[Merger] Loaded all 3 standalone model groups:\n"
        f"  - Model 1 (Static ASL): Stage-1 ({len(s1_labels)} routes) + {len(specialists)} Specialists\n"
        f"  - Model 2 (Static Words): {len(w_labels)} classes\n"
        f"  - Model 3 (Dynamic): 25-frame ({len(dyn_labels)} classes) + Keyframes ({len(kf_labels)} classes)"
    )

    print("\n[Merger] Building 265-dim OOD-Aware Meta-Router dataset from all 3 model groups...")
    X_asl, X_word, has_2h, y_domain, y_fine = _collect_meta_samples(df, registry)

    # Compute per-class 64-dim penultimate MLP embedding centroids for each model's OWN in-domain data
    asl_mask = y_domain == "static_asl"
    words_mask = y_domain == "static_words"
    kf_mask = y_domain == "dynamic_keyframe"

    asl_embs_in = s1_embed.predict(_scale_batch(X_asl[asl_mask], s1_mean, s1_scale), batch_size=512, verbose=0)
    asl_centroids = _compute_class_centroids(asl_embs_in, y_fine[asl_mask])

    w_embs_in = w_embed.predict(_scale_batch(X_word[words_mask], w_mean, w_scale), batch_size=512, verbose=0)
    words_centroids = _compute_class_centroids(w_embs_in, y_fine[words_mask])

    kf_embs_in = kf_embed.predict(_scale_batch(X_word[kf_mask], kf_mean, kf_scale), batch_size=512, verbose=0)
    kf_centroids = _compute_class_centroids(kf_embs_in, y_fine[kf_mask])

    centroids_path = os.path.join(MODELS_DIR, "meta_centroids.npz")
    np.savez(
        centroids_path,
        asl_centroids=asl_centroids,
        words_centroids=words_centroids,
        kf_centroids=kf_centroids,
    )
    print(
        f"[Merger] Saved OOD embedding centroids to '{centroids_path}' "
        f"(ASL={asl_centroids.shape}, Words={words_centroids.shape}, DynKF={kf_centroids.shape})"
    )

    asl_stats = _batch_asl_stats(
        X_asl, has_2h, s1_model, s1_embed, asl_centroids, s1_labels, s1_mean, s1_scale, specialists
    )
    words_stats = _batch_masked_stats(
        X_word, has_2h, w_model, w_embed, words_centroids, w_labels, w_mean, w_scale,
        STRICT_ONE_HAND_WORDS, STRICT_TWO_HAND_WORDS,
    )
    kf_stats = _batch_masked_stats(
        X_word, has_2h, kf_model, kf_embed, kf_centroids, kf_labels, kf_mean, kf_scale,
        STRICT_ONE_HAND_DYN_KEYFRAMES, STRICT_TWO_HAND_DYN_KEYFRAMES,
    )

    X_meta = np.concatenate(
        [X_word, has_2h.reshape(-1, 1), asl_stats, words_stats, kf_stats],
        axis=1,
    ).astype(np.float32)
    assert X_meta.shape[1] == EXPECTED_META_DIM, f"Expected {EXPECTED_META_DIM}, got {X_meta.shape[1]}"

    _, _, _, stats_meta = train_and_evaluate_mlp_with_scaler(
        X=X_meta,
        y=y_domain,
        model_prefix="meta_router",
        model_title="Merged OOD-Aware Meta-Router MLP (static_asl vs static_words vs dynamic_keyframe)",
        epochs=90,
        batch_size=64,
    )
    print_detailed_report("OOD-AWARE META-ROUTER MERGER MLP REPORT", stats_meta)

    manifest = {
        "architecture": "100% Pure Keras MLP Separate-Training + OOD-Aware Meta-Router Ensemble",
        "meta_dim": EXPECTED_META_DIM,
        "models": {
            "static_asl_stage1": {
                "file": "asl_stage1_router_mlp.keras",
                "input_dim": EXPECTED_ASL_DIM,
                "classes": [str(c) for c in s1_labels],
            },
            "static_asl_specialists": {
                c_name: {
                    "file": f"asl_specialist_{c_name.lower()}_mlp.keras",
                    "input_dim": EXPECTED_ASL_DIM,
                    "classes": [str(c) for c in sp["labels"]],
                }
                for c_name, sp in specialists.items()
            },
            "static_words": {
                "file": "words_static_mlp.keras",
                "input_dim": EXPECTED_WORD_DIM,
                "classes": [str(c) for c in w_labels],
            },
            "dynamic_trajectory": {
                "file": "dynamic_mlp.keras",
                "input_dim": EXPECTED_DYNAMIC_DIM,
                "classes": [str(c) for c in dyn_labels],
            },
            "dynamic_keyframe": {
                "file": "dynamic_keyframe_mlp.keras",
                "input_dim": EXPECTED_WORD_DIM,
                "classes": [str(c) for c in kf_labels],
            },
            "meta_router": {
                "file": "meta_router_mlp.keras",
                "input_dim": EXPECTED_META_DIM,
                "classes": stats_meta["classes"],
                "val_acc": stats_meta["val_acc"],
            },
        },
    }
    manifest_path = os.path.join(MODELS_DIR, "merged_ensemble_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"\n[DONE] Saved merged ensemble manifest to '{manifest_path}' "
        f"(Meta-Router Val Acc: {stats_meta['val_acc']*100:.2f}%)."
    )
    return {"meta_router": stats_meta, "manifest": manifest}


if __name__ == "__main__":
    main()
