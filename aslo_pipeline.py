"""
ASLO (AI Sign Language Online) Pipeline Module (`aslo_pipeline.py`)
===================================================================
Provides high-performance, stateless modules for real-time (15-24 FPS) processing:
1. LowLightEnhancer: Adaptive CLAHE & luminance correction for < 100 Lux environments.
2. KeypointInterpolator: Smoothing, occlusion handling, >1.5m distance checks & interpolation.
3. MotionBasedRouter: Motion-based routing separating static poses from dynamic gestures.
4. SoftmaxMarginVerifier: Confidence and margin ambiguity rejection verifier.
5. ASLOEnsembleRouter: Stateless Soft Cooperative Two-Stage & Multi-MLP Ensemble Router
   wrapping all Keras MLPs with `@tf.function(reduce_retracing=True)` and applying
   saved `StandardScaler` normalizers and probabilistic Stage-1 x Stage-2 fusion.
"""

import collections
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
from keras import models
import mediapipe as mp
import numpy as np
import tensorflow as tf

import aslo_features


# =====================================================================
# 1. ADAPTIVE LIGHTING PREPROCESSING (LOW-LIGHT & OVEREXPOSURE / GLARE)
# =====================================================================

class AdaptiveLightingEnhancer:
    def __init__(self, base_clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.base_clip_limit = base_clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe_cache: Dict[float, cv2.CLAHE] = {}
        self._gamma_tables: Dict[float, np.ndarray] = {}

    def _get_clahe(self, clip_limit: float) -> cv2.CLAHE:
        clip_key = round(float(clip_limit), 1)
        if clip_key not in self._clahe_cache:
            self._clahe_cache[clip_key] = cv2.createCLAHE(
                clipLimit=clip_key, tileGridSize=self.tile_grid_size
            )
        return self._clahe_cache[clip_key]

    def _get_gamma_table(self, inv_gamma: float) -> np.ndarray:
        gamma_key = round(float(inv_gamma), 2)
        if gamma_key not in self._gamma_tables:
            table = np.array([((i / 255.0) ** gamma_key) * 255 for i in range(256)]).astype("uint8")
            self._gamma_tables[gamma_key] = table
        return self._gamma_tables[gamma_key]

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, str]:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        mean_l = float(np.mean(l_channel))
        highlight_ratio = float(np.mean(l_channel > 215))

        if mean_l < 75.0:
            adaptive_clip = np.clip(self.base_clip_limit + (75.0 - mean_l) * 0.05, 2.0, 4.5)
            clahe = self._get_clahe(adaptive_clip)
            l_enhanced = clahe.apply(l_channel)

            if mean_l < 45.0:
                lut = self._get_gamma_table(1.0 / 0.82)
                l_enhanced = cv2.LUT(l_enhanced, lut)

            lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR), mean_l, "LOW_LIGHT"

        elif mean_l > 135.0 or highlight_ratio > 0.12:
            gamma_power = float(np.clip(1.30 + (mean_l - 125.0) * 0.012 + highlight_ratio * 1.5, 1.35, 2.30))
            lut = self._get_gamma_table(gamma_power)
            l_darkened = cv2.LUT(l_channel, lut)

            clahe = self._get_clahe(2.0)
            l_enhanced = clahe.apply(l_darkened)

            lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR), mean_l, "HIGH_LIGHT"

        return frame_bgr, mean_l, "OPTIMAL"


LowLightEnhancer = AdaptiveLightingEnhancer


# =====================================================================
# 2. KEYPOINT FILTERING & INTERPOLATION (OCCLUSION & > 1.5M DISTANCE)
# =====================================================================

class KeypointInterpolator:
    def __init__(self, alpha: float = 0.65, max_missing_frames: int = 5):
        self.alpha = alpha
        self.max_missing_frames = max_missing_frames
        self.history: Dict[str, collections.deque] = {
            "Right": collections.deque(maxlen=10),
            "Left": collections.deque(maxlen=10),
        }
        self.consecutive_missing: Dict[str, int] = {"Right": 0, "Left": 0}
        self.last_valid_coords: Dict[str, Optional[np.ndarray]] = {"Right": None, "Left": None}
        self.last_valid_velocities: Dict[str, np.ndarray] = {
            "Right": np.zeros((21, 3), dtype=np.float32),
            "Left": np.zeros((21, 3), dtype=np.float32),
        }

    def update(
        self,
        hand_landmarks,
        hand_label: str = "Right",
        min_presence_conf: float = 0.45
    ) -> Tuple[Optional[np.ndarray], bool, Dict]:
        metrics = {
            "hand_scale": 0.0,
            "is_far": False,
            "occluded_joints": 0,
            "consecutive_missing": self.consecutive_missing[hand_label],
        }

        if hand_landmarks is None:
            self.consecutive_missing[hand_label] += 1
            if (
                self.consecutive_missing[hand_label] > self.max_missing_frames
                or self.last_valid_coords[hand_label] is None
            ):
                self.last_valid_coords[hand_label] = None
                return None, False, metrics

            decay = 0.85 ** self.consecutive_missing[hand_label]
            predicted_coords = (
                self.last_valid_coords[hand_label]
                + self.last_valid_velocities[hand_label] * decay
            )
            return predicted_coords, True, metrics

        raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        palm_vector = raw_coords[9] - raw_coords[0]
        hand_scale = float(np.linalg.norm(palm_vector[:2]))
        metrics["hand_scale"] = hand_scale
        is_far = hand_scale < 0.080
        metrics["is_far"] = is_far

        occluded_count = 0
        cleaned_coords = raw_coords.copy()

        if self.last_valid_coords[hand_label] is not None:
            prev_coords = self.last_valid_coords[hand_label]
            eff_alpha = self.alpha * 0.6 if is_far else self.alpha

            for i in range(21):
                delta = float(np.linalg.norm(raw_coords[i] - prev_coords[i]))
                if delta > max(0.18, hand_scale * 2.5):
                    cleaned_coords[i] = prev_coords[i] + self.last_valid_velocities[hand_label][i] * 0.5
                    occluded_count += 1
                else:
                    cleaned_coords[i] = eff_alpha * raw_coords[i] + (1.0 - eff_alpha) * prev_coords[i]

            current_vel = cleaned_coords - prev_coords
            self.last_valid_velocities[hand_label] = (
                0.7 * current_vel + 0.3 * self.last_valid_velocities[hand_label]
            )
        else:
            self.last_valid_velocities[hand_label] = np.zeros((21, 3), dtype=np.float32)

        metrics["occluded_joints"] = occluded_count
        self.last_valid_coords[hand_label] = cleaned_coords
        self.consecutive_missing[hand_label] = 0

        return cleaned_coords, (occluded_count > 0 or is_far), metrics

    def reset(self, hand_label: Optional[str] = None):
        labels = [hand_label] if hand_label else ["Right", "Left"]
        for lbl in labels:
            self.consecutive_missing[lbl] = 0
            self.last_valid_coords[lbl] = None
            self.last_valid_velocities[lbl] = np.zeros((21, 3), dtype=np.float32)
            self.history[lbl].clear()


# =====================================================================
# 3. MOTION-BASED ROUTING (STATIC VS. DYNAMIC GESTURE)
# =====================================================================

class MotionBasedRouter:
    def __init__(
        self,
        window_size: int = 5,
        motion_threshold: float = 0.055,
        stationary_threshold: float = 0.022,
    ):
        self.window_size = window_size
        self.motion_threshold = motion_threshold
        self.stationary_threshold = stationary_threshold
        self.pos_history = collections.deque(maxlen=window_size)
        self.current_mode = "STATIC"
        self.motion_magnitude = 0.0

    def update(self, hand_coords: Optional[np.ndarray], hand_scale: float = 0.15) -> Tuple[str, float]:
        if hand_coords is None:
            self.pos_history.clear()
            self.motion_magnitude = 0.0
            return "STATIC", 0.0

        centroid = np.mean(hand_coords[[0, 5, 9], :2], axis=0)
        self.pos_history.append(centroid)

        if len(self.pos_history) < 2:
            return self.current_mode, 0.0

        displacements = []
        for i in range(1, len(self.pos_history)):
            disp = float(np.linalg.norm(self.pos_history[i] - self.pos_history[i - 1]))
            scale = max(hand_scale, 0.05)
            displacements.append(disp / scale)

        self.motion_magnitude = float(np.mean(displacements))

        if self.motion_magnitude >= self.motion_threshold:
            self.current_mode = "DYNAMIC"
        elif self.motion_magnitude <= self.stationary_threshold:
            self.current_mode = "STATIC"

        return self.current_mode, self.motion_magnitude

    def reset(self):
        self.pos_history.clear()
        self.current_mode = "STATIC"
        self.motion_magnitude = 0.0


# =====================================================================
# 4. SOFTMAX MARGIN & AMBIGUITY REJECTION VERIFIER
# =====================================================================

class SoftmaxMarginVerifier:
    def __init__(self, min_confidence: float = 0.55, min_margin: float = 0.10, verbose: bool = False):
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.verbose = verbose

    def verify(
        self,
        probs: np.ndarray,
        label_classes: np.ndarray,
        min_confidence: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> Tuple[Optional[str], float, str, Dict]:
        req_conf = min_confidence if min_confidence is not None else self.min_confidence
        req_margin = min_margin if min_margin is not None else self.min_margin

        flat_probs = np.asarray(probs, dtype=np.float32).flatten()
        if len(flat_probs) == 0:
            return None, 0.0, "low_confidence", {
                "top_label": "—", "second_label": "—", "margin": 0.0,
                "p_top": 0.0, "p_second": 0.0
            }

        sorted_indices = np.argsort(flat_probs)[::-1]
        top_idx = int(sorted_indices[0])
        second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_idx

        p_top = float(flat_probs[top_idx])
        p_second = float(flat_probs[second_idx]) if len(sorted_indices) > 1 else 0.0
        margin = float(p_top - p_second)

        top_label = str(label_classes[top_idx]) if label_classes is not None and top_idx < len(label_classes) else "—"
        second_label = str(label_classes[second_idx]) if label_classes is not None and second_idx < len(label_classes) and len(sorted_indices) > 1 else "—"

        telemetry = {
            "top_label": top_label,
            "second_label": second_label,
            "margin": margin,
            "p_top": p_top,
            "p_second": p_second,
        }

        if p_top < req_conf:
            return None, p_top, "low_confidence", telemetry

        if margin < req_margin:
            return None, p_top, "ambiguous", telemetry

        return top_label, p_top, "valid", telemetry


# =====================================================================
# PART 3: SOFT COOPERATIVE ENSEMBLE ROUTER (`ASLOEnsembleRouter`)
# =====================================================================

class _SavedScaler:
    """Lightweight fast numpy StandardScaler loader (`(x - mean) / scale`)."""
    def __init__(self, npz_path: str, expected_dim: int):
        data = np.load(npz_path)
        self.mean = np.asarray(data["mean"], dtype=np.float32).reshape(1, expected_dim)
        self.scale = np.asarray(data["scale"], dtype=np.float32).reshape(1, expected_dim)
        self.scale = np.where(np.abs(self.scale) < 1e-7, 1.0, self.scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        return np.clip((arr - self.mean) / self.scale, -5.0, 5.0).astype(np.float32)


class ASLOEnsembleRouter:
    """
    100% Stateless Soft Cooperative Multi-MLP Ensemble Router.
    Coordinates:
    1. Model Group 1 (Static ASL Cooperative Two-Stage MLPs):
       - Stage-1 Router (`asl_stage1_router_mlp.keras`) -> outputs $P_1(c)$ over shape families & distinct singletons.
       - Stage-2 Specialists (`asl_specialist_<cluster>_mlp.keras`) -> outputs $P_2(s \mid c)$ within each family.
       - Soft Probabilistic Fusion: evaluates the top-2 Stage-1 candidates and computes
         $P_{\text{final}}(s) = P_1(c) \times P_2(s \mid c)$ for clusters or $P_1(c)$ for singletons,
         merging all sign probabilities into a unified ranking before margin gating (`p_top >= 0.55`, `margin >= 0.10`).
    2. Model Group 2 (`words_static_mlp.keras`): Dedicated Static Words & Control Signs MLP.
    3. Model Group 3 (`dynamic_mlp.keras`): Dedicated 25-frame Dynamic Gestures MLP (`p_top >= 0.68`).
    All Keras models are wrapped with `@tf.function(reduce_retracing=True)` for zero-latency inference.
    """
    MOTION_THRESHOLD = 0.035
    DYNAMIC_MIN_CONFIDENCE = 0.68
    STATIC_MIN_CONFIDENCE = 0.55
    STATIC_MIN_MARGIN = 0.10

    SPECIALIST_DISPLAY_NAMES = {
        "CLUSTER_FIST": "Fist Specialist",
        "CLUSTER_UPRIGHT": "Upright Specialist",
        "CLUSTER_CURVE_PINCH": "Curve/Pinch Specialist",
        "CLUSTER_OPEN": "Open-Hand Specialist",
    }

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.verifier = SoftmaxMarginVerifier(
            min_confidence=self.STATIC_MIN_CONFIDENCE,
            min_margin=self.STATIC_MIN_MARGIN,
        )

        self.registry: Dict = {}
        self.shape_families: Dict[str, List[str]] = {}

        # Stage-1 Router
        self.asl_stage1_model = None
        self.asl_stage1_labels = None
        self.asl_stage1_scaler: Optional[_SavedScaler] = None
        self._fast_asl_stage1 = None

        # Stage-2 Specialists keyed by cluster name (e.g. "CLUSTER_FIST")
        self.specialists: Dict[str, Dict] = {}
        self.asl_specialists: Dict[str, Dict] = self.specialists

        # Backward compatibility attributes for direct fist model access
        self.asl_fist_model = None
        self.asl_fist_labels = None
        self._fast_asl_fist = None

        # Model Group 2: Static Words
        self.words_model = None
        self.words_labels = None
        self.words_scaler: Optional[_SavedScaler] = None
        self._fast_words = None

        # Model Group 3A: Dynamic Gestures (25-Frame Trajectory)
        self.dynamic_model = None
        self.dynamic_labels = None
        self.dynamic_scaler: Optional[_SavedScaler] = None
        self._fast_dynamic = None

        # Model Group 3B: Dynamic Keyframes (_START / _END)
        self.dynamic_keyframe_model = None
        self.dynamic_keyframe_labels = None
        self.dynamic_keyframe_scaler: Optional[_SavedScaler] = None
        self._fast_dynamic_keyframe = None

        # Merged Meta-Router MLP (routes static_asl vs static_words vs dynamic_keyframe with OOD cosine similarity)
        self.meta_router_model = None
        self.meta_router_labels = None
        self.meta_router_scaler: Optional[_SavedScaler] = None
        self._fast_meta_router = None
        self._fast_asl_embed = None
        self._fast_words_embed = None
        self._fast_kf_embed = None
        self.asl_centroids: Optional[np.ndarray] = None
        self.words_centroids: Optional[np.ndarray] = None
        self.kf_centroids: Optional[np.ndarray] = None
        self.EXPECTED_META_DIM = aslo_features.EXPECTED_WORD_DIM + 1 + 12  # 265

        self._loaded_mtime = 0.0
        self.is_ready = False
        self.load_all_models()

    @staticmethod
    def _wrap_tf_predict(keras_model, input_dim: int):
        @tf.function(reduce_retracing=True)
        def _fn(x):
            return keras_model(x, training=False)
        # Warmup graph immediately
        _fn(tf.zeros((1, input_dim), dtype=tf.float32))
        return _fn

    def load_all_models(self) -> bool:
        """Loads registry, all Keras MLP models, label encoders, and StandardScalers."""
        reg_path = os.path.join(self.models_dir, "gesture_registry.json")
        if not os.path.exists(reg_path):
            print(f"[ASLOEnsembleRouter] Registry '{reg_path}' not found.")
            self.is_ready = False
            return False

        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                self.registry = json.load(f)
            self.shape_families = self.registry.get("shape_families", {})

            # 1. Load Stage-1 ASL Router
            s1_m_path = os.path.join(self.models_dir, "asl_stage1_router_mlp.keras")
            s1_l_path = os.path.join(self.models_dir, "asl_stage1_router_labels.npy")
            s1_s_path = os.path.join(self.models_dir, "asl_stage1_router_scaler.npz")
            if not (os.path.exists(s1_m_path) and os.path.exists(s1_l_path) and os.path.exists(s1_s_path)):
                self.is_ready = False
                return False

            self.asl_stage1_model = models.load_model(s1_m_path)
            self.asl_stage1_labels = np.load(s1_l_path, allow_pickle=True)
            self.asl_stage1_scaler = _SavedScaler(s1_s_path, aslo_features.EXPECTED_ASL_DIM)
            assert self.asl_stage1_model.input_shape[-1] == aslo_features.EXPECTED_ASL_DIM
            self._fast_asl_stage1 = self._wrap_tf_predict(self.asl_stage1_model, aslo_features.EXPECTED_ASL_DIM)

            # 2. Load Stage-2 Specialists for each active shape family
            self.specialists.clear()
            for cluster_name in self.shape_families.keys():
                short_key = cluster_name.lower()
                prefix = os.path.join(self.models_dir, f"asl_specialist_{short_key}")
                cm_path = f"{prefix}_mlp.keras"
                cl_path = f"{prefix}_labels.npy"
                cs_path = f"{prefix}_scaler.npz"
                if not (os.path.exists(cm_path) and os.path.exists(cl_path) and os.path.exists(cs_path)):
                    print(f"[ASLOEnsembleRouter] Missing specialist files for {cluster_name}")
                    self.is_ready = False
                    return False

                c_model = models.load_model(cm_path)
                c_labels = np.load(cl_path, allow_pickle=True)
                c_scaler = _SavedScaler(cs_path, aslo_features.EXPECTED_ASL_DIM)
                assert c_model.input_shape[-1] == aslo_features.EXPECTED_ASL_DIM
                c_fast = self._wrap_tf_predict(c_model, aslo_features.EXPECTED_ASL_DIM)

                self.specialists[cluster_name] = {
                    "model": c_model,
                    "labels": c_labels,
                    "scaler": c_scaler,
                    "fast_fn": c_fast,
                    "display_name": self.SPECIALIST_DISPLAY_NAMES.get(cluster_name, cluster_name),
                }

                if cluster_name == "CLUSTER_FIST":
                    self.asl_fist_model = c_model
                    self.asl_fist_labels = c_labels
                    self._fast_asl_fist = c_fast

            # 3. Load Static Words MLP
            w_m_path = os.path.join(self.models_dir, "words_static_mlp.keras")
            w_l_path = os.path.join(self.models_dir, "words_static_labels.npy")
            if not os.path.exists(w_l_path):
                w_l_path = os.path.join(self.models_dir, "words_labels.npy")
            w_s_path = os.path.join(self.models_dir, "words_static_scaler.npz")
            if not os.path.exists(w_s_path):
                w_s_path = os.path.join(self.models_dir, "words_scaler.npz")

            if not (os.path.exists(w_m_path) and os.path.exists(w_l_path) and os.path.exists(w_s_path)):
                self.is_ready = False
                return False

            self.words_model = models.load_model(w_m_path)
            self.words_labels = np.load(w_l_path, allow_pickle=True)
            self.words_scaler = _SavedScaler(w_s_path, aslo_features.EXPECTED_WORD_DIM)
            assert self.words_model.input_shape[-1] == aslo_features.EXPECTED_WORD_DIM
            self._fast_words = self._wrap_tf_predict(self.words_model, aslo_features.EXPECTED_WORD_DIM)

            # 4. Load Dynamic Gestures 25-Frame MLP
            d_m_path = os.path.join(self.models_dir, "dynamic_mlp.keras")
            d_l_path = os.path.join(self.models_dir, "dynamic_labels.npy")
            d_s_path = os.path.join(self.models_dir, "dynamic_scaler.npz")
            if not (os.path.exists(d_m_path) and os.path.exists(d_l_path) and os.path.exists(d_s_path)):
                self.is_ready = False
                return False

            self.dynamic_model = models.load_model(d_m_path)
            self.dynamic_labels = np.load(d_l_path, allow_pickle=True)
            self.dynamic_scaler = _SavedScaler(d_s_path, aslo_features.EXPECTED_DYNAMIC_DIM)
            assert self.dynamic_model.input_shape[-1] == aslo_features.EXPECTED_DYNAMIC_DIM
            self._fast_dynamic = self._wrap_tf_predict(self.dynamic_model, aslo_features.EXPECTED_DYNAMIC_DIM)

            # 5. Load Model Group 3B: Dynamic Keyframes (_START / _END) MLP
            kf_m_path = os.path.join(self.models_dir, "dynamic_keyframe_mlp.keras")
            kf_l_path = os.path.join(self.models_dir, "dynamic_keyframe_labels.npy")
            kf_s_path = os.path.join(self.models_dir, "dynamic_keyframe_scaler.npz")
            mtimes = [os.path.getmtime(s1_m_path), os.path.getmtime(w_m_path), os.path.getmtime(d_m_path)]
            if os.path.exists(kf_m_path) and os.path.exists(kf_l_path) and os.path.exists(kf_s_path):
                self.dynamic_keyframe_model = models.load_model(kf_m_path)
                self.dynamic_keyframe_labels = np.load(kf_l_path, allow_pickle=True)
                self.dynamic_keyframe_scaler = _SavedScaler(kf_s_path, aslo_features.EXPECTED_WORD_DIM)
                assert self.dynamic_keyframe_model.input_shape[-1] == aslo_features.EXPECTED_WORD_DIM
                self._fast_dynamic_keyframe = self._wrap_tf_predict(
                    self.dynamic_keyframe_model, aslo_features.EXPECTED_WORD_DIM
                )
                mtimes.append(os.path.getmtime(kf_m_path))
            else:
                self.dynamic_keyframe_model = None
                self.dynamic_keyframe_labels = None
                self.dynamic_keyframe_scaler = None
                self._fast_dynamic_keyframe = None

            # 6. Load Merged OOD-Aware Meta-Router MLP (265 dims) & Penultimate Embedding Centroids
            mr_m_path = os.path.join(self.models_dir, "meta_router_mlp.keras")
            mr_l_path = os.path.join(self.models_dir, "meta_router_labels.npy")
            mr_s_path = os.path.join(self.models_dir, "meta_router_scaler.npz")
            cent_path = os.path.join(self.models_dir, "meta_centroids.npz")
            if (
                os.path.exists(mr_m_path)
                and os.path.exists(mr_l_path)
                and os.path.exists(mr_s_path)
                and os.path.exists(cent_path)
            ):
                self.meta_router_model = models.load_model(mr_m_path)
                self.meta_router_labels = np.load(mr_l_path, allow_pickle=True)
                self.meta_router_scaler = _SavedScaler(mr_s_path, self.EXPECTED_META_DIM)
                assert self.meta_router_model.input_shape[-1] == self.EXPECTED_META_DIM
                self._fast_meta_router = self._wrap_tf_predict(
                    self.meta_router_model, self.EXPECTED_META_DIM
                )
                c_data = np.load(cent_path)
                self.asl_centroids = np.asarray(c_data["asl_centroids"], dtype=np.float32)
                self.words_centroids = np.asarray(c_data["words_centroids"], dtype=np.float32)
                self.kf_centroids = np.asarray(c_data["kf_centroids"], dtype=np.float32)

                asl_emb_m = models.Model(self.asl_stage1_model.input, self.asl_stage1_model.layers[-2].output)
                w_emb_m = models.Model(self.words_model.input, self.words_model.layers[-2].output)
                self._fast_asl_embed = self._wrap_tf_predict(asl_emb_m, aslo_features.EXPECTED_ASL_DIM)
                self._fast_words_embed = self._wrap_tf_predict(w_emb_m, aslo_features.EXPECTED_WORD_DIM)
                if self.dynamic_keyframe_model is not None:
                    kf_emb_m = models.Model(self.dynamic_keyframe_model.input, self.dynamic_keyframe_model.layers[-2].output)
                    self._fast_kf_embed = self._wrap_tf_predict(kf_emb_m, aslo_features.EXPECTED_WORD_DIM)
                else:
                    self._fast_kf_embed = None
                mtimes.append(os.path.getmtime(mr_m_path))
            else:
                self.meta_router_model = None
                self.meta_router_labels = None
                self.meta_router_scaler = None
                self._fast_meta_router = None
                self._fast_asl_embed = None
                self._fast_words_embed = None
                self._fast_kf_embed = None
                self.asl_centroids = None
                self.words_centroids = None
                self.kf_centroids = None

            self._loaded_mtime = max(mtimes)
            self.is_ready = True
            print(
                f"[ASLOEnsembleRouter] Ready! "
                f"Stage-1 ASL ({len(self.asl_stage1_labels)} routes), "
                f"Stage-2 Specialists ({list(self.specialists.keys())}), "
                f"Static Words ({len(self.words_labels)}), "
                f"Dynamic ({len(self.dynamic_labels)}), "
                f"Dynamic Keyframes ({len(self.dynamic_keyframe_labels) if self.dynamic_keyframe_labels is not None else 0}), "
                f"Meta-Router ({list(self.meta_router_labels) if self.meta_router_labels is not None else 'disabled'})"
            )
            return True
        except Exception as e:
            print(f"[ASLOEnsembleRouter] Failed to load models: {e}")
            self.is_ready = False
            return False

    def fuse_stage1_and_stage2_probabilities(
        self,
        asl_features_1d: np.ndarray,
        override_stage1_probs: Optional[Dict[str, float]] = None,
        word_features_1d: Optional[np.ndarray] = None,
        allow_word_specialist: bool = False,
    ) -> Dict:
        """
        PART 3.1: Soft Probabilistic Two-Stage Fusion (with Cooperative 1-Hand Word Specialist Routing).
        1. Runs `asl_stage1_router_mlp` to get Stage-1 probabilities $P_1(c)$ (or uses `override_stage1_probs` for testing split-confidence fusion).
        2. Selects the **top 2** Stage-1 candidates $(c_1, c_2)$.
        3. For each candidate $c \in \{c_1, c_2\}$:
           - If $c$ is a single distinct letter, its final score is $P_{\text{final}}(c) = P_1(c)$.
           - If $c$ is a shape-family cluster (e.g., `CLUSTER_FIST`, `CLUSTER_UPRIGHT`), runs its matching
             Stage-2 Specialist MLP to get $P_2(s \mid c)$ and computes the joint probability for every sign $s$ in that family:
               $$P_{\text{final}}(s) = P_1(c) \times P_2(s \mid c)$$
           - If $c == \text{"CLUSTER\_WORD\_1H"}$ (in unified auto mode), runs `words_static_mlp` on `word_features_1d`,
             filters to 1-hand static words (`Fine`, `I love you`, `Thank you`, `YOU`, `hello`), and computes:
               $$P_{\text{final}}(w) = P_1(\text{CLUSTER\_WORD\_1H}) \times P_{\text{words}}(w \mid \text{1-hand})$$
        4. Merges all resulting sign probabilities into one unified ranking, picks `top_sign` and `second_sign`,
           and applies margin gating (`p_top >= 0.55` and `p_top - p_second >= 0.10`).
        """
        raw_asl = aslo_features.validate_asl_feature_vector(asl_features_1d)

        if override_stage1_probs is not None:
            s1_probs = np.array(
                [float(override_stage1_probs.get(str(lbl), 0.0)) for lbl in self.asl_stage1_labels],
                dtype=np.float32,
            )
        else:
            s1_scaled = self.asl_stage1_scaler.transform(raw_asl)
            s1_probs = self._fast_asl_stage1(tf.constant(s1_scaled, dtype=tf.float32)).numpy()[0].copy()

        # If ASL-only mode is requested (`allow_word_specialist=False`), zero out CLUSTER_WORD_1H and renormalize
        if not allow_word_specialist:
            for i_lbl, lbl_name in enumerate(self.asl_stage1_labels):
                if str(lbl_name) == "CLUSTER_WORD_1H":
                    s1_probs[i_lbl] = 0.0
            s1_sum = float(np.sum(s1_probs))
            if s1_sum > 1e-6:
                s1_probs = (s1_probs / s1_sum).astype(np.float32)

        # Pick top 3 Stage-1 candidates so CLUSTER_WORD_1H and the top 2 ASL clusters are always evaluated together
        sorted_s1_idxs = np.argsort(s1_probs)[::-1]
        top_candidates = list(sorted_s1_idxs[:2])
        if allow_word_specialist:
            for i_lbl, lbl_name in enumerate(self.asl_stage1_labels):
                if str(lbl_name) == "CLUSTER_WORD_1H" and i_lbl not in top_candidates:
                    top_candidates.append(i_lbl)

        final_scores: Dict[str, float] = {}
        sign_p2_map: Dict[str, float] = {}
        sign_specialist_map: Dict[str, str] = {}
        evaluated_specialists: List[str] = []
        ONE_HAND_WORDS_SET = {
            "fine", "i love you", "love", "thank you", "you",
            "hello", "neutral", "start", "stop", "space", "backspace", "back space",
            "j_start", "j_end", "z_start", "z_end"
        }

        for idx in top_candidates:
            c_label = str(self.asl_stage1_labels[int(idx)])
            p1_c = float(s1_probs[int(idx)])

            if c_label == "CLUSTER_WORD_1H" and allow_word_specialist and self._fast_words is not None:
                evaluated_specialists.append("CLUSTER_WORD_1H")
                if word_features_1d is None:
                    raw_a = raw_asl.flatten()
                    w_vec = np.concatenate([raw_a, np.zeros(aslo_features.EXPECTED_ASL_DIM + 4, dtype=np.float32)])
                else:
                    w_vec = word_features_1d
                raw_w = aslo_features.validate_word_feature_vector(w_vec)
                w_scaled = self.words_scaler.transform(raw_w)
                w_probs_raw = self._fast_words(tf.constant(w_scaled, dtype=tf.float32)).numpy()[0].copy()

        # Probability assigned by words_static_mlp to the negative 'ASL_LETTER' class
                p_asl_letter_in_words = 0.0
                for w_i, w_name in enumerate(self.words_labels):
                    w_up = str(w_name).strip().upper()
                    if w_up == "ASL_LETTER":
                        p_asl_letter_in_words = float(w_probs_raw[w_i])
                        w_probs_raw[w_i] = 0.0

                w_probs_cond = w_probs_raw.copy()
                for w_i, w_name in enumerate(self.words_labels):
                    if str(w_name).strip().lower() not in ONE_HAND_WORDS_SET:
                        w_probs_cond[w_i] = 0.0
                w_sum = float(np.sum(w_probs_cond))
                if w_sum > 1e-6:
                    w_probs_cond = w_probs_cond / w_sum

                # Cooperative probability: only boost CLUSTER_WORD_1H when Stage-1 already gives p1_c >= 0.22
                # and Model 2 strongly rejects ASL_LETTER (p_asl_letter_in_words < 0.20)
                if p1_c >= 0.22 and p_asl_letter_in_words < 0.20:
                    p_word_gate = max(p1_c, 0.60 * p1_c + 0.40 * (1.0 - p_asl_letter_in_words))
                else:
                    p_word_gate = p1_c

                for w_i, w_name in enumerate(self.words_labels):
                    w_str = str(w_name)
                    if w_str.strip().lower() not in ONE_HAND_WORDS_SET:
                        continue
                    p2_val = float(w_probs_cond[w_i])
                    direct_w_val = float(w_probs_raw[w_i])
                    joint_p = max(p1_c * p2_val, p_word_gate * direct_w_val)
                    if joint_p > final_scores.get(w_str, -1.0):
                        final_scores[w_str] = joint_p
                        sign_p2_map[w_str] = max(p2_val, direct_w_val)
                        sign_specialist_map[w_str] = "Static Words MLP"
            elif c_label in self.specialists:
                if p1_c <= 1e-6:
                    continue
                spec = self.specialists[c_label]
                evaluated_specialists.append(c_label)
                c_scaled = spec["scaler"].transform(raw_asl)
                p2_probs = spec["fast_fn"](tf.constant(c_scaled, dtype=tf.float32)).numpy()[0]
                for s_idx, s_name in enumerate(spec["labels"]):
                    s_str = str(s_name)
                    p2_val = float(p2_probs[s_idx])
                    joint_p = p1_c * p2_val
                    if joint_p > final_scores.get(s_str, -1.0):
                        final_scores[s_str] = joint_p
                        sign_p2_map[s_str] = p2_val
                        sign_specialist_map[s_str] = spec["display_name"]
            elif c_label != "CLUSTER_WORD_1H":
                if p1_c <= 1e-6:
                    continue
                if p1_c > final_scores.get(c_label, -1.0):
                    final_scores[c_label] = p1_c
                    sign_p2_map[c_label] = p1_c
                    sign_specialist_map[c_label] = "Stage-1 Direct"

        if not final_scores:
            return {
                "prediction": None,
                "top_label": "—",
                "second_label": "—",
                "confidence": 0.0,
                "status": "low_confidence",
                "telemetry": {"top_label": "—", "second_label": "—", "margin": 0.0, "p_top": 0.0, "p_second": 0.0},
                "gesture_type": "static_asl",
                "model_source": "asl_stage1_router_mlp",
                "specialist_used": "None",
                "routed_via_fist_specialist": False,
            }

        # Camera-frame orientation disambiguation for P vs Q vs X (prevents cross-override under webcam tilt):
        # asl_features[115:118] = thumb tip(4), [118:121] = index tip(8), [121:124] = middle MCP(9), [70] = mid PIP angle
        if override_stage1_probs is None and final_scores:
            y4_cam = float(raw_asl[0, 116])
            y8_cam = float(raw_asl[0, 119])
            mid_pip_ang = float(raw_asl[0, 70])
            if y8_cam > 0.08 and y4_cam > 0.28 and mid_pip_ang < 2.05:
                # Clearly downward-pointing Q (both index & thumb pointing down, middle curled)
                for bad_lbl in ("X", "P"):
                    if bad_lbl in final_scores:
                        final_scores[bad_lbl] *= 0.08
            elif y8_cam < -0.92 and y4_cam < -0.15 and mid_pip_ang < 1.65:
                # Clearly upright hooked X (index up, middle tightly curled)
                for bad_lbl in ("Q", "P"):
                    if bad_lbl in final_scores:
                        final_scores[bad_lbl] *= 0.08
            elif -1.05 < y8_cam < 0.05 and y4_cam < 0.18 and mid_pip_ang >= 1.85:
                # Clearly forward/downward P (middle extended, thumb not pointing straight down)
                for bad_lbl in ("Q", "X"):
                    if bad_lbl in final_scores:
                        final_scores[bad_lbl] *= 0.08

            # 3D Rotation-Invariant U vs V vs R separation using canonical palm-frame spread/crossing invariants:
            # raw_asl[0, 89] = lat_tip_ratio = (x8 - x12)/(x5 - x9)
            # raw_asl[0, 91] = spread_tip_mcp_ratio = ||P8 - P12|| / ||P5 - P9||
            top_prelim = max(final_scores, key=final_scores.get)
            if top_prelim in ("U", "V", "R"):
                lat_tip = float(raw_asl[0, 89])
                spread_r = float(raw_asl[0, 91])
                if lat_tip < 0.42:
                    for bad_lbl in ("U", "V"):
                        if bad_lbl in final_scores:
                            final_scores[bad_lbl] *= 0.15
                elif lat_tip > 1.48 and spread_r > 1.50:
                    for bad_lbl in ("U", "R"):
                        if bad_lbl in final_scores:
                            final_scores[bad_lbl] *= 0.15
                elif 0.55 <= lat_tip <= 1.28 and spread_r <= 1.38:
                    for bad_lbl in ("V", "R"):
                        if bad_lbl in final_scores:
                            final_scores[bad_lbl] *= 0.15

        ranked = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_sign, p_top = ranked[0][0], float(ranked[0][1])
        second_sign, p_second = (ranked[1][0], float(ranked[1][1])) if len(ranked) > 1 else (top_sign, 0.0)
        margin = float(p_top - p_second)
        top_p2 = float(sign_p2_map.get(top_sign, p_top))

        # Cooperative confidence when Stage-1 probability is split (e.g. 55%/45%) but Stage-2 Specialist is decisive
        effective_p_top = p_top
        if p_top < self.STATIC_MIN_CONFIDENCE and p_top >= 0.40 and top_p2 >= 0.80 and margin >= self.STATIC_MIN_MARGIN:
            effective_p_top = float(np.sqrt(p_top * top_p2))

        telemetry = {
            "top_label": top_sign,
            "second_label": second_sign,
            "margin": margin,
            "p_top": effective_p_top,
            "p_joint_raw": p_top,
            "p_stage2": top_p2,
            "p_second": p_second,
            "stage1_top2": [
                (str(self.asl_stage1_labels[int(i)]), float(s1_probs[int(i)])) for i in sorted_s1_idxs[:2]
            ],
            "evaluated_specialists": evaluated_specialists,
        }

        specialist_used = sign_specialist_map.get(top_sign, "Stage-1 Direct")
        is_fist_specialist = (specialist_used == "Fist Specialist")
        is_word_specialist = (specialist_used == "Static Words MLP")
        if is_word_specialist:
            gesture_type = "static_words"
            model_source = "words_static_mlp"
        else:
            gesture_type = "static_asl"
            model_source = (
                f"asl_specialist_{self.registry.get('sign_to_family', {}).get(top_sign, '').lower()}_mlp"
                if top_sign in self.registry.get("sign_to_family", {})
                else "asl_stage1_router_mlp"
            )

        if effective_p_top < self.STATIC_MIN_CONFIDENCE:
            status = "low_confidence"
            pred = None
        elif margin < self.STATIC_MIN_MARGIN:
            status = "ambiguous"
            pred = None
        else:
            status = "valid"
            pred = top_sign

        return {
            "prediction": pred,
            "top_label": top_sign,
            "second_label": second_sign,
            "confidence": effective_p_top,
            "status": status,
            "telemetry": telemetry,
            "gesture_type": gesture_type,
            "model_source": model_source,
            "specialist_used": specialist_used,
            "routed_via_fist_specialist": is_fist_specialist,
        }

    def predict_static_words(
        self,
        word_features_1d: np.ndarray,
        mask_asl_letter: bool = True,
        mask_dynamic_poses: bool = False,
        force_two_hand: bool = False,
    ) -> Dict:
        """
        Runs Model Group 2 (`words_static_mlp.keras`) with its fitted `StandardScaler`.
        Enforces strict 1-hand vs 2-hand structural compatibility and 2D orientation invariants:
        - When 2 hands are present (`has_second_hand == True`), masks out strictly 1-hand words
          (`YOU`, `Fine`, `I love you`, `Thank you`, `hello`) so 2-hand signs (`STOP`, `START`,
          `SPACE`, `BACKSPACE`, `How are you`, `Nice to meet you`) are NEVER overridden by `YOU`.
        - When only 1 hand is present (`has_second_hand == False`), masks out strictly 2-hand words
          (`ME`, `Love`, `start`, `stop`, `space`, `backspace`, `neutral`) so 1-hand ASL letters are NEVER overridden.
        """
        raw_w = aslo_features.validate_word_feature_vector(word_features_1d)
        w_scaled = self.words_scaler.transform(raw_w)
        probs_w = self._fast_words(tf.constant(w_scaled, dtype=tf.float32)).numpy()[0].copy()

        has_first_hand = bool(np.any(np.abs(raw_w[0, 0:124]) > 1e-4))
        has_second_hand = bool(np.any(np.abs(raw_w[0, 124:248]) > 1e-4)) or force_two_hand
        STRICT_ONE_HAND_WORDS = {"you", "fine", "i love you", "thank you", "hello", "j_start", "j_end", "z_start", "z_end"}
        STRICT_TWO_HAND_WORDS = {
            "me", "love", "start", "stop", "space", "backspace", "back space", "neutral",
            "how are you_start", "how are you_end", "nice to meet you_start", "nice to meet you_end",
        }

        for i_w, w_lbl in enumerate(self.words_labels):
            w_clean = str(w_lbl).strip().lower()
            if mask_asl_letter and w_clean == "asl_letter":
                probs_w[i_w] = 0.0
            elif mask_dynamic_poses and (w_clean.endswith("_start") or w_clean.endswith("_end")):
                probs_w[i_w] = 0.0
            elif has_first_hand and has_second_hand and w_clean in STRICT_ONE_HAND_WORDS:
                probs_w[i_w] = 0.0
            elif (not has_second_hand) and w_clean in STRICT_TWO_HAND_WORDS:
                probs_w[i_w] = 0.0

        # 2D Foreshortening- & Z-depth-invariant disambiguation using orient_feats:
        # [u_t_2d_x(115), u_t_2d_y(116), dy_4_8(117), u42_x(118), u42_y(119), dy_4_5(120), u_i_2d_x(121), u_i_2d_y(122), dy_8_5(123)]
        if has_first_hand:
            o0 = raw_w[0, 115:124]
            o1 = raw_w[0, 239:248] if has_second_hand else o0

            # 1. Both hands (or single hand) pointing index finger(s) DOWNWARD (STOP):
            #    In STOP, BOTH hands point downward (o0[7] > 0.45 and o1[7] > 0.45), whereas in How are you_END / Nice to meet you_END
            #    the right hand points UP (o0[7] < -0.35) while only the left hand is lower!
            both_pointing_down = (float(o0[7]) > 0.45 and float(o1[7]) > 0.45 and float(o0[8]) > 0.20 and float(o1[8]) > 0.20)
            one_pointing_up = (min(float(o0[7]), float(o1[7])) < -0.35)
            if both_pointing_down:
                for i_w, w_lbl in enumerate(self.words_labels):
                    if str(w_lbl).strip().lower() in ("you", "me", "how are you_start", "how are you_end", "nice to meet you_start", "nice to meet you_end"):
                        probs_w[i_w] = 0.0
            elif one_pointing_up and has_second_hand:
                for i_w, w_lbl in enumerate(self.words_labels):
                    if str(w_lbl).strip().lower() == "stop":
                        probs_w[i_w] *= 0.05

            # 2. Both thumbs pointing vertically UP above closed fists (START: 👍 👍):
            #    In START, BOTH thumbs point vertically UP (u42_y < -0.78, |u42_x| < 0.48, dy_4_8 < -0.30),
            #    whereas in How are you_START the right thumb is horizontal (u_t_y ~ +0.07)!
            both_thumbs_up = (
                float(o0[4]) < -0.78 and float(o1[4]) < -0.78
                and abs(float(o0[3])) < 0.48 and abs(float(o1[3])) < 0.48
                and float(o0[2]) < -0.30 and float(o1[2]) < -0.30
            )
            if both_thumbs_up:
                for i_w, w_lbl in enumerate(self.words_labels):
                    w_low = str(w_lbl).strip().lower()
                    if w_low in ("me", "love", "how are you_start", "how are you_end", "nice to meet you_start", "nice to meet you_end"):
                        probs_w[i_w] *= 0.02
                    elif w_low == "start":
                        probs_w[i_w] = max(float(probs_w[i_w]), 0.65)

        s_w = float(np.sum(probs_w))
        if s_w > 1e-6:
            probs_w = probs_w / s_w

        lbl_w, conf_w, status_w, telem_w = self.verifier.verify(
            probs_w,
            self.words_labels,
            min_confidence=self.STATIC_MIN_CONFIDENCE,
            min_margin=self.STATIC_MIN_MARGIN,
        )
        if lbl_w == "ASL_LETTER" or telem_w.get("top_label") == "ASL_LETTER":
            lbl_w = None
            status_w = "asl_letter"
        return {
            "prediction": lbl_w,
            "confidence": conf_w,
            "status": status_w,
            "telemetry": telem_w,
            "gesture_type": "static_words",
            "model_source": "words_static_mlp",
            "specialist_used": "Static Words MLP",
            "routed_via_fist_specialist": False,
        }

    def predict_dynamic_keyframes(self, word_features_1d: np.ndarray) -> Dict:
        """
        Runs Model Group 3B (`dynamic_keyframe_mlp.keras`) trained purely on the 8 `_START`/`_END`
        dynamic keyframe classes (`How are you_START/END`, `Nice to meet you_START/END`, `J_START/END`, `Z_START/END`).
        Enforces strict 1-hand vs 2-hand structural masking.
        """
        if self._fast_dynamic_keyframe is None or self.dynamic_keyframe_scaler is None:
            return {
                "prediction": None,
                "confidence": 0.0,
                "status": "low_confidence",
                "telemetry": {"top_label": "—", "second_label": "—", "margin": 0.0, "p_top": 0.0, "p_second": 0.0},
                "gesture_type": "dynamic",
                "model_source": "dynamic_keyframe_mlp",
                "specialist_used": "Dynamic Keyframe MLP",
                "routed_via_fist_specialist": False,
            }

        raw_w = aslo_features.validate_word_feature_vector(word_features_1d)
        kf_scaled = self.dynamic_keyframe_scaler.transform(raw_w)
        probs_kf = self._fast_dynamic_keyframe(tf.constant(kf_scaled, dtype=tf.float32)).numpy()[0].copy()

        has_second_hand = bool(np.any(np.abs(raw_w[0, 124:248]) > 1e-4))
        STRICT_1H_KF = {"j_start", "j_end", "z_start", "z_end"}
        STRICT_2H_KF = {
            "how are you_start", "how are you_end",
            "nice to meet you_start", "nice to meet you_end",
        }
        for i_k, k_lbl in enumerate(self.dynamic_keyframe_labels):
            k_low = str(k_lbl).strip().lower()
            if has_second_hand and k_low in STRICT_1H_KF:
                probs_kf[i_k] = 0.0
            elif (not has_second_hand) and k_low in STRICT_2H_KF:
                probs_kf[i_k] = 0.0

        s_k = float(np.sum(probs_kf))
        if s_k > 1e-6:
            probs_kf = probs_kf / s_k

        lbl_k, conf_k, status_k, telem_k = self.verifier.verify(
            probs_kf,
            self.dynamic_keyframe_labels,
            min_confidence=self.STATIC_MIN_CONFIDENCE,
            min_margin=self.STATIC_MIN_MARGIN,
        )
        return {
            "prediction": lbl_k,
            "confidence": conf_k,
            "status": status_k,
            "telemetry": telem_k,
            "gesture_type": "dynamic",
            "model_source": "dynamic_keyframe_mlp",
            "specialist_used": "Dynamic Keyframe MLP",
            "routed_via_fist_specialist": False,
        }

    def predict_merged_auto(
        self,
        asl_features_1d: Optional[np.ndarray],
        word_features_1d: Optional[np.ndarray],
        hand_type: str = "right",
        motion_speed: float = 0.0,
        wrist_y: Optional[float] = None,
    ) -> Dict:
        """
        Unified Merged Router (`meta_router_mlp.keras`):
        Evaluates all 3 separately trained model groups (Model 1: Static ASL, Model 2: Static Words,
        Model 3B: Dynamic Keyframes), feeds `[word_features(252), has_second_hand(1), asl_stats(3),
        words_stats(3), dyn_kf_stats(3)]` (262 dims) into `meta_router_mlp.keras`, and returns the
        winning model group's verified prediction.
        """
        hand_type_clean = (hand_type or "right").lower().strip()
        if word_features_1d is None and asl_features_1d is not None:
            raw_a = np.asarray(asl_features_1d, dtype=np.float32).flatten()
            word_features_1d = np.concatenate([raw_a, np.zeros(aslo_features.EXPECTED_ASL_DIM + 4, dtype=np.float32)])

        if word_features_1d is None:
            if asl_features_1d is not None:
                return self.fuse_stage1_and_stage2_probabilities(asl_features_1d)
            return {
                "prediction": None,
                "confidence": 0.0,
                "status": "ambiguous",
                "telemetry": {"top_label": "—", "second_label": "—", "margin": 0.0, "p_top": 0.0, "p_second": 0.0},
                "gesture_type": "static_asl",
                "model_source": "none",
                "specialist_used": "None",
                "routed_via_fist_specialist": False,
            }

        raw_w = aslo_features.validate_word_feature_vector(word_features_1d)
        has_second_hand = bool(np.any(np.abs(raw_w[0, 124:248]) > 1e-4))
        effective_two_hand = has_second_hand or (hand_type_clean == "both")

        # 1. Model Group 2: Pure Static Words
        res_words = self.predict_static_words(raw_w, force_two_hand=(hand_type_clean == "both"))
        tw = res_words.get("telemetry") or {}
        w_p1 = float(tw.get("p_top", 0.0))
        w_p2 = float(tw.get("p_second", 0.0))
        w_margin = float(w_p1 - w_p2)

        # 2. Model Group 3B: Pure Dynamic Keyframes
        res_kf = self.predict_dynamic_keyframes(raw_w)
        tk = res_kf.get("telemetry") or {}
        kf_p1 = float(tk.get("p_top", 0.0))
        kf_p2 = float(tk.get("p_second", 0.0))
        kf_margin = float(kf_p1 - kf_p2)

        # 3. Model Group 1: Pure Static ASL (only for 1-hand poses)
        res_asl = None
        if (not effective_two_hand) and asl_features_1d is not None:
            res_asl = self.fuse_stage1_and_stage2_probabilities(asl_features_1d)
            ta = res_asl.get("telemetry") or {}
            a_p1 = float(ta.get("p_joint_raw", ta.get("p_top", 0.0)))
            a_p2 = float(ta.get("p_second", 0.0))
            a_margin = float(a_p1 - a_p2)
        else:
            a_p1, a_p2, a_margin = 0.0, 0.0, 0.0

        # Geometric 2-hand START (both thumbs up) & STOP (both pointing down) always belong to Static Words
        if effective_two_hand:
            o0 = raw_w[0, 115:124]
            o1 = raw_w[0, 239:248] if has_second_hand else o0
            both_pointing_down = (
                float(o0[7]) > 0.45 and float(o1[7]) > 0.45
                and float(o0[8]) > 0.20 and float(o1[8]) > 0.20
            )
            both_thumbs_up = (
                float(o0[4]) < -0.78 and float(o1[4]) < -0.78
                and abs(float(o0[3])) < 0.48 and abs(float(o1[3])) < 0.48
                and float(o0[2]) < -0.30 and float(o1[2]) < -0.30
            )
            if both_pointing_down or both_thumbs_up:
                return res_words

        # Fallback if meta_router_mlp is not loaded yet
        if self._fast_meta_router is None or self.meta_router_scaler is None:
            if effective_two_hand:
                return res_words if w_p1 >= kf_p1 else res_kf
            candidates = [(res_asl, a_p1), (res_words, w_p1), (res_kf, kf_p1)]
            candidates = [(r, s) for r, s in candidates if r is not None]
            return max(candidates, key=lambda x: x[1])[0]

        # 4. Run OOD-Aware Merged Meta-Router MLP (265 dims) with penultimate embedding cosine similarity
        def _cos_sim(embed_fn, scaler_obj, raw_vec, centroids):
            if embed_fn is None or scaler_obj is None or centroids is None:
                return 0.0
            emb = embed_fn(tf.constant(scaler_obj.transform(raw_vec), dtype=tf.float32)).numpy()[0]
            norm_e = float(np.linalg.norm(emb))
            if norm_e < 1e-7:
                return 0.0
            return float(np.max((emb / norm_e) @ centroids.T))

        a_cos = (
            _cos_sim(
                self._fast_asl_embed,
                self.asl_stage1_scaler,
                aslo_features.validate_asl_feature_vector(asl_features_1d),
                self.asl_centroids,
            )
            if ((not effective_two_hand) and asl_features_1d is not None)
            else 0.0
        )
        w_cos = _cos_sim(self._fast_words_embed, self.words_scaler, raw_w, self.words_centroids)
        kf_cos = _cos_sim(self._fast_kf_embed, self.dynamic_keyframe_scaler, raw_w, self.kf_centroids)

        meta_vec = np.concatenate(
            [
                raw_w.flatten(),
                np.array([1.0 if effective_two_hand else 0.0], dtype=np.float32),
                np.array(
                    [
                        a_p1, a_p2, a_margin, a_cos,
                        w_p1, w_p2, w_margin, w_cos,
                        kf_p1, kf_p2, kf_margin, kf_cos,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        meta_scaled = self.meta_router_scaler.transform(meta_vec)
        meta_probs = self._fast_meta_router(tf.constant(meta_scaled, dtype=tf.float32)).numpy()[0].copy()

        # Structural constraint: 2-hand inputs can never be static_asl
        if effective_two_hand or res_asl is None:
            for i_m, m_lbl in enumerate(self.meta_router_labels):
                if str(m_lbl) == "static_asl":
                    meta_probs[i_m] = 0.0
            m_sum = float(np.sum(meta_probs))
            if m_sum > 1e-6:
                meta_probs = meta_probs / m_sum

        # If caller explicitly passed 2 hands but slot-1 was empty (e.g., 1-hand CSV row in test), prefer Static Words
        if effective_two_hand and not has_second_hand:
            return res_words

        sorted_m = np.argsort(meta_probs)[::-1]
        best_idx = int(sorted_m[0])
        second_idx = int(sorted_m[1]) if len(sorted_m) > 1 else best_idx
        best_domain = str(self.meta_router_labels[best_idx])
        meta_conf = float(meta_probs[best_idx])
        meta_second = float(meta_probs[second_idx]) if len(sorted_m) > 1 else 0.0
        meta_margin = float(meta_conf - meta_second)
        meta_prob_map = {str(lbl): float(meta_probs[i]) for i, lbl in enumerate(self.meta_router_labels)}

        # 5. Targeted Geometric Disambiguation for Physically Adjacent Cross-Model Pairs:
        #    (a) B (Static ASL) vs hello (Static Words): both are 4-finger flat palms with tucked thumb,
        #        differentiated by palm roll angle `|atan2(x9, -y9)|`: B is strictly vertical (<= 8.5 deg),
        #        while hello is tilted sideways (>= 11.5 deg).
        #    (b) Stationary P / Q / X / D (Static ASL) vs Z_START / Z_END (Dynamic Keyframes):
        #        when the hand is NOT actively moving (`motion_speed < 0.025`) and ASL strongly detects
        #        P, Q, X, or D, never let Z_START/Z_END override the static ASL letter!
        if (not effective_two_hand) and res_asl is not None and asl_features_1d is not None:
            raw_asl_chk = aslo_features.validate_asl_feature_vector(asl_features_1d)
            asl_pred_u = str(res_asl.get("prediction") or res_asl.get("top_label") or "").strip().upper()
            words_pred_l = str(res_words.get("prediction") or (res_words.get("telemetry") or {}).get("top_label") or "").strip().lower()
            kf_pred_u = str(res_kf.get("prediction") or (res_kf.get("telemetry") or {}).get("top_label") or "").strip().upper()

            if asl_pred_u == "B" and words_pred_l == "hello":
                palm_x9 = float(raw_asl_chk[0, 121])
                palm_y9 = float(raw_asl_chk[0, 122])
                roll_deg = abs(float(np.degrees(np.arctan2(palm_x9, -palm_y9))))
                # hello is signed high near forehead (wrist_y <= 0.36) OR tilted sideways (|roll| >= 10 deg);
                # B is signed straight vertical (|roll| < 10 deg) at chest height (wrist_y > 0.36).
                is_hello_geom = (wrist_y is not None and wrist_y <= 0.36) or (roll_deg >= 10.0)
                if is_hello_geom and float(res_words.get("confidence") or 0.0) >= 0.55:
                    best_domain = "static_words"
                    meta_conf = max(meta_conf, 0.88)
                    meta_margin = max(meta_margin, 0.45)
                elif (not is_hello_geom) and float(res_asl.get("confidence") or 0.0) >= 0.55:
                    best_domain = "static_asl"
                    meta_conf = max(meta_conf, 0.88)
                    meta_margin = max(meta_margin, 0.45)

            if (
                best_domain == "dynamic_keyframe"
                and kf_pred_u in ("Z_START", "Z_END")
                and motion_speed < 0.025
                and asl_pred_u in ("P", "Q", "X", "D")
                and float(res_asl.get("confidence") or 0.0) >= 0.65
            ):
                best_domain = "static_asl"
                meta_conf = max(meta_conf, 0.85)
                meta_margin = max(meta_margin, 0.40)

        if best_domain == "static_asl" and res_asl is not None:
            winner = res_asl
        elif best_domain == "dynamic_keyframe":
            winner = res_kf
        else:
            winner = res_words

        winner = dict(winner)
        winner_telem = dict(winner.get("telemetry") or {})
        winner_telem["meta_domain"] = best_domain
        winner_telem["meta_conf"] = meta_conf
        winner_telem["meta_margin"] = meta_margin
        winner_telem["meta_probs"] = meta_prob_map
        winner_telem["ood_cos_sims"] = {"asl": a_cos, "words": w_cos, "dyn_kf": kf_cos}
        winner["telemetry"] = winner_telem

        # Strict Meta-Router Gate: never let an uncertain cross-domain guess override another model!
        req_meta_conf = 0.72 if (best_domain == "dynamic_keyframe" and not has_second_hand) else 0.65
        if meta_conf < req_meta_conf or meta_margin < 0.18:
            winner["prediction"] = None
            winner["status"] = "ambiguous"
        return winner

    def _evaluate_single_dynamic_window(
        self,
        window_frames: List,
        req_conf: float,
        enforce_motion_envelope: bool,
    ) -> Dict:
        raw_d = aslo_features.validate_dynamic_feature_vector(
            aslo_features.extract_dynamic_mlp_features(window_frames)
        )
        d_scaled = self.dynamic_scaler.transform(raw_d)
        probs_d = self._fast_dynamic(tf.constant(d_scaled, dtype=tf.float32)).numpy()[0].copy()

        total_len = float(raw_d[0, 389])
        bw = float(raw_d[0, 390])
        bh = float(raw_d[0, 391])
        shape_change = float(np.linalg.norm(raw_d[0, 0:63] - raw_d[0, 230:293]))
        has_real_hand = bool(np.any(np.abs(raw_d[0, 0:63]) > 1e-4))

        lbl_d, conf_d, status_d, telem_d = self.verifier.verify(
            probs_d,
            self.dynamic_labels,
            min_confidence=req_conf,
            min_margin=self.STATIC_MIN_MARGIN,
        )

        if lbl_d == "STATIC_HOLD" or telem_d.get("top_label") == "STATIC_HOLD":
            lbl_d = None
            status_d = "static_hold"

        if enforce_motion_envelope and lbl_d is not None:
            if has_real_hand:
                sh_mid_end = float(np.linalg.norm(raw_d[0, 115:178] - raw_d[0, 230:293]))
                ext_s = raw_d[0, 78:83]    # [T, I, M, R, P] at start
                ext_m = raw_d[0, 193:198]  # [T, I, M, R, P] at mid
                ext_e = raw_d[0, 308:313]  # [T, I, M, R, P] at end
                lbl_up = str(lbl_d).strip().upper()

                # 1. Global active dynamic trajectory & continuous mid->end morphing envelope:
                #    All 4 dynamic signs have total_len >= 0.90, max(bw, bh) >= 0.44, shape_change >= 0.80,
                #    and sh_mid_end >= 0.30 (whereas moving into a static sign and holding it has sh_mid_end < 0.28).
                if total_len < 0.88 or max(bw, bh) < 0.42 or shape_change < 0.78 or sh_mid_end < 0.28:
                    lbl_d = None
                    status_d = "static_hold"
                elif lbl_up == "J":
                    # J requires the Pinky finger (index 4) to be extended and dominant over Index (index 1)
                    max_pinky = max(float(ext_s[4]), float(ext_m[4]), float(ext_e[4]))
                    pinky_adv = max(float(ext_s[4] - ext_s[1]), float(ext_m[4] - ext_m[1]))
                    if max_pinky < 1.55 or pinky_adv < 0.35 or total_len < 1.25 or bh < 0.40 or shape_change < 1.05 or sh_mid_end < 0.45:
                        lbl_d = None
                        status_d = "static_hold"
                elif lbl_up == "Z":
                    # Z requires Index finger (index 1) extended while Middle/Ring/Pinky are curled, plus a 2D zig-zag path
                    max_idx = max(float(ext_s[1]), float(ext_m[1]), float(ext_e[1]))
                    min_mid = min(float(ext_s[2]), float(ext_m[2]))
                    idx_adv = max(float(ext_s[1] - ext_s[2]), float(ext_m[1] - ext_m[2]))
                    if max_idx < 1.05 or min_mid > 0.88 or idx_adv < 0.35 or bw < 0.65 or bh < 0.65 or total_len < 1.60 or sh_mid_end < 0.35:
                        lbl_d = None
                        status_d = "static_hold"
                elif "HOW ARE YOU" in lbl_up:
                    # How are you requires continuous outward palm opening (sh_se >= 1.40, sh_mid_end >= 0.80, bh >= 0.68, total_len >= 0.90)
                    if total_len < 0.90 or bh < 0.65 or shape_change < 1.40 or sh_mid_end < 0.78:
                        lbl_d = None
                        status_d = "static_hold"
                elif "NICE TO MEET YOU" in lbl_up:
                    # Nice to meet you requires both horizontal approach (bw >= 0.50) and vertical travel (bh >= 0.60)
                    if total_len < 0.88 or bw < 0.48 or bh < 0.60 or shape_change < 1.35 or sh_mid_end < 0.30:
                        lbl_d = None
                        status_d = "static_hold"

        return {
            "prediction": lbl_d,
            "confidence": conf_d,
            "status": status_d,
            "telemetry": telem_d,
            "gesture_type": "dynamic",
            "model_source": "dynamic_mlp",
            "specialist_used": "Dynamic 25-Frame MLP",
            "routed_via_fist_specialist": False,
        }

    def predict_dynamic_sequence(
        self,
        sequence_buffer: List,
        min_confidence: Optional[float] = None,
        enforce_motion_envelope: bool = False,
    ) -> Dict:
        """
        Runs Model Group 3 (`dynamic_mlp.keras`) with multi-scale active-window segmentation:
        evaluates the full buffer and trailing active sub-windows (`[-12:]`, `[-16:]`, `[-20:]`)
        so that fast (0.5s) and normal (1.0s) dynamic gestures (`J, Z, How are you, Nice to meet you`)
        align their Start, Middle, and End keyframes even when preceded by stationary waiting frames.
        """
        req_conf = min_confidence if min_confidence is not None else self.DYNAMIC_MIN_CONFIDENCE
        best_res = self._evaluate_single_dynamic_window(sequence_buffer, req_conf, enforce_motion_envelope)

        n_frames = len(sequence_buffer) if sequence_buffer is not None else 0
        if enforce_motion_envelope and n_frames >= 14:
            for tail_len in (12, 16, 20):
                if n_frames > tail_len:
                    sub_res = self._evaluate_single_dynamic_window(
                        sequence_buffer[-tail_len:], req_conf, enforce_motion_envelope
                    )
                    if sub_res["prediction"] is not None:
                        if best_res["prediction"] is None or sub_res["confidence"] > best_res["confidence"]:
                            best_res = sub_res

        return best_res

    def route_and_predict(
        self,
        asl_features_1d: Optional[np.ndarray] = None,
        word_features_1d: Optional[np.ndarray] = None,
        hand_type: str = "right",
        mode: str = "auto",
        sequence_buffer: Optional[List] = None,
        motion_speed: float = 0.0,
        static_features_1d: Optional[np.ndarray] = None,
        wrist_y: Optional[float] = None,
    ) -> Dict:
        """
        PART 3.2: 3-Type Cooperative Routing (`static_asl`, `static_words`, `dynamic_gestures`):
        - Dynamic Gate: If `mode == "dynamic"` OR (`mode == "auto"` and `motion_speed >= MOTION_THRESHOLD` and `len(sequence_buffer) == 25`):
          run `dynamic_mlp.keras`. If confidence >= 0.68, return immediately and suppress static models.
        - Static Gate:
          * `mode == "asl"`: Run Static ASL Cooperative Two-Stage MLPs only.
          * `mode == "words"`: Run Static Words MLP only.
          * `mode == "auto"`:
            - If `hand_type == "both"`, route strictly to Static Words MLP.
            - If `hand_type == "right"`, run both Static ASL (Soft Two-Stage) and Static Words MLP,
              compare their calibrated top confidence & margin, and return the winner tagged with
              `gesture_type` and `specialist_used`.
        """
        mode_clean = (mode or "auto").lower().strip()
        hand_type_clean = (hand_type or "right").lower().strip()
        seq_len = len(sequence_buffer) if sequence_buffer is not None else 0

        # Auto hot-reload if any model on disk was retrained or merged
        try:
            check_files = [
                os.path.join(self.models_dir, "asl_stage1_router_mlp.keras"),
                os.path.join(self.models_dir, "words_static_mlp.keras"),
                os.path.join(self.models_dir, "dynamic_mlp.keras"),
                os.path.join(self.models_dir, "dynamic_keyframe_mlp.keras"),
                os.path.join(self.models_dir, "meta_router_mlp.keras"),
            ]
            existing_mtimes = [os.path.getmtime(p) for p in check_files if os.path.exists(p)]
            if existing_mtimes and max(existing_mtimes) > self._loaded_mtime + 0.5:
                self.load_all_models()
        except Exception:
            pass

        # Legacy 310-dim row support if passed via static_features_1d
        if static_features_1d is not None:
            arr_st = np.asarray(static_features_1d, dtype=np.float32).flatten()
            if arr_st.size == aslo_features.FEATURE_LEN:
                if asl_features_1d is None:
                    asl_features_1d = aslo_features.csv_row_to_asl_features(arr_st)
                if word_features_1d is None:
                    word_features_1d = aslo_features.csv_row_to_word_features(arr_st)
            elif arr_st.size == aslo_features.EXPECTED_ASL_DIM and asl_features_1d is None:
                asl_features_1d = arr_st
            elif arr_st.size == aslo_features.EXPECTED_WORD_DIM and word_features_1d is None:
                word_features_1d = arr_st

        # 0. Dynamic Gate (Evaluated when mode=='dynamic' OR in 'auto' mode whenever sequence_buffer >= 8 frames
        #    and either instantaneous motion_speed >= MOTION_THRESHOLD or the window itself has active trajectory travel)
        if mode_clean == "dynamic":
            if seq_len >= 4:
                res_dyn = self.predict_dynamic_sequence(
                    sequence_buffer, min_confidence=self.DYNAMIC_MIN_CONFIDENCE
                )
                if res_dyn["prediction"] is not None and res_dyn["confidence"] >= self.DYNAMIC_MIN_CONFIDENCE:
                    return res_dyn
                return {
                    "prediction": None,
                    "confidence": res_dyn["confidence"],
                    "status": "ambiguous",
                    "telemetry": res_dyn["telemetry"],
                    "gesture_type": "dynamic",
                    "model_source": "dynamic_mlp",
                    "specialist_used": "Dynamic 25-Frame MLP",
                    "routed_via_fist_specialist": False,
                }
            return {
                "prediction": None,
                "confidence": 0.0,
                "status": "ambiguous",
                "telemetry": {"top_label": "—", "second_label": "—", "margin": 0.0, "p_top": 0.0, "p_second": 0.0},
                "gesture_type": "dynamic",
                "model_source": "dynamic_mlp",
                "specialist_used": "Dynamic 25-Frame MLP",
                "routed_via_fist_specialist": False,
            }

        # 1. Control & Static vs Dynamic Cooperative Gate in Unified 'auto' Mode:
        #    - First, run the Merged Meta-Router (`predict_merged_auto`) so ASL, Static Words, and Dynamic Keyframes
        #      are cleanly separated. If the merged result is a Static Control Word (`START`, `STOP`, `SPACE`,
        #      `BACKSPACE`, `NEUTRAL`, `LOVE`) with strong confidence, return immediately so dynamic motion
        #      never overrides control gestures!
        TWO_HAND_CONTROL_SET = {"start", "stop", "space", "backspace", "back space", "neutral", "love"}
        precomputed_merged_res = None
        if mode_clean == "auto" and (word_features_1d is not None or asl_features_1d is not None):
            precomputed_merged_res = self.predict_merged_auto(
                asl_features_1d=asl_features_1d,
                word_features_1d=word_features_1d,
                hand_type=hand_type_clean,
                motion_speed=float(motion_speed or 0.0),
                wrist_y=wrist_y,
            )
            pred_ctrl_low = str(precomputed_merged_res.get("prediction") or "").strip().lower()
            conf_ctrl = float(precomputed_merged_res.get("confidence") or 0.0)
            if (
                precomputed_merged_res.get("gesture_type") == "static_words"
                and precomputed_merged_res.get("status") == "valid"
                and pred_ctrl_low in TWO_HAND_CONTROL_SET
                and conf_ctrl >= 0.58
            ):
                if hand_type_clean == "both" or pred_ctrl_low in ("start", "stop"):
                    return precomputed_merged_res

        # 2. Dynamic Gate in Unified 'auto' Mode:
        #    Evaluates Model Group 3 (`dynamic_mlp.keras`) ONLY when the hand is actively moving right now
        #    (`motion_speed >= 0.032` or `recent_tail_disp >= 0.35`) AND has traversed a real dynamic trajectory (`window_disp >= 0.65`),
        #    so stationary or settling static signs NEVER false-trigger dynamic gestures!
        window_disp = 0.0
        recent_tail_disp = 0.0
        window_shape_change = 0.0
        sh_mid_end_est = 0.0
        res_dyn_candidate = None
        if mode_clean == "auto" and seq_len >= 8:
            try:
                s_arr = aslo_features.normalize_sequence_length(sequence_buffer, target_len=aslo_features.SEQ_LEN)
                sc_est = float(np.median(np.linalg.norm(s_arr[:, 9, :] - s_arr[:, 0, :], axis=1))) + 1e-4
                w_disp = float(np.max(np.linalg.norm(s_arr[:, 0, :2] - s_arr[0:1, 0, :2], axis=1))) / sc_est
                i_disp = float(np.max(np.linalg.norm(s_arr[:, 8, :2] - s_arr[0:1, 8, :2], axis=1))) / sc_est
                p_disp = float(np.max(np.linalg.norm(s_arr[:, 20, :2] - s_arr[0:1, 20, :2], axis=1))) / sc_est
                window_disp = max(w_disp, i_disp * 0.75, p_disp * 0.75)

                # Recent tail motion across the last 6 frames (~0.24s) verifies the hand is STILL actively moving
                # and has not already settled into a static gesture hold!
                tail_arr = s_arr[-6:]
                tw_disp = float(np.max(np.linalg.norm(tail_arr[:, 0, :2] - tail_arr[0:1, 0, :2], axis=1))) / sc_est
                ti_disp = float(np.max(np.linalg.norm(tail_arr[:, 8, :2] - tail_arr[0:1, 8, :2], axis=1))) / sc_est
                tp_disp = float(np.max(np.linalg.norm(tail_arr[:, 20, :2] - tail_arr[0:1, 20, :2], axis=1))) / sc_est
                recent_tail_disp = max(tw_disp, ti_disp * 0.75, tp_disp * 0.75)

                sh_start = aslo_features.extract_hand_shape_features(s_arr[0])[:63]
                sh_mid = aslo_features.extract_hand_shape_features(s_arr[aslo_features.SEQ_LEN // 2])[:63]
                sh_end = aslo_features.extract_hand_shape_features(s_arr[-1])[:63]
                window_shape_change = float(np.linalg.norm(sh_end - sh_start))
                sh_mid_end_est = float(np.linalg.norm(sh_end - sh_mid))
            except Exception:
                window_disp = 0.0
                recent_tail_disp = 0.0
                window_shape_change = 0.0
                sh_mid_end_est = 0.0

            # Require active ongoing motion (`motion_speed >= 0.032` or `recent_tail_disp >= 0.35`)
            # AND substantial trajectory travel (`window_disp >= 0.65`) AND continuous mid->end shape change (`sh_mid_end_est >= 0.30`)
            should_check_dyn = (
                (motion_speed >= 0.032 or recent_tail_disp >= 0.35)
                and window_disp >= 0.65
                and window_shape_change >= 0.80
                and sh_mid_end_est >= 0.28
            )
            if should_check_dyn:
                res_dyn = self.predict_dynamic_sequence(
                    sequence_buffer,
                    min_confidence=0.72,
                    enforce_motion_envelope=True,
                )
                res_dyn_candidate = res_dyn
                if res_dyn["prediction"] is not None and res_dyn["confidence"] >= 0.72:
                    return res_dyn
                has_valid_static = (
                    (asl_features_1d is not None and np.any(np.abs(asl_features_1d) > 1e-4))
                    or (word_features_1d is not None and np.any(np.abs(word_features_1d) > 1e-4))
                )
                if not has_valid_static:
                    return {
                        "prediction": None,
                        "confidence": res_dyn["confidence"],
                        "status": "ambiguous",
                        "telemetry": res_dyn["telemetry"],
                        "gesture_type": "dynamic",
                        "model_source": "dynamic_mlp",
                        "specialist_used": "Dynamic 25-Frame MLP",
                        "routed_via_fist_specialist": False,
                    }

        def _guard_dynamic_vs_me_you(static_res: Dict) -> Dict:
            """
            Prevents static 'ME' or 'YOU' from hijacking an active dynamic gesture
            ('How are you', 'Nice to meet you', 'J', 'Z') ONLY when the sequence buffer
            passes the strict dynamic motion & anatomical envelope in `res_dyn_candidate`.
            """
            pred_u = str(static_res.get("prediction") or "").strip().upper()
            if pred_u in ("ME", "YOU") and res_dyn_candidate is not None:
                if res_dyn_candidate.get("prediction") is not None and float(res_dyn_candidate.get("confidence") or 0.0) >= 0.65:
                    return res_dyn_candidate
            return static_res

        # 3. Dedicated Mode Gates (`asl`, `words`) vs Unified Merged `auto` Mode
        if mode_clean in ("asl", "static_asl"):
            if hand_type_clean == "both" and word_features_1d is not None:
                res_ctrl = self.predict_static_words(word_features_1d)
                if (
                    res_ctrl.get("status") == "valid"
                    and str(res_ctrl.get("prediction") or "").strip().lower() in TWO_HAND_CONTROL_SET
                ):
                    return res_ctrl
            if asl_features_1d is None:
                return {
                    "prediction": None,
                    "confidence": 0.0,
                    "status": "ambiguous",
                    "telemetry": {"top_label": "—", "second_label": "—", "margin": 0.0, "p_top": 0.0, "p_second": 0.0},
                    "gesture_type": "static_asl",
                    "model_source": "none",
                    "specialist_used": "None",
                    "routed_via_fist_specialist": False,
                }
            return self.fuse_stage1_and_stage2_probabilities(asl_features_1d)

        if mode_clean in ("words", "static_words"):
            if word_features_1d is None and asl_features_1d is not None:
                raw_a = np.asarray(asl_features_1d, dtype=np.float32).flatten()
                word_features_1d = np.concatenate([raw_a, np.zeros(aslo_features.EXPECTED_ASL_DIM + 4, dtype=np.float32)])
            if word_features_1d is None:
                return {
                    "prediction": None,
                    "confidence": 0.0,
                    "status": "ambiguous",
                    "telemetry": {"top_label": "—", "second_label": "—", "margin": 0.0, "p_top": 0.0, "p_second": 0.0},
                    "gesture_type": "static_words",
                    "model_source": "none",
                    "specialist_used": "None",
                    "routed_via_fist_specialist": False,
                }
            return self.predict_static_words(word_features_1d)

        # Unified `auto` mode: return Meta-Router merged decision across Static ASL, Static Words, and Dynamic Keyframes
        if precomputed_merged_res is not None:
            return _guard_dynamic_vs_me_you(precomputed_merged_res)
        return _guard_dynamic_vs_me_you(
            self.predict_merged_auto(
                asl_features_1d=asl_features_1d,
                word_features_1d=word_features_1d,
                hand_type=hand_type_clean,
                motion_speed=float(motion_speed or 0.0),
                wrist_y=wrist_y,
            )
        )


# Alias for backward compatibility
StatelessHierarchicalMLPRouter = ASLOEnsembleRouter


# =====================================================================
# 5. LEGACY DYNAMIC TIMER & DATASET LOADER (BACKWARD COMPATIBILITY)
# =====================================================================

class TimerDynamicGestureManager:
    def __init__(
        self,
        min_duration: float = 0.35,
        max_duration: float = 2.80,
        pause_speed_threshold: float = 0.020,
        pause_required_frames: int = 4,
        max_buffer_frames: int = 32,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.pause_speed_threshold = pause_speed_threshold
        self.pause_required_frames = pause_required_frames
        self.max_buffer_frames = max_buffer_frames
        self.is_active = False
        self.gesture_name: Optional[str] = None
        self.start_time = 0.0
        self.frame_buffer = collections.deque(maxlen=max_buffer_frames)
        self.still_counter = 0
        self.is_paused = False
        self.pause_start_time = 0.0
        self.pause_accumulator = 0.0

    def reset(self):
        self.is_active = False
        self.gesture_name = None
        self.frame_buffer.clear()


class ASLODatasetLoader:
    def __init__(self, dataset_path: str = "picture_file"):
        self.dataset_path = dataset_path
        self.enhancer = LowLightEnhancer()
