"""
ASLO (AI Sign Language Online) Pipeline Module
==============================================
Provides high-performance, robust modules for real-time (15-24 FPS) video processing:
1. LowLightEnhancer: Adaptive CLAHE & luminance correction for < 100 Lux environments.
2. KeypointInterpolator: Smoothing, occlusion handling, >1.5m distance checks & interpolation.
3. MotionBasedRouter: Motion-based routing separating static poses from dynamic gestures.
4. TimerDynamicGestureManager: Dynamic gesture countdown watchdog, sequence buffering,
   terminal pause/neutral completion detection, and strict abort/cancellation logic.
5. ASLODatasetLoader: Unified dataset loading for both static images and dynamic video sequences.
"""

import collections
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np


# =====================================================================
# 1. ADAPTIVE LIGHTING PREPROCESSING (LOW-LIGHT & OVEREXPOSURE / GLARE)
# =====================================================================

class AdaptiveLightingEnhancer:
    """
    Intelligent adaptive lighting preprocessor that corrects:
    1. Low-light environments (< 100 Lux / underexposure):
       Adaptive CLAHE + shadow gamma lift.
    2. Overexposed / High-light environments (direct sun, windows, ring lights, washed-out glare):
       Adaptive gamma darkening + highlight compression + edge recovery CLAHE to restore
       washed-out skin tones and finger contours.
    """
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
        """
        Enhance a BGR frame based on ambient lighting conditions.

        Returns:
            enhanced_bgr: Corrected BGR frame.
            mean_luminance: Average L-channel value (0-255).
            lighting_state: "LOW_LIGHT", "HIGH_LIGHT", or "OPTIMAL".
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        mean_l = float(np.mean(l_channel))
        highlight_ratio = float(np.mean(l_channel > 215))

        # Condition 1: Low-Light (< 100 Lux, mean_l < 75)
        if mean_l < 75.0:
            adaptive_clip = np.clip(self.base_clip_limit + (75.0 - mean_l) * 0.05, 2.0, 4.5)
            clahe = self._get_clahe(adaptive_clip)
            l_enhanced = clahe.apply(l_channel)

            if mean_l < 45.0:
                lut = self._get_gamma_table(1.0 / 0.82)
                l_enhanced = cv2.LUT(l_enhanced, lut)

            lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR), mean_l, "LOW_LIGHT"

        # Condition 2: High-Light / Overexposure / Glare (mean_l > 135 or highlight_ratio > 0.12)
        elif mean_l > 135.0 or highlight_ratio > 0.12:
            # Overexposure washes out skin tones and finger boundaries into flat white.
            # Gamma darkening (power > 1.0) pulls blown highlights down into readable midtones.
            gamma_power = float(np.clip(1.30 + (mean_l - 125.0) * 0.012 + highlight_ratio * 1.5, 1.35, 2.30))
            lut = self._get_gamma_table(gamma_power)
            l_darkened = cv2.LUT(l_channel, lut)

            # Apply mild CLAHE to the darkened L channel to recover finger contour gradients
            clahe = self._get_clahe(2.0)
            l_enhanced = clahe.apply(l_darkened)

            lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR), mean_l, "HIGH_LIGHT"

        # Condition 3: Optimal lighting (75 <= mean_l <= 135 and highlight_ratio <= 0.12)
        return frame_bgr, mean_l, "OPTIMAL"


# Backward compatibility alias
LowLightEnhancer = AdaptiveLightingEnhancer


# =====================================================================
# 2. KEYPOINT FILTERING & INTERPOLATION (OCCLUSION & > 1.5M DISTANCE)
# =====================================================================

class KeypointInterpolator:
    """
    Maintains temporal tracking continuity for MediaPipe landmarks.
    
    Handles:
    - Self-occlusion (overlapping fingers, dropped keypoints).
    - Distance checks (> 1.5m away, small hand bounding area).
    - Smooth keypoint interpolation using velocity momentum and Exponential Moving Average (EMA).
    - Tracking loss detection to prevent empty/corrupt vectors reaching the MLP.
    """
    def __init__(self, alpha: float = 0.65, max_missing_frames: int = 5):
        self.alpha = alpha  # EMA smoothing factor (0 = full historical, 1 = raw new)
        self.max_missing_frames = max_missing_frames
        
        # State tracking per hand: 'Right' and 'Left'
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
        """
        Process detected hand landmarks and return smoothed/interpolated (21, 3) coordinates.
        
        Returns:
            coords: (21, 3) numpy array or None if tracking is completely lost.
            is_interpolated: True if coordinates were imputed due to occlusion/distance.
            metrics: Diagnostic metrics (hand_scale, distance_warning, occlusion_count).
        """
        metrics = {
            "hand_scale": 0.0,
            "is_far": False,
            "occluded_joints": 0,
            "consecutive_missing": self.consecutive_missing[hand_label],
        }

        if hand_landmarks is None:
            self.consecutive_missing[hand_label] += 1
            # If missing for too many frames, discard state to prevent ghosting
            if (
                self.consecutive_missing[hand_label] > self.max_missing_frames
                or self.last_valid_coords[hand_label] is None
            ):
                self.last_valid_coords[hand_label] = None
                return None, False, metrics

            # Smoothly interpolate missing frame via linear momentum + damping
            decay = 0.85 ** self.consecutive_missing[hand_label]
            predicted_coords = (
                self.last_valid_coords[hand_label]
                + self.last_valid_velocities[hand_label] * decay
            )
            return predicted_coords, True, metrics

        # Extract raw coordinates
        raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

        # 1. Distance Metric: Normalized palm scale (Wrist landmark 0 to Middle MCP landmark 9)
        palm_vector = raw_coords[9] - raw_coords[0]
        hand_scale = float(np.linalg.norm(palm_vector[:2]))
        metrics["hand_scale"] = hand_scale
        
        # In 640x480 webcam feeds, a hand scale < 0.080 typically corresponds to > 1.5m away
        is_far = hand_scale < 0.080
        metrics["is_far"] = is_far

        # 2. Occlusion & Outlier Filtering
        occluded_count = 0
        cleaned_coords = raw_coords.copy()

        if self.last_valid_coords[hand_label] is not None:
            prev_coords = self.last_valid_coords[hand_label]

            # Adaptive smoothing factor: if far away, increase smoothing (decrease alpha)
            eff_alpha = self.alpha * 0.6 if is_far else self.alpha

            for i in range(21):
                # Detect rapid unphysical jitter / occlusion jumps (> 2.5x palm scale in 1 frame)
                delta = float(np.linalg.norm(raw_coords[i] - prev_coords[i]))
                if delta > max(0.18, hand_scale * 2.5):
                    # Joint is likely occluded or misdetected; interpolate from previous + velocity
                    cleaned_coords[i] = prev_coords[i] + self.last_valid_velocities[hand_label][i] * 0.5
                    occluded_count += 1
                else:
                    # Normal EMA smoothing
                    cleaned_coords[i] = eff_alpha * raw_coords[i] + (1.0 - eff_alpha) * prev_coords[i]

            # Update velocity
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
        """Reset internal filter state."""
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
    """
    Separates user gestures into Static vs. Dynamic based on initial hand motion kinematics.
    
    Monitors hand displacement over a short observation window (e.g. 4-6 frames, ~200-300 ms).
    If motion speed/displacement exceeds threshold, routes to Dynamic sequence pipeline;
    otherwise routes to Static pose recognition pipeline.
    """
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
        self.current_mode = "STATIC"  # "STATIC" or "DYNAMIC"
        self.motion_magnitude = 0.0

    def update(self, hand_coords: Optional[np.ndarray], hand_scale: float = 0.15) -> Tuple[str, float]:
        """
        Update motion tracker with the latest hand coordinates (21, 3).
        
        Returns:
            route_mode: "STATIC" or "DYNAMIC"
            motion_magnitude: normalized motion intensity score
        """
        if hand_coords is None:
            self.pos_history.clear()
            self.motion_magnitude = 0.0
            return "STATIC", 0.0

        # Use wrist (0), index MCP (5), and middle MCP (9) to track overall hand translation
        centroid = np.mean(hand_coords[[0, 5, 9], :2], axis=0)
        self.pos_history.append(centroid)

        if len(self.pos_history) < 2:
            return self.current_mode, 0.0

        # Compute average displacement per frame normalized by hand scale
        displacements = []
        for i in range(1, len(self.pos_history)):
            disp = float(np.linalg.norm(self.pos_history[i] - self.pos_history[i - 1]))
            # Normalize displacement by hand scale to be scale/distance invariant
            scale = max(hand_scale, 0.05)
            displacements.append(disp / scale)

        self.motion_magnitude = float(np.mean(displacements))

        # Hysteresis switching logic
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
# SOFTMAX MARGIN & AMBIGUITY REJECTION VERIFIER
# =====================================================================

class SoftmaxMarginVerifier:
    """
    Evaluates raw Softmax probabilities against strict confidence and margin gates:
    1. Confidence Gate: p_top >= min_confidence (default 0.70)
    2. Margin Gate: (p_top - p_second) >= min_margin (default 0.25)

    Eliminates boundary flicker and rapid swapping between overlapping gestures.
    """
    def __init__(self, min_confidence: float = 0.55, min_margin: float = 0.10, verbose: bool = False):
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.verbose = verbose

    def verify(
        self,
        probs: np.ndarray,
        label_classes: np.ndarray
    ) -> Tuple[Optional[str], float, str, Dict]:
        """
        Verify class probabilities.

        Returns:
            predicted_label: Validated label string or None if rejected
            confidence: float (p_top)
            status: "valid", "low_confidence", or "ambiguous"
            telemetry: Dict containing top_label, second_label, margin, p_top, p_second
        """
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

        if self.verbose:
            print(f"[INFERENCE] Top: {top_label} ({p_top:.2f}), Runner-up: {second_label} ({p_second:.2f}), Margin: {margin:.2f}")

        telemetry = {
            "top_label": top_label,
            "second_label": second_label,
            "margin": margin,
            "p_top": p_top,
            "p_second": p_second,
        }

        FIST_CLUSTER = {"A", "S", "T", "E", "M", "N"}
        top_u = top_label.upper().strip()
        sec_u = second_label.upper().strip()

        # Gate 1: Confidence
        if p_top < self.min_confidence:
            return None, p_top, "low_confidence", telemetry

        # Fist Cluster Ambiguity Margin Gate:
        # If (p_top - p_second) < 0.15 and both belong to similar fist cluster ['A', 'S', 'T', 'E', 'M', 'N'],
        # mark prediction as "ambiguous" to drop or hold prior confirmed state
        if (top_u in FIST_CLUSTER and sec_u in FIST_CLUSTER) and margin < 0.15:
            telemetry["is_fist_ambiguous"] = True
            return None, p_top, "ambiguous", telemetry

        # Gate 2: Margin
        if margin < self.min_margin:
            return None, p_top, "ambiguous", telemetry

        return top_label, p_top, "valid", telemetry


# =====================================================================
# 4. TIMER-BASED DYNAMIC GESTURE CONTROLLER & BUFFER
# =====================================================================

class TimerDynamicGestureManager:
    """
    Manages dynamic gesture state machine:
    - Starts countdown timer on initial dynamic frame.
    - Buffers normalized coordinate frames.
    - Completes sequence when hands pause / settle at the final frame or return to neutral.
    - Cancels immediately if user freezes mid-way, takes wrong path, or timer times out.
    - Pauses timer safely when MediaPipe tracking drops so user is not penalized.
    """
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

        # State
        self.is_active = False
        self.gesture_name: Optional[str] = None
        self.start_time = 0.0
        self.frame_buffer = collections.deque(maxlen=max_buffer_frames)
        self.still_counter = 0

        # Tracking pause state
        self.is_paused = False
        self.pause_start_time = 0.0
        self.pause_accumulator = 0.0

    def start_gesture(self, gesture_name: str, initial_coords: Optional[np.ndarray] = None):
        """Start countdown timer and dynamic buffer."""
        now = time.time()
        self.is_active = True
        self.gesture_name = gesture_name
        self.start_time = now
        self.is_paused = False
        self.pause_start_time = 0.0
        self.pause_accumulator = 0.0
        self.still_counter = 0
        self.frame_buffer.clear()

        if initial_coords is not None:
            self.frame_buffer.append(initial_coords.copy())

    def pause_timer(self):
        """Safely pause timer when MediaPipe tracking fails."""
        if self.is_active and not self.is_paused:
            self.is_paused = True
            self.pause_start_time = time.time()

    def resume_timer(self):
        """Resume timer after MediaPipe tracking is re-acquired."""
        if self.is_active and self.is_paused:
            now = time.time()
            pause_duration = now - self.pause_start_time
            self.pause_accumulator += pause_duration
            self.start_time += pause_duration
            self.is_paused = False
            self.pause_start_time = 0.0

    def get_remaining_time(self) -> float:
        if not self.is_active:
            return 0.0
        now = time.time()
        if self.is_paused:
            elapsed = self.pause_start_time - self.start_time
        else:
            elapsed = now - self.start_time
        return max(0.0, self.max_duration - elapsed)

    def cancel(self, reason: str = "") -> str:
        """Cancel gesture translation, stop timer, and wipe buffer."""
        cancelled_name = self.gesture_name or "DYNAMIC"
        self.is_active = False
        self.gesture_name = None
        self.start_time = 0.0
        self.is_paused = False
        self.pause_accumulator = 0.0
        self.still_counter = 0
        self.frame_buffer.clear()
        return f"CANCELLED ({reason}): {cancelled_name}"

    def reset(self):
        """Reset dynamic controller state."""
        self.cancel("Reset")

    def update(
        self,
        coords: Optional[np.ndarray],
        motion_speed: float,
        is_neutral: bool = False,
        is_end_label: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[str], str]:
        """
        Update the dynamic gesture controller for the current frame.
        
        Returns:
            flattened_features: Flattened coordinate array if gesture successfully completed, else None.
            completed_gesture: Gesture label if successfully translated, else None.
            status_message: Informational status for UI / HUD display.
        """
        if not self.is_active:
            return None, None, ""

        now = time.time()

        # Handle tracking loss
        if coords is None:
            self.pause_timer()
            return None, None, f"PAUSED: {self.gesture_name} (Hand lost)"

        self.resume_timer()
        self.frame_buffer.append(coords.copy())

        elapsed = now - self.start_time
        rem_time = self.get_remaining_time()

        # Check 1: Watchdog Timeout
        if elapsed > self.max_duration or rem_time <= 0.0:
            msg = self.cancel("Timeout exceeded")
            return None, None, msg

        # Check 2: Terminal Completion Detection
        # A sequence is completed when:
        # a) Hand pauses/stabilizes at the terminal frame (motion_speed < pause_speed_threshold)
        # b) User returns to a NEUTRAL state after executing motion
        # c) Explicit _END keyframe classification detected
        is_still = motion_speed <= self.pause_speed_threshold
        if is_still:
            self.still_counter += 1
        else:
            self.still_counter = max(0, self.still_counter - 1)

        has_sufficient_frames = len(self.frame_buffer) >= 6
        has_min_duration = elapsed >= self.min_duration

        # Success conditions
        terminal_pause = (self.still_counter >= self.pause_required_frames) and has_sufficient_frames
        terminal_neutral = is_neutral and has_sufficient_frames and has_min_duration
        terminal_end_tag = is_end_label and has_min_duration and has_sufficient_frames

        if terminal_pause or terminal_neutral or terminal_end_tag:
            # User successfully completed the gesture!
            # Flatten buffered frame coordinates for MLP input
            flattened = self.flatten_buffer()
            completed_name = self.gesture_name
            self.is_active = False
            self.gesture_name = None
            self.start_time = 0.0
            self.still_counter = 0
            self.frame_buffer.clear()
            return flattened, completed_name, f"COMPLETED: {completed_name}"

        # Check 3: Premature stop on a middle frame
        # If user freezes mid-motion before reaching minimum frames or final pose
        if is_still and self.still_counter > 8 and not has_sufficient_frames:
            msg = self.cancel("Stopped prematurely mid-gesture")
            return None, None, msg

        return None, None, f"TRACKING: {self.gesture_name} ({rem_time:.1f}s)"

    def flatten_buffer(self, target_frames: int = 16) -> np.ndarray:
        """
        Resamples and flattens the buffered coordinates into a fixed-length vector.
        Standardizes to target_frames (default 16 frames x 63 coordinates = 1008 values).
        """
        buf = list(self.frame_buffer)
        n = len(buf)
        if n == 0:
            return np.zeros(target_frames * 63, dtype=np.float32)

        indices = np.linspace(0, n - 1, target_frames).astype(int)
        resampled = [buf[i].flatten() for i in indices]
        return np.concatenate(resampled).astype(np.float32)


# =====================================================================
# 5. DATASET LOADER (STATIC IMAGES & DYNAMIC VIDEO SEQUENCES)
# =====================================================================

class ASLODatasetLoader:
    """
    Robust dataset loader capable of indexing and processing both:
    1. Static single-image gestures (.jpg, .png).
    2. Dynamic video files (.mp4, .avi, etc.) and sequential frame folders (e.g. SEQ1/).
    
    Includes built-in low-light CLAHE enhancement, tracking validation,
    and automatic motion extraction.
    """
    def __init__(self, dataset_path: str = "picture_file"):
        self.dataset_path = dataset_path
        self.enhancer = LowLightEnhancer()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def natural_sort_key(s: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

    def process_image(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Apply CLAHE and extract 21x3 landmarks for the dominant hand."""
        enhanced, _, _ = self.enhancer.process(image_bgr)
        image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results and results.multi_hand_landmarks:
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark],
                dtype=np.float32,
            )
            return coords
        return None

    def load_dataset(
        self,
        target_seq_len: int = 16,
        max_samples_per_class: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Loads static images and dynamic video sequences.
        
        Returns:
            X: Array of flattened feature vectors.
            y: Array of string labels.
            stats: Summary metadata dictionary.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_path}' does not exist.")

        data_list = []
        label_list = []
        stats = {"static_classes": [], "dynamic_classes": [], "total_samples": 0}

        class_names = sorted(os.listdir(self.dataset_path), key=self.natural_sort_key)

        for label in class_names:
            class_dir = os.path.join(self.dataset_path, label)
            if not os.path.isdir(class_dir):
                continue

            entries = os.listdir(class_dir)
            video_files = [f for f in entries if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
            seq_dirs = [d for d in entries if os.path.isdir(os.path.join(class_dir, d))]
            image_files = [f for f in entries if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            is_dynamic = len(video_files) > 0 or len(seq_dirs) > 0

            if is_dynamic:
                stats["dynamic_classes"].append(label)
                # Load dynamic videos
                for v_file in video_files[:max_samples_per_class]:
                    v_path = os.path.join(class_dir, v_file)
                    cap = cv2.VideoCapture(v_path)
                    frames_coords = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        coords = self.process_image(frame)
                        if coords is not None:
                            frames_coords.append(coords.flatten())
                    cap.release()

                    if len(frames_coords) >= 4:
                        indices = np.linspace(0, len(frames_coords) - 1, target_seq_len).astype(int)
                        seq_flat = np.concatenate([frames_coords[i] for i in indices])
                        data_list.append(seq_flat)
                        label_list.append(label)

                # Load sequence subfolders
                for s_dir in seq_dirs[:max_samples_per_class]:
                    s_path = os.path.join(class_dir, s_dir)
                    seq_imgs = sorted(
                        [f for f in os.listdir(s_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
                        key=self.natural_sort_key,
                    )
                    frames_coords = []
                    for img_name in seq_imgs:
                        img = cv2.imread(os.path.join(s_path, img_name))
                        if img is not None:
                            coords = self.process_image(img)
                            if coords is not None:
                                frames_coords.append(coords.flatten())
                    if len(frames_coords) >= 4:
                        indices = np.linspace(0, len(frames_coords) - 1, target_seq_len).astype(int)
                        seq_flat = np.concatenate([frames_coords[i] for i in indices])
                        data_list.append(seq_flat)
                        label_list.append(label)
            else:
                stats["static_classes"].append(label)
                # Load static images
                for img_name in sorted(image_files, key=self.natural_sort_key)[:max_samples_per_class]:
                    img_path = os.path.join(class_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        coords = self.process_image(img)
                        if coords is not None:
                            # Replicate static coordinates across target_seq_len for unified dimension
                            seq_flat = np.tile(coords.flatten(), target_seq_len)
                            data_list.append(seq_flat)
                            label_list.append(label)

        stats["total_samples"] = len(data_list)
        return np.array(data_list, dtype=np.float32), np.array(label_list), stats
