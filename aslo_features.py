"""
ASLO True 3D Hand-Shape & Multi-MLP Feature Engine (`aslo_features.py`)
======================================================================
Single source of truth for both training and live inference.
Exports:
- `EXPECTED_SHAPE_DIM` (115): Rotation-invariant 3D hand-shape fingerprint in canonical palm basis.
- `EXPECTED_ASL_DIM` (124): `extract_hand_shape_features` (115) + 3D palm orientation basis (9).
- `EXPECTED_WORD_DIM` (252): Dual-hand shape & orientation features (2 * 124) + inter-hand offset & distance (4).
- `EXPECTED_DYNAMIC_DIM` (393): 3 keyframe hand-shape fingerprints (3 * 115 = 345) + global trajectory & velocity envelope (48).
"""

import collections
import itertools
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# =====================================================================
# FEATURE DIMENSION CONSTANTS & SINGLE SOURCE OF TRUTH
# =====================================================================

EXPECTED_SHAPE_DIM = 115
EXPECTED_ASL_DIM = EXPECTED_SHAPE_DIM + 9  # 124
EXPECTED_WORD_DIM = 2 * EXPECTED_ASL_DIM + 4  # 252
SEQ_LEN = 25
EXPECTED_DYNAMIC_DIM = 3 * EXPECTED_SHAPE_DIM + 48  # 393

# Legacy compatibility constants for raw 310-dim CSV storage
FIST_GEOMETRY_LEN = 42
FEATURE_LEN = 226 + 2 * FIST_GEOMETRY_LEN  # 310
EXPECTED_FEATURE_DIM = FEATURE_LEN
EXPECTED_STATIC_DIM = EXPECTED_ASL_DIM

# Anatomical Hand-Shape Families for Cooperative Two-Stage Static ASL MLPs
ASL_SHAPE_FAMILIES = {
    "CLUSTER_FIST": ["A", "S", "T", "N", "M", "E"],
    "CLUSTER_UPRIGHT": ["D", "K", "R", "U", "V", "H", "1", "2"],
    "CLUSTER_CURVE_PINCH": ["C", "O", "F", "G", "P", "Q", "X"],
    "CLUSTER_OPEN": ["B", "W", "4", "5", "3"],
}
FIST_CLUSTER_SET = set(ASL_SHAPE_FAMILIES["CLUSTER_FIST"])
FIST_CLUSTER_LABEL = "CLUSTER_FIST"


class LandmarkPoint:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class HandLandmarks:
    """Wraps raw landmark points (list of dicts, tuples, or (21, 3)/(63,) arrays) into
    a MediaPipe-compatible object exposing a .landmark attribute with 21 LandmarkPoint items."""

    def __init__(self, raw_points):
        self.landmark = []
        if raw_points is not None:
            if hasattr(raw_points, "landmark"):
                for lm in raw_points.landmark[:21]:
                    self.landmark.append(LandmarkPoint(lm.x, lm.y, getattr(lm, "z", 0.0)))
            else:
                arr_check = np.asarray(raw_points) if not isinstance(raw_points, list) else None
                if arr_check is not None and arr_check.ndim == 1 and arr_check.size == 63:
                    raw_points = arr_check.reshape(21, 3)
                if len(raw_points) > 0:
                    for p in raw_points[:21]:
                        if isinstance(p, dict):
                            self.landmark.append(LandmarkPoint(p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)))
                        elif isinstance(p, (list, tuple, np.ndarray)):
                            x = p[0] if len(p) > 0 else 0.0
                            y = p[1] if len(p) > 1 else 0.0
                            z = p[2] if len(p) > 2 else 0.0
                            self.landmark.append(LandmarkPoint(x, y, z))
                        elif hasattr(p, "x") and hasattr(p, "y"):
                            self.landmark.append(LandmarkPoint(p.x, p.y, getattr(p, "z", 0.0)))
                        else:
                            self.landmark.append(LandmarkPoint(0.0, 0.0, 0.0))
        while len(self.landmark) < 21:
            self.landmark.append(LandmarkPoint(0.0, 0.0, 0.0))


def _coords(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, getattr(lm, "z", 0.0)], dtype=np.float32)


def _to_21x3_array(hand_input) -> np.ndarray:
    """Converts HandLandmarks, list of dicts, or numpy array into (21, 3) float32 array."""
    if hand_input is None:
        return np.zeros((21, 3), dtype=np.float32)
    if hasattr(hand_input, "landmark"):
        pts = [[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in hand_input.landmark[:21]]
        while len(pts) < 21:
            pts.append([0.0, 0.0, 0.0])
        return np.asarray(pts, dtype=np.float32)
    if isinstance(hand_input, dict):
        inner = hand_input.get("points") or hand_input.get("landmarks") or []
        return _to_21x3_array(inner)
    if isinstance(hand_input, (list, tuple)):
        if len(hand_input) == 0:
            return np.zeros((21, 3), dtype=np.float32)
        if isinstance(hand_input[0], dict):
            pts = [[float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))] for p in hand_input[:21]]
            while len(pts) < 21:
                pts.append([0.0, 0.0, 0.0])
            return np.asarray(pts, dtype=np.float32)
    arr = np.asarray(hand_input, dtype=np.float32)
    if arr.size == 63:
        return arr.reshape(21, 3)
    if arr.ndim == 2 and arr.shape[0] >= 21 and arr.shape[1] >= 2:
        if arr.shape[1] == 2:
            return np.hstack([arr[:21, :2], np.zeros((21, 1), dtype=np.float32)])
        return arr[:21, :3].astype(np.float32)
    return np.zeros((21, 3), dtype=np.float32)


# =====================================================================
# PART 1.1: CANONICAL 3D PALM COORDINATE FRAME (ROTATION-INVARIANT)
# =====================================================================

def compute_canonical_palm_frame(pts_21x3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Projects 21 hand landmarks into a rotation-, translation-, and scale-invariant
    orthonormal 3D local palm coordinate basis:
    1. Translate relative to Wrist (0): P'_i = P_i - P_0
    2. Scale by palm size: S = ||P'_9|| + 1e-6
    3. Orthonormal local palm basis (x_palm, y_palm, z_palm):
       - y_palm = normalize(P'_9) (wrist to middle MCP = palm 'up' axis)
       - v_across = P'_5 - P'_17 (pinky MCP to index MCP)
       - z_palm = normalize(v_across x y_palm) (palm normal vector)
       - x_palm = normalize(y_palm x z_palm)
    4. Project all 21 scaled landmarks onto basis:
       local_pts = scaled_pts @ np.stack([x_palm, y_palm, z_palm], axis=1)

    Returns:
        local_pts: (21, 3) float32 palm-aligned coordinates
        basis_matrix: (3, 3) float32 columns [x_palm, y_palm, z_palm]
        scaled_pts: (21, 3) float32 wrist-centered, palm-scaled camera-frame points
        scale: float palm scale S
    """
    pts = np.asarray(pts_21x3, dtype=np.float32).reshape(21, 3)
    if np.all(np.abs(pts) < 1e-7):
        return (
            np.zeros((21, 3), dtype=np.float32),
            np.eye(3, dtype=np.float32),
            np.zeros((21, 3), dtype=np.float32),
            1.0,
        )

    p_prime = pts - pts[0:1, :]
    scale = float(np.linalg.norm(p_prime[9])) + 1e-6
    scaled_pts = p_prime / scale

    # 1. y_palm: wrist (0) -> middle MCP (9)
    y_norm = float(np.linalg.norm(p_prime[9]))
    if y_norm < 1e-6:
        y_palm = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        y_palm = (p_prime[9] / y_norm).astype(np.float32)

    # 2. v_across: pinky MCP (17) -> index MCP (5)
    v_across = (p_prime[5] - p_prime[17]).astype(np.float32)

    # 3. z_palm = normalize(v_across x y_palm)
    z_raw = np.cross(v_across, y_palm)
    z_norm = float(np.linalg.norm(z_raw))
    if z_norm < 1e-6:
        # Fallback orthogonal vector if collinear
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(y_palm[2]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        z_raw = np.cross(fallback, y_palm)
        z_norm = float(np.linalg.norm(z_raw)) + 1e-6
    z_palm = (z_raw / z_norm).astype(np.float32)

    # 4. x_palm = normalize(y_palm x z_palm)
    x_raw = np.cross(y_palm, z_palm)
    x_norm = float(np.linalg.norm(x_raw)) + 1e-6
    x_palm = (x_raw / x_norm).astype(np.float32)

    basis_matrix = np.stack([x_palm, y_palm, z_palm], axis=1).astype(np.float32)
    local_pts = (scaled_pts @ basis_matrix).astype(np.float32)
    return local_pts, basis_matrix, scaled_pts, scale


def _angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = float(np.dot(v1, v2) / (n1 * n2))
    dot = max(min(dot, 1.0), -1.0)
    return float(np.arccos(dot))


def _vec_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = float(np.dot(v1, v2) / (n1 * n2))
    dot = max(min(dot, 1.0), -1.0)
    return float(np.arccos(dot))


# =====================================================================
# PART 1.2: EXPLICIT HAND-SHAPE FINGERPRINT (`extract_hand_shape_features`)
# =====================================================================

def extract_hand_shape_features(hand_landmarks) -> np.ndarray:
    """
    Computes the 115-dim rotation-invariant 3D hand-shape fingerprint:
    1. Flattened palm-aligned 3D coordinates (`local_pts`, 63 values).
    2. Per-Finger Extension & Curl State (All 5 Fingers):
       - 15 3D flexion angles at every articulated joint:
         Thumb (0-1-2, 1-2-3, 2-3-4), Index (0-5-6, 5-6-7, 6-7-8),
         Middle (0-9-10, 9-10-11, 10-11-12), Ring (0-13-14, 13-14-15, 14-15-16),
         Pinky (0-17-18, 17-18-19, 18-19-20).
       - 5 Extension ratios: dist(tip, wrist) / (dist(mcp, wrist) + 1e-6) for (4/2, 8/5, 12/9, 16/13, 20/17).
    3. Finger Spread & Crossing Geometry (separates U/V/R/W/B):
       - 3 Adjacent fingertip distances: (8,12), (12,16), (16,20).
       - 3 Angles between adjacent finger vectors: (5->8 vs 9->12), (9->12 vs 13->16), (13->16 vs 17->20).
       - 4 Signed local X-crossing offsets: (local_pts[8,0]-local_pts[12,0]), (local_pts[8,1]-local_pts[12,1]),
         (local_pts[12,0]-local_pts[16,0]), (local_pts[16,0]-local_pts[20,0]).
    4. Pinch & Closed-Fist Thumb-Slot Geometry:
       - 4 Thumb Tip (4) distances to other fingertips (8, 12, 16, 20) (separates O/C/F/D).
       - 18 Signed local palm-frame offsets (dx, dy, dz) from Thumb Tip (4) to:
         Index MCP (5), Index PIP (6), Middle MCP (9), Middle PIP (10), Ring MCP (13), Pinky MCP (17)
         (6 joints * 3 = 18 values, separates A/S/T/N/M/E).
    Total = 63 + 15 + 5 + 3 + 3 + 4 + 4 + 18 = 115 values (`EXPECTED_SHAPE_DIM`).
    """
    pts = _to_21x3_array(hand_landmarks)
    if np.all(np.abs(pts) < 1e-7):
        return np.zeros(EXPECTED_SHAPE_DIM, dtype=np.float32)

    local_pts, _, _, _ = compute_canonical_palm_frame(pts)

    # 1. Palm-aligned 3D coordinates (63 values)
    feats = local_pts.flatten().tolist()

    # 2A. 15 3D joint flexion angles (rotation-invariant)
    joint_triplets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),         # Thumb
        (0, 5, 6), (5, 6, 7), (6, 7, 8),         # Index
        (0, 9, 10), (9, 10, 11), (10, 11, 12),   # Middle
        (0, 13, 14), (13, 14, 15), (14, 15, 16), # Ring
        (0, 17, 18), (17, 18, 19), (18, 19, 20), # Pinky
    ]
    for a, b, c in joint_triplets:
        feats.append(_angle(local_pts[a], local_pts[b], local_pts[c]))

    # 2B. 5 Extension ratios: dist(tip, wrist) / (dist(mcp, wrist) + 1e-6)
    w0 = local_pts[0]
    finger_tip_mcp = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]
    for tip_i, mcp_i in finger_tip_mcp:
        d_tip = float(np.linalg.norm(local_pts[tip_i] - w0))
        d_mcp = float(np.linalg.norm(local_pts[mcp_i] - w0)) + 1e-6
        feats.append(d_tip / d_mcp)

    # 3. Finger Spread & Crossing Geometry (10 values)
    d_8_12 = float(np.linalg.norm(local_pts[8] - local_pts[12]))
    d_12_16 = float(np.linalg.norm(local_pts[12] - local_pts[16]))
    d_16_20 = float(np.linalg.norm(local_pts[16] - local_pts[20]))
    feats.extend([d_8_12, d_12_16, d_16_20])

    v_idx = local_pts[8] - local_pts[5]
    v_mid = local_pts[12] - local_pts[9]
    v_rng = local_pts[16] - local_pts[13]
    v_pky = local_pts[20] - local_pts[17]
    feats.extend([
        _vec_angle(v_idx, v_mid),
        _vec_angle(v_mid, v_rng),
        _vec_angle(v_rng, v_pky),
    ])

    # Dimensionless 3D rotation-invariant U vs V vs R crossing & spread invariants in canonical palm frame:
    # 1) lat_tip_ratio: (x_8 - x_12) / (x_5 - x_9) -> R < 0.42 (crossed), U in [0.50, 1.30] (parallel), V > 1.45 (spread)
    # 2) lat_dip_ratio: (x_7 - x_11) / (x_5 - x_9) -> R < 0.58, U in [0.62, 1.28], V > 1.35
    # 3) spread_tip_mcp_ratio: ||P_8 - P_12|| / ||P_5 - P_9|| -> R ~ 0.40, U ~ 1.03, V ~ 2.14
    # 4) signed_cross_xy: signed 2D sine between index (5->8) and middle (9->12) in canonical palm XY plane -> R < -0.06, U ~ 0.0, V > +0.06
    dx_mcp = float(local_pts[5, 0] - local_pts[9, 0])
    dx_mcp_safe = dx_mcp if abs(dx_mcp) >= 0.04 else (0.04 if dx_mcp >= 0.0 else -0.04)
    lat_tip_ratio = float(np.clip((local_pts[8, 0] - local_pts[12, 0]) / dx_mcp_safe, -4.0, 4.0))
    lat_dip_ratio = float(np.clip((local_pts[7, 0] - local_pts[11, 0]) / dx_mcp_safe, -4.0, 4.0))
    d_5_9 = float(np.linalg.norm(local_pts[5] - local_pts[9])) + 1e-6
    spread_tip_mcp_ratio = float(np.clip(d_8_12 / d_5_9, 0.0, 6.0))
    v_idx_2d = local_pts[8, :2] - local_pts[5, :2]
    v_mid_2d = local_pts[12, :2] - local_pts[9, :2]
    denom_2d = float(np.linalg.norm(v_idx_2d) * np.linalg.norm(v_mid_2d)) + 1e-6
    hand_sign = 1.0 if dx_mcp >= 0.0 else -1.0
    signed_cross_xy = float(((v_idx_2d[0] * v_mid_2d[1] - v_idx_2d[1] * v_mid_2d[0]) / denom_2d) * hand_sign)

    feats.extend([
        lat_tip_ratio,
        lat_dip_ratio,
        spread_tip_mcp_ratio,
        signed_cross_xy,
    ])

    # 4A. Pinch distances: Thumb Tip (4) to all 4 other fingertips (8, 12, 16, 20) (4 values)
    p4 = local_pts[4]
    for tip_i in (8, 12, 16, 20):
        feats.append(float(np.linalg.norm(p4 - local_pts[tip_i])))

    # 4B. Signed local palm-frame offsets (dx, dy, dz) from Thumb Tip (4) to knuckles (5, 6, 9, 10, 13, 17) (18 values)
    for kn_i in (5, 6, 9, 10, 13, 17):
        diff = p4 - local_pts[kn_i]
        feats.extend([float(diff[0]), float(diff[1]), float(diff[2])])

    arr = np.asarray(feats, dtype=np.float32)
    assert arr.shape == (EXPECTED_SHAPE_DIM,), f"Expected shape ({EXPECTED_SHAPE_DIM},), got {arr.shape}"
    return arr


# =====================================================================
# PART 1.3: SEPARATE FEATURE EXTRACTORS FOR EACH GESTURE TYPE
# =====================================================================

def extract_asl_features(hand_landmarks) -> np.ndarray:
    """
    Extracts 124-dim feature vector for Static ASL (Stage 1 & Stage 2 Specialists):
    - `extract_hand_shape_features` (115 dims, rotation-invariant hand shape)
    - Camera-frame key pointing vectors `[scaled_pts[4], scaled_pts[8], scaled_pts[9]]` flattened (9 dims:
      Thumb Tip 4, Index Tip 8, and Middle MCP 9 in wrist-relative palm-scaled camera coordinates)
      so orientation-dependent signs (G vs Q, H vs U, K vs P, and Start/Stop/Space/Backspace vs A/D/L/G)
      are strictly separated.
    """
    pts = _to_21x3_array(hand_landmarks)
    if np.all(np.abs(pts) < 1e-7):
        return np.zeros(EXPECTED_ASL_DIM, dtype=np.float32)

    shape_feats = extract_hand_shape_features(pts)
    _, _, scaled_pts, _ = compute_canonical_palm_frame(pts)
    orient_feats = np.concatenate([scaled_pts[4], scaled_pts[8], scaled_pts[9]]).astype(np.float32)  # 9 values

    out = np.concatenate([shape_feats, orient_feats]).astype(np.float32)
    assert out.shape == (EXPECTED_ASL_DIM,), f"Expected ({EXPECTED_ASL_DIM},), got {out.shape}"
    return out


def _extract_word_hand_features(pts_21x3: np.ndarray) -> np.ndarray:
    """
    Extracts 124-dim feature vector for a single hand in Static Words (`words_static_mlp.keras`):
    - 2D-span normalized, Z-damped camera-frame wrist-centered 3D coordinates `word_scaled_pts.flatten()` (63 dims),
      making foreshortened thumbs-up (`picture_file/start`) and live-webcam thumbs-up (`👍 👍`) identical
      while eliminating noisy MediaPipe Z-depth drift between static photos and live video.
    - Rotation-invariant anatomical hand-shape features `extract_hand_shape_features(pts)[63:115]` (52 dims:
      15 joint angles + 5 extension ratios + 10 finger spread/crossing features + 4 pinch distances + 18 thumb slots).
    - 2D foreshortening- & depth-invariant pointing vectors & vertical offsets (9 dims, indices 115:124):
      * `[u_t_2d_x, u_t_2d_y, dy_4_8]` (thumb 0->4 2D unit direction + vertical thumb-above-index offset)
      * `[u42_x, u42_y, dy_4_5]` (thumb bone 2->4 2D unit direction + vertical thumb-above-knuckle offset)
      * `[u_i_2d_x, u_i_2d_y, dy_8_5]` (index 0->8 2D unit direction + vertical index-below-knuckle offset)
    Total = 63 + 52 + 9 = 124 dims (`EXPECTED_ASL_DIM`).
    """
    pts = _to_21x3_array(pts_21x3)
    if np.all(np.abs(pts) < 1e-7):
        return np.zeros(EXPECTED_ASL_DIM, dtype=np.float32)

    shape_feats = extract_hand_shape_features(pts)
    p_prime = pts - pts[0:1, :]
    max_span_xy = float(np.max(np.linalg.norm(p_prime[:, :2], axis=1)))
    max_span_3d = float(np.max(np.linalg.norm(p_prime, axis=1)))
    robust_scale = max(max_span_xy, 0.35 * max_span_3d) + 1e-6
    word_scaled_pts = (p_prime / robust_scale).astype(np.float32)
    word_scaled_pts[:, 2] *= 0.30

    n4_2d = float(np.linalg.norm(p_prime[4, :2])) + 1e-6
    u_t_2d_x = float(p_prime[4, 0] / n4_2d)
    u_t_2d_y = float(p_prime[4, 1] / n4_2d)
    dy_4_8 = float((p_prime[4, 1] - p_prime[8, 1]) / robust_scale)

    v42_2d = p_prime[4, :2] - p_prime[2, :2]
    n42_2d = float(np.linalg.norm(v42_2d)) + 1e-6
    u42_x = float(v42_2d[0] / n42_2d)
    u42_y = float(v42_2d[1] / n42_2d)
    dy_4_5 = float((p_prime[4, 1] - p_prime[5, 1]) / robust_scale)

    n8_2d = float(np.linalg.norm(p_prime[8, :2])) + 1e-6
    u_i_2d_x = float(p_prime[8, 0] / n8_2d)
    u_i_2d_y = float(p_prime[8, 1] / n8_2d)
    dy_8_5 = float((p_prime[8, 1] - p_prime[5, 1]) / robust_scale)

    orient_feats = np.array(
        [u_t_2d_x, u_t_2d_y, dy_4_8, u42_x, u42_y, dy_4_5, u_i_2d_x, u_i_2d_y, dy_8_5],
        dtype=np.float32,
    )
    out = np.concatenate([word_scaled_pts.flatten(), shape_feats[63:], orient_feats]).astype(np.float32)
    assert out.shape == (EXPECTED_ASL_DIM,), f"Expected ({EXPECTED_ASL_DIM},), got {out.shape}"
    return out


def extract_static_word_features(hands, hand_type: str = "right") -> np.ndarray:
    """
    Extracts 252-dim feature vector for Static Words (`EXPECTED_WORD_DIM = 252`):
    - Hand 0 word features (camera-frame 3D coords + invariant shape angles/ratios + pointing vectors = 124 dims)
    - Hand 1 word features (124 dims; zero-padded when 1 hand)
    - Inter-hand relative orientation & offset vector (4 dims)
    """
    r_pts = np.zeros((21, 3), dtype=np.float32)
    l_pts = np.zeros((21, 3), dtype=np.float32)

    if isinstance(hands, np.ndarray) and hands.ndim == 1 and hands.size >= 126:
        r_pts = hands[0:63].reshape(21, 3).astype(np.float32)
        l_pts = hands[63:126].reshape(21, 3).astype(np.float32)
    elif isinstance(hands, tuple) and len(hands) == 2:
        r_pts = _to_21x3_array(hands[0])
        l_pts = _to_21x3_array(hands[1])
    elif isinstance(hands, list) and len(hands) > 0 and isinstance(hands[0], dict) and ("points" in hands[0] or "landmarks" in hands[0]):
        for h in hands[:2]:
            lbl = (h.get("label") or "Right").capitalize().strip()
            pts = _to_21x3_array(h.get("points") or h.get("landmarks") or [])
            if lbl == "Left" and np.all(np.abs(l_pts) < 1e-7):
                l_pts = pts
            elif np.all(np.abs(r_pts) < 1e-7):
                r_pts = pts
            else:
                l_pts = pts
    else:
        r_pts = _to_21x3_array(hands)

    has_r = not np.all(np.abs(r_pts) < 1e-7)
    has_l = not np.all(np.abs(l_pts) < 1e-7)

    # If only slot 1 (l_pts) has landmarks and slot 0 (r_pts) is zeros, promote l_pts to slot 0
    # so single-hand inputs ALWAYS occupy slot 0 (0:124) and never leave slot 0 all-zeros!
    if (not has_r) and has_l:
        r_pts = l_pts
        l_pts = np.zeros((21, 3), dtype=np.float32)
        has_r = True
        has_l = False

    if (hand_type or "").lower().strip() in ("right", "single") and not has_l:
        l_pts = np.zeros((21, 3), dtype=np.float32)

    r_feats = _extract_word_hand_features(r_pts) if has_r else np.zeros(EXPECTED_ASL_DIM, dtype=np.float32)
    l_feats = _extract_word_hand_features(l_pts) if has_l else np.zeros(EXPECTED_ASL_DIM, dtype=np.float32)

    # Inter-hand relative orientation & offset vector (4 dims)
    if has_r and has_l:
        r_vec9 = r_pts[9] - r_pts[0]
        l_vec9 = l_pts[9] - l_pts[0]
        r_u9 = r_vec9 / (float(np.linalg.norm(r_vec9)) + 1e-6)
        l_u9 = l_vec9 / (float(np.linalg.norm(l_vec9)) + 1e-6)
        diff = l_u9 - r_u9
        dist = float(np.linalg.norm(diff))
        inter_feats = np.array([float(diff[0]), float(diff[1]), float(diff[2]), dist], dtype=np.float32)
    else:
        inter_feats = np.zeros(4, dtype=np.float32)

    out = np.concatenate([r_feats, l_feats, inter_feats]).astype(np.float32)
    assert out.shape == (EXPECTED_WORD_DIM,), f"Expected ({EXPECTED_WORD_DIM},), got {out.shape}"
    return out


def normalize_sequence_length(sequence_frames, target_len: int = SEQ_LEN) -> np.ndarray:
    """
    Resamples or pads an arbitrary list of landmark frames into (target_len, 21, 3).
    Automatically trims leading waist/lap resting frames (wrist_y >= 0.78) and hand-swap teleport jumps
    (> 0.32 normalized units in a single frame) when raw camera coordinates are passed, so kf_start
    always aligns with the true active gesture start in signing space.
    """
    if sequence_frames is None or len(sequence_frames) == 0:
        return np.zeros((target_len, 21, 3), dtype=np.float32)
    frames = [_to_21x3_array(f) for f in sequence_frames]
    # Check if frames are in raw camera space (wrist [0] not centered at 0)
    if len(frames) >= 8 and float(np.max(np.abs([f[0, :2] for f in frames]))) > 1e-3:
        # 1. Trim hand-swap teleport jumps (> 0.32 in a single step) if at least 5 frames follow
        last_jump = 0
        for idx_f in range(1, len(frames)):
            step_d = float(np.linalg.norm(frames[idx_f][0, :2] - frames[idx_f - 1][0, :2]))
            if step_d > 0.32 and (len(frames) - idx_f) >= 5:
                last_jump = idx_f
        if last_jump > 0:
            frames = frames[last_jump:]
        # 2. Trim leading resting frames at bottom of camera view (wrist_y >= 0.78)
        first_active = 0
        while first_active < len(frames) - 5 and float(frames[first_active][0, 1]) >= 0.78:
            first_active += 1
        if first_active > 0:
            frames = frames[first_active:]

    n = len(frames)
    if n == target_len:
        return np.stack(frames, axis=0).astype(np.float32)
    indices = np.linspace(0, n - 1, target_len).astype(int)
    resampled = [frames[i] for i in indices]
    return np.stack(resampled, axis=0).astype(np.float32)


def extract_dynamic_mlp_features(sequence_buffer) -> np.ndarray:
    """
    Converts a 25-frame landmark sliding window into a flat 1D vector (`EXPECTED_DYNAMIC_DIM = 393`):
    1. Aligned hand-shape fingerprints (`extract_hand_shape_features`, 115 dims each) at:
       - Start (frame 0)
       - Middle (frame 12)
       - End (frame 24)
       -> 3 * 115 = 345 dims.
    2. Global wrist & fingertip trajectory & velocity envelope (48 dims):
       - Net wrist displacement (dx, dy, dz) from start to end (3 dims)
       - 5 evenly sampled normalized wrist path points [0, 6, 12, 18, 24] (5 * 3 = 15 dims)
       - Mean and max velocity (vx, vy, speed) for wrist (0), index tip (8), and pinky tip (20) (3 * 6 = 18 dims)
       - First-half vs second-half directional velocity (vx1, vy1, vx2, vy2) for index tip (8) & pinky tip (20) (2 * 4 = 8 dims)
       - Path bounding box (total path length, width, height, aspect ratio) (4 dims)
    Total = 345 + 48 = 393 dims.
    """
    seq = normalize_sequence_length(sequence_buffer, target_len=SEQ_LEN)  # (25, 21, 3)

    palm_scales = np.linalg.norm(seq[:, 9, :] - seq[:, 0, :], axis=1)
    valid_scales = palm_scales[palm_scales > 1e-4]
    avg_scale = float(np.median(valid_scales)) if len(valid_scales) > 0 else 1.0
    avg_scale = max(avg_scale, 1e-4)

    # 1. Key-Moment Aligned Hand-Shape Fingerprints (Start=0, Middle=12, End=24): 3 * 115 = 345 dims
    kf_start = extract_hand_shape_features(seq[0])
    kf_mid = extract_hand_shape_features(seq[SEQ_LEN // 2])
    kf_end = extract_hand_shape_features(seq[SEQ_LEN - 1])

    traj_feats = []

    # 2A. Global wrist net displacement (3 dims) + 5 sampled wrist path points (15 dims) = 18 dims
    w_start = seq[0, 0, :]
    w_end = seq[-1, 0, :]
    net_disp = ((w_end - w_start) / avg_scale).tolist()
    traj_feats.extend(net_disp)

    sample_idxs = [0, 6, 12, 18, 24]
    for idx in sample_idxs:
        w_off = ((seq[idx, 0, :] - w_start) / avg_scale).tolist()
        traj_feats.extend(w_off)

    # 2B. Mean & max velocity for wrist (0), index tip (8), pinky tip (20): 3 * 6 = 18 dims
    for j_idx in (0, 8, 20):
        path = (seq[:, j_idx, :2] - seq[0, j_idx, :2]) / avg_scale
        diffs = np.diff(path, axis=0)
        vx = diffs[:, 0]
        vy = diffs[:, 1]
        speeds = np.hypot(vx, vy)
        traj_feats.extend([
            float(np.mean(vx)),
            float(np.mean(vy)),
            float(np.mean(speeds)),
            float(np.max(np.abs(vx))),
            float(np.max(np.abs(vy))),
            float(np.max(speeds)),
        ])

    # 2C. First-half vs Second-half velocity for index tip (8) and pinky tip (20): 2 * 4 = 8 dims
    half = (SEQ_LEN - 1) // 2
    for j_idx in (8, 20):
        path = (seq[:, j_idx, :2] - seq[0, j_idx, :2]) / avg_scale
        diffs = np.diff(path, axis=0)
        v1 = np.mean(diffs[:half], axis=0)
        v2 = np.mean(diffs[half:], axis=0)
        traj_feats.extend([float(v1[0]), float(v1[1]), float(v2[0]), float(v2[1])])

    # 2D. Overall active trajectory bounding box & path length (4 dims)
    # Combine wrist + index tip + pinky tip envelope
    active_path = (seq[:, [0, 8, 20], :2] - seq[0:1, [0, 8, 20], :2]) / avg_scale
    wrist_diffs = np.diff(active_path[:, 0, :], axis=0)
    total_len = float(np.sum(np.hypot(wrist_diffs[:, 0], wrist_diffs[:, 1])))
    min_xy = np.min(active_path.reshape(-1, 2), axis=0)
    max_xy = np.max(active_path.reshape(-1, 2), axis=0)
    bw = float(max_xy[0] - min_xy[0])
    bh = float(max_xy[1] - min_xy[1])
    aspect = float(bw / (bh + 1e-4))
    traj_feats.extend([total_len, bw, bh, aspect])

    out = np.concatenate([kf_start, kf_mid, kf_end, np.asarray(traj_feats, dtype=np.float32)]).astype(np.float32)
    assert out.shape == (EXPECTED_DYNAMIC_DIM,), f"Expected ({EXPECTED_DYNAMIC_DIM},), got {out.shape}"
    return out


# =====================================================================
# VALIDATORS & BACKWARD COMPATIBILITY HELPERS
# =====================================================================

def validate_shape_vector(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        assert arr.shape == (EXPECTED_SHAPE_DIM,), f"Shape mismatch: {arr.shape} != ({EXPECTED_SHAPE_DIM},)"
        arr = np.expand_dims(arr, axis=0)
    assert arr.shape == (1, EXPECTED_SHAPE_DIM), f"Shape mismatch: {arr.shape} != (1, {EXPECTED_SHAPE_DIM})"
    return arr


def validate_asl_feature_vector(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        assert arr.shape == (EXPECTED_ASL_DIM,), f"ASL feature dimension mismatch: expected ({EXPECTED_ASL_DIM},), got {arr.shape}"
        arr = np.expand_dims(arr, axis=0)
    assert arr.shape == (1, EXPECTED_ASL_DIM), f"ASL feature dimension mismatch: expected (1, {EXPECTED_ASL_DIM}), got {arr.shape}"
    assert not np.isnan(arr).any() and not np.isinf(arr).any()
    return arr


def validate_word_feature_vector(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        assert arr.shape == (EXPECTED_WORD_DIM,), f"Word feature dimension mismatch: expected ({EXPECTED_WORD_DIM},), got {arr.shape}"
        arr = np.expand_dims(arr, axis=0)
    assert arr.shape == (1, EXPECTED_WORD_DIM), f"Word feature dimension mismatch: expected (1, {EXPECTED_WORD_DIM}), got {arr.shape}"
    assert not np.isnan(arr).any() and not np.isinf(arr).any()
    return arr


def validate_dynamic_feature_vector(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        assert arr.shape == (EXPECTED_DYNAMIC_DIM,), f"Dynamic feature dimension mismatch: expected ({EXPECTED_DYNAMIC_DIM},), got {arr.shape}"
        arr = np.expand_dims(arr, axis=0)
    assert arr.shape == (1, EXPECTED_DYNAMIC_DIM), f"Dynamic feature dimension mismatch: expected (1, {EXPECTED_DYNAMIC_DIM}), got {arr.shape}"
    assert not np.isnan(arr).any() and not np.isinf(arr).any()
    return arr


def csv_row_to_asl_features(row_310: np.ndarray) -> np.ndarray:
    """Converts a 310-dim CSV row (where indices 0:63 are wrist-normalized 3D landmarks) into 124-dim `extract_asl_features`."""
    pts = np.asarray(row_310[:63], dtype=np.float32).reshape(21, 3)
    return extract_asl_features(pts)


def csv_row_to_word_features(row_310: np.ndarray) -> np.ndarray:
    """Converts a 310-dim CSV row (indices 0:63 = Right hand, 63:126 = Left hand) into 252-dim `extract_static_word_features`."""
    return extract_static_word_features(np.asarray(row_310, dtype=np.float32))


# Legacy raw CSV extraction functions kept so collect_data.py continues working seamlessly
class _Results:
    def __init__(self, hands_res):
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        if hands_res and hands_res.multi_hand_landmarks:
            assigned_left = False
            assigned_right = False
            for lms, handed in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                mp_label = handed.classification[0].label
                if mp_label == "Left":
                    if not assigned_right:
                        self.right_hand_landmarks = lms
                        assigned_right = True
                    else:
                        self.left_hand_landmarks = lms
                        assigned_left = True
                else:
                    if not assigned_left:
                        self.left_hand_landmarks = lms
                        assigned_left = True
                    else:
                        self.right_hand_landmarks = lms
                        assigned_right = True


def _normalised_hand(hand_landmarks) -> list[float]:
    points = [_coords(lm) for lm in hand_landmarks.landmark]
    wrist = points[0]
    scale = float(np.linalg.norm(points[9] - points[0])) + 1e-6
    norm = []
    for pt in points:
        norm.extend(((pt - wrist) / scale).tolist())
    return norm


def _thumb_distances(hand_landmarks) -> list[float]:
    points = np.array([_coords(lm) for lm in hand_landmarks.landmark])
    thumb_tip = points[4]
    scale = float(np.linalg.norm(points[9] - points[0])) + 1e-6
    dists = np.linalg.norm(points - thumb_tip, axis=1) / scale
    return np.delete(dists, 4).tolist()


def _fingertip_distances(hand_landmarks) -> list[float]:
    points = np.array([_coords(lm) for lm in hand_landmarks.landmark])
    scale = float(np.linalg.norm(points[9] - points[0])) + 1e-6
    tips = np.array([4, 8, 12, 16, 20])
    tip_points = points[tips]
    pairs = list(itertools.combinations(range(5), 2))
    p1 = tip_points[[p[0] for p in pairs]]
    p2 = tip_points[[p[1] for p in pairs]]
    pairwise_dists = np.linalg.norm(p1 - p2, axis=1) / scale
    wrist_dists = np.linalg.norm(tip_points - points[0], axis=1) / scale
    return pairwise_dists.tolist() + wrist_dists.tolist()


def _joint_angles(hand_landmarks) -> list[float]:
    pts = [_coords(lm) for lm in hand_landmarks.landmark]
    triplets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        (0, 17, 18), (17, 18, 19), (18, 19, 20),
    ]
    return [_angle(pts[a], pts[b], pts[c]) for a, b, c in triplets]


def _fist_geometry_from_norm_coords(norm_pts: np.ndarray) -> list[float]:
    pts = norm_pts.astype(np.float32)
    v45 = pts[4] - pts[5]
    v46 = pts[4] - pts[6]
    v49 = pts[4] - pts[9]
    v_th = pts[4] - pts[1]
    v_idx = pts[6] - pts[5]
    u_th = v_th / (float(np.linalg.norm(v_th)) + 1e-6)
    u_idx = v_idx / (float(np.linalg.norm(v_idx)) + 1e-6)
    dot = float(np.clip(np.dot(u_th, u_idx), -1.0, 1.0))
    angle = float(np.arccos(dot))
    mid_59 = (pts[5] + pts[9]) * 0.5
    v_t_slot = pts[4] - mid_59
    mid_913 = (pts[9] + pts[13]) * 0.5
    v_slot_n = pts[4] - mid_913
    mid_1317 = (pts[13] + pts[17]) * 0.5
    v_slot_m = pts[4] - mid_1317
    d_n = float(np.linalg.norm(v_slot_n))
    d_m = float(np.linalg.norm(v_slot_m))
    return [
        float(np.linalg.norm(v45)), float(np.linalg.norm(v46)), float(np.linalg.norm(v49)),
        float(np.linalg.norm(pts[4] - pts[10])), float(np.linalg.norm(pts[4] - pts[13])), float(np.linalg.norm(pts[4] - pts[8])),
        float(v45[0]), float(v45[1]), float(v45[2]),
        float(v46[0]), float(v46[1]), float(v46[2]),
        float(v49[0]), float(v49[1]), float(v49[2]),
        float(v_th[0]), float(v_th[1]), float(v_th[2]),
        dot, angle,
        float(np.linalg.norm(pts[8] - pts[0])), float(np.linalg.norm(pts[12] - pts[0])),
        float(np.linalg.norm(pts[16] - pts[0])), float(np.linalg.norm(pts[20] - pts[0])),
        float(v_t_slot[0]), float(v_t_slot[1]), float(v_t_slot[2]), float(np.linalg.norm(v_t_slot)),
        float(pts[4][1] - pts[6][1]), float(pts[4][1] - pts[10][1]), float(pts[4][1] - (pts[6][1] + pts[10][1]) * 0.5),
        float(v_slot_n[0]), float(v_slot_n[1]), float(v_slot_n[2]), d_n,
        float(v_slot_m[0]), float(v_slot_m[1]), float(v_slot_m[2]), d_m,
        float(pts[4][0] - pts[13][0]), float(pts[4][1] - pts[13][1]), float(d_n - d_m),
    ]


def _fist_disambiguation_features(hand_landmarks) -> list[float]:
    pts = np.array([_coords(lm) for lm in hand_landmarks.landmark], dtype=np.float32)
    scale = float(np.linalg.norm(pts[9] - pts[0])) + 1e-6
    return _fist_geometry_from_norm_coords((pts - pts[0]) / scale)


def _fist_disambiguation_from_norm_coords(coords_63) -> list[float]:
    norm_pts = np.asarray(coords_63, dtype=np.float32).reshape(21, 3)
    if np.all(norm_pts == 0):
        return [0.0] * FIST_GEOMETRY_LEN
    return _fist_geometry_from_norm_coords(norm_pts)


def extract_features(hands_res) -> list[float]:
    results = _Results(hands_res)
    feats = []
    feats += _normalised_hand(results.left_hand_landmarks) if results.left_hand_landmarks else [0.0] * 63
    feats += _normalised_hand(results.right_hand_landmarks) if results.right_hand_landmarks else [0.0] * 63
    if results.left_hand_landmarks:
        feats += _thumb_distances(results.left_hand_landmarks) + _joint_angles(results.left_hand_landmarks) + _fingertip_distances(results.left_hand_landmarks)
    else:
        feats += [0.0] * 50
    if results.right_hand_landmarks:
        feats += _thumb_distances(results.right_hand_landmarks) + _joint_angles(results.right_hand_landmarks) + _fingertip_distances(results.right_hand_landmarks)
    else:
        feats += [0.0] * 50
    feats += _fist_disambiguation_features(results.left_hand_landmarks) if results.left_hand_landmarks else [0.0] * FIST_GEOMETRY_LEN
    feats += _fist_disambiguation_features(results.right_hand_landmarks) if results.right_hand_landmarks else [0.0] * FIST_GEOMETRY_LEN
    return feats


def extract_primary_hand_features(hands_res, dominant_hand="right") -> list[float]:
    results = _Results(hands_res)
    if results.left_hand_landmarks and results.right_hand_landmarks:
        if dominant_hand == "left":
            results.left_hand_landmarks = None
        else:
            results.right_hand_landmarks = None
    feats = []
    feats += _normalised_hand(results.left_hand_landmarks) if results.left_hand_landmarks else [0.0] * 63
    feats += _normalised_hand(results.right_hand_landmarks) if results.right_hand_landmarks else [0.0] * 63
    if results.left_hand_landmarks:
        feats += _thumb_distances(results.left_hand_landmarks) + _joint_angles(results.left_hand_landmarks) + _fingertip_distances(results.left_hand_landmarks)
    else:
        feats += [0.0] * 50
    if results.right_hand_landmarks:
        feats += _thumb_distances(results.right_hand_landmarks) + _joint_angles(results.right_hand_landmarks) + _fingertip_distances(results.right_hand_landmarks)
    else:
        feats += [0.0] * 50
    feats += _fist_disambiguation_features(results.left_hand_landmarks) if results.left_hand_landmarks else [0.0] * FIST_GEOMETRY_LEN
    feats += _fist_disambiguation_features(results.right_hand_landmarks) if results.right_hand_landmarks else [0.0] * FIST_GEOMETRY_LEN
    return feats
