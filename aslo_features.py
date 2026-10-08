import collections
import itertools
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

FIST_GEOMETRY_LEN = 24
FEATURE_LEN = 226 + 2 * FIST_GEOMETRY_LEN  # 274


class LandmarkPoint:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class HandLandmarks:
    """Wraps raw landmark points (list of dicts, tuples, or [x, y, z] lists) into
    a MediaPipe-compatible object exposing a .landmark attribute with 21 LandmarkPoint items."""

    def __init__(self, raw_points):
        self.landmark = []
        if raw_points is not None and len(raw_points) > 0:
            for p in raw_points:
                if isinstance(p, dict):
                    self.landmark.append(LandmarkPoint(p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)))
                elif isinstance(p, (list, tuple)):
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


class _Results:
    def __init__(self, hands_res):
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        if hands_res and hands_res.multi_hand_landmarks:
            assigned_left = False
            assigned_right = False

            for lms, handed in zip(
                hands_res.multi_hand_landmarks, hands_res.multi_handedness
            ):
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


def _coords(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, getattr(lm, "z", 0.0)], dtype=np.float32)


def _normalise_points(points, origin, scale):
    if scale < 1e-6:
        scale = 1.0
    normalised = []
    for pt in points:
        normalised.extend(((pt - origin) / scale).tolist())
    return normalised


def _normalised_hand(hand_landmarks) -> list[float]:
    points = [_coords(lm) for lm in hand_landmarks.landmark]
    wrist = points[0]
    scale = float(np.linalg.norm(points[9] - points[0])) + 1e-6
    return _normalise_points(points, wrist, scale)


def _thumb_distances(hand_landmarks) -> list[float]:
    points = np.array([_coords(lm) for lm in hand_landmarks.landmark])
    thumb_tip = points[4]
    scale = float(np.linalg.norm(points[9] - points[0])) + 1e-6
    diffs = points - thumb_tip
    dists = np.linalg.norm(diffs, axis=1) / scale
    dists = np.delete(dists, 4)
    return dists.tolist()


def _fingertip_distances(hand_landmarks) -> list[float]:
    points = np.array([_coords(lm) for lm in hand_landmarks.landmark])
    scale = float(np.linalg.norm(points[9] - points[0])) + 1e-6
        
    tips = np.array([4, 8, 12, 16, 20])
    tip_points = points[tips]
    
    pairs = list(itertools.combinations(range(5), 2))
    p1 = tip_points[[p[0] for p in pairs]]
    p2 = tip_points[[p[1] for p in pairs]]
    pairwise_dists = np.linalg.norm(p1 - p2, axis=1) / scale
    
    wrist = points[0]
    wrist_dists = np.linalg.norm(tip_points - wrist, axis=1) / scale
    
    return pairwise_dists.tolist() + wrist_dists.tolist()


def _angle(p1, p2, p3) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    dot = np.dot(v1, v2) / (v1_norm * v2_norm)
    dot = max(min(dot, 1.0), -1.0)
    return float(np.arccos(dot))


def _joint_angles(hand_landmarks) -> list[float]:
    pts = [_coords(lm) for lm in hand_landmarks.landmark]
    angles = []
    angles.append(_angle(pts[0], pts[1], pts[2]))
    angles.append(_angle(pts[1], pts[2], pts[3]))
    angles.append(_angle(pts[2], pts[3], pts[4]))

    angles.append(_angle(pts[0], pts[5], pts[6]))
    angles.append(_angle(pts[5], pts[6], pts[7]))
    angles.append(_angle(pts[6], pts[7], pts[8]))

    angles.append(_angle(pts[0], pts[9], pts[10]))
    angles.append(_angle(pts[9], pts[10], pts[11]))
    angles.append(_angle(pts[10], pts[11], pts[12]))

    angles.append(_angle(pts[0], pts[13], pts[14]))
    angles.append(_angle(pts[13], pts[14], pts[15]))
    angles.append(_angle(pts[14], pts[15], pts[16]))

    angles.append(_angle(pts[0], pts[17], pts[18]))
    angles.append(_angle(pts[17], pts[18], pts[19]))
    angles.append(_angle(pts[18], pts[19], pts[20]))
    return angles


def _fist_geometry_from_norm_coords(norm_pts: np.ndarray) -> list[float]:
    """Computes explicit 24-dim thumb-to-finger relational geometry:
    1. Distances and vectors from Thumb Tip (4) to Index MCP (5), Index PIP (6), Middle MCP (9), Middle PIP (10), Ring MCP (13), and Index Tip (8).
    2. Thumb curl vector (1 -> 4) and direction alignment with Index proximal vector (5 -> 6).
    3. Fingertip tuck depths from wrist (0) for fingertips 8, 12, 16, 20.
    """
    pts = norm_pts.astype(np.float32)

    # 1. Thumb Tip (4) relative to Index MCP (5), Index PIP (6), Middle MCP (9), Middle PIP (10)
    v45 = pts[4] - pts[5]
    d45 = float(np.linalg.norm(v45))

    v46 = pts[4] - pts[6]
    d46 = float(np.linalg.norm(v46))

    v49 = pts[4] - pts[9]
    d49 = float(np.linalg.norm(v49))

    d410 = float(np.linalg.norm(pts[4] - pts[10]))
    d413 = float(np.linalg.norm(pts[4] - pts[13]))
    d48 = float(np.linalg.norm(pts[4] - pts[8]))

    # 2. Thumb curl vector (1 -> 4) and Index proximal vector (5 -> 6)
    v_th = pts[4] - pts[1]
    v_idx = pts[6] - pts[5]

    norm_th = float(np.linalg.norm(v_th)) + 1e-6
    norm_idx = float(np.linalg.norm(v_idx)) + 1e-6
    u_th = v_th / norm_th
    u_idx = v_idx / norm_idx

    dot = float(np.clip(np.dot(u_th, u_idx), -1.0, 1.0))
    angle = float(np.arccos(dot))

    # 3. Fingertip tuck depths from wrist (landmark 0)
    tuck8 = float(np.linalg.norm(pts[8] - pts[0]))
    tuck12 = float(np.linalg.norm(pts[12] - pts[0]))
    tuck16 = float(np.linalg.norm(pts[16] - pts[0]))
    tuck20 = float(np.linalg.norm(pts[20] - pts[0]))

    feats = [
        d45, d46, d49, d410, d413, d48,
        float(v45[0]), float(v45[1]), float(v45[2]),
        float(v46[0]), float(v46[1]), float(v46[2]),
        float(v49[0]), float(v49[1]), float(v49[2]),
        float(v_th[0]), float(v_th[1]), float(v_th[2]),
        dot, angle,
        tuck8, tuck12, tuck16, tuck20
    ]
    return feats


def resolve_fist_tie_breaker(norm_pts: np.ndarray, top_label: str, runner_up_label: str = None, margin: float = 0.0) -> Optional[str]:
    """
    Deterministic Geometric Tie-Breakers for closed-fist cluster ['A', 'S', 'T', 'N', 'M', 'E'].
    Evaluates exact knuckle & thumb placement geometry:
    - 'A': Thumb Tip (4) rests laterally along outer edge of Index MCP (5), pointing upright, not crossing over fingers.
    - 'S': Thumb Tip (4) crosses directly over center of curled fingers (across index ID 6 and middle ID 10).
    - 'T': Thumb Tip (4) is tucked strictly between index (5/6) and middle (9/10), poking upward.
    - 'N': Thumb Tip (4) crosses under index and middle fingers, protruding between middle and ring fingers.
    - 'M': Thumb Tip (4) crosses under index, middle, and ring fingers, protruding between ring and pinky.
    - 'E': All four fingertips curl tightly into palm base with the thumb curled flat beneath them.
    """
    pts = np.asarray(norm_pts, dtype=np.float32)
    if pts.shape[0] < 21:
        return top_label

    p0 = pts[0]    # Wrist
    p4 = pts[4]    # Thumb Tip
    p5 = pts[5]    # Index MCP
    p6 = pts[6]    # Index PIP
    p8 = pts[8]    # Index Tip
    p9 = pts[9]    # Middle MCP
    p10 = pts[10]  # Middle PIP
    p12 = pts[12]  # Middle Tip
    p13 = pts[13]  # Ring MCP
    p14 = pts[14]  # Ring PIP
    p16 = pts[16]  # Ring Tip
    p17 = pts[17]  # Pinky MCP
    p18 = pts[18]  # Pinky PIP
    p20 = pts[20]  # Pinky Tip

    # Coordinate frame invariant directional vectors
    norm_rad = float(np.linalg.norm(p5 - p9)) + 1e-6
    u_rad = (p5 - p9) / norm_rad  # Points radial / lateral (toward thumb side outside index)

    norm_up = float(np.linalg.norm(p5 - p0)) + 1e-6
    u_up = (p5 - p0) / norm_up    # Points proximal -> distal (upward along fingers)

    norm_cross = float(np.linalg.norm(p17 - p5)) + 1e-6
    u_cross = (p17 - p5) / norm_cross  # Points from index toward pinky (across front of fingers)

    lat_proj = float(np.dot(p4 - p5, u_rad))
    up_proj = float(np.dot(p4 - p5, u_up))
    cross_proj = float(np.dot(p4 - p5, u_cross))

    d45 = float(np.linalg.norm(p4 - p5))
    d46 = float(np.linalg.norm(p4 - p6))
    d49 = float(np.linalg.norm(p4 - p9))
    d410 = float(np.linalg.norm(p4 - p10))
    d414 = float(np.linalg.norm(p4 - p14))

    # Fingertip tucks from wrist
    tuck8 = float(np.linalg.norm(p8 - p0))
    tuck12 = float(np.linalg.norm(p12 - p0))
    tuck16 = float(np.linalg.norm(p16 - p0))
    tuck20 = float(np.linalg.norm(p20 - p0))
    avg_tuck = (tuck8 + tuck12 + tuck16 + tuck20) / 4.0

    # 1. Check 'E': all 4 fingertips tightly tucked into palm, thumb pulled flat underneath
    if up_proj < -0.05 and avg_tuck < 0.95 and cross_proj > 0.05:
        return "E"

    # 2. Check 'A': Thumb rests laterally on outer edge of index, pointing upright (NOT crossing over fingers)
    if (lat_proj > 0.04 or cross_proj < -0.05) and up_proj > 0.10 and d49 > 0.38:
        return "A"

    # 3. Check 'T': Thumb tip tucked between index and middle, poking upward
    if d46 < 0.22 and d410 < 0.25 and up_proj > 0.15:
        return "T"

    # 4. Check 'N': Thumb tucked under index & middle, protruding between middle and ring
    if cross_proj > 0.01 and cross_proj < 0.12 and d410 < 0.28 and d414 < 0.30:
        return "N"

    # 5. Check 'M': Thumb tucked under index, middle & ring, protruding between ring and pinky
    if cross_proj >= 0.13 and d414 < 0.32:
        return "M"

    # 6. Check 'S': Thumb wrapped horizontally across front of curled fingers (across index 6 and middle 10)
    if d46 < 0.40 and d410 < 0.40 and cross_proj > -0.02:
        return "S"

    if top_label in ["A", "S", "T", "N", "M", "E"] and margin >= 0.20:
        return top_label

    return top_label or "S"


def _fist_disambiguation_features(hand_landmarks) -> list[float]:
    """Computes explicit 23-dim fist disambiguation features from MediaPipe HandLandmarks."""
    pts = np.array([_coords(lm) for lm in hand_landmarks.landmark], dtype=np.float32)
    scale = float(np.linalg.norm(pts[9] - pts[0])) + 1e-6
    norm_pts = (pts - pts[0]) / scale
    return _fist_geometry_from_norm_coords(norm_pts)


def _fist_disambiguation_from_norm_coords(coords_63) -> list[float]:
    """Extracts identical 23-dim fist disambiguation features directly from normalized coordinates."""
    norm_pts = np.asarray(coords_63, dtype=np.float32).reshape(21, 3)
    if np.all(norm_pts == 0):
        return [0.0] * FIST_GEOMETRY_LEN
    return _fist_geometry_from_norm_coords(norm_pts)


def extract_features(hands_res) -> list[float]:
    results = _Results(hands_res)
    feats = []

    if results.left_hand_landmarks:
        feats += _normalised_hand(results.left_hand_landmarks)
    else:
        feats += [0.0] * 63

    if results.right_hand_landmarks:
        feats += _normalised_hand(results.right_hand_landmarks)
    else:
        feats += [0.0] * 63

    if results.left_hand_landmarks:
        feats += _thumb_distances(results.left_hand_landmarks)
        feats += _joint_angles(results.left_hand_landmarks)
        feats += _fingertip_distances(results.left_hand_landmarks)
    else:
        feats += [0.0] * 50

    if results.right_hand_landmarks:
        feats += _thumb_distances(results.right_hand_landmarks)
        feats += _joint_angles(results.right_hand_landmarks)
        feats += _fingertip_distances(results.right_hand_landmarks)
    else:
        feats += [0.0] * 50

    # Fist Disambiguation Features: 23 for Hand 0, 23 for Hand 1
    if results.left_hand_landmarks:
        feats += _fist_disambiguation_features(results.left_hand_landmarks)
    else:
        feats += [0.0] * FIST_GEOMETRY_LEN

    if results.right_hand_landmarks:
        feats += _fist_disambiguation_features(results.right_hand_landmarks)
    else:
        feats += [0.0] * FIST_GEOMETRY_LEN

    assert len(feats) == FEATURE_LEN, f"Expected {FEATURE_LEN} features, got {len(feats)}"
    return feats


def extract_primary_hand_features(hands_res, dominant_hand="right") -> list[float]:
    """Returns features with the secondary (resting) hand zeroed out to prevent noise in 1-hand gestures."""
    results = _Results(hands_res)

    if results.left_hand_landmarks and results.right_hand_landmarks:
        if dominant_hand == "left":
            results.left_hand_landmarks = None
        elif dominant_hand == "right":
            results.right_hand_landmarks = None
        else:
            y_r = results.left_hand_landmarks.landmark[0].y
            y_l = results.right_hand_landmarks.landmark[0].y
            if y_r <= y_l:
                results.right_hand_landmarks = None
            else:
                results.left_hand_landmarks = None

    feats = []

    if results.left_hand_landmarks:
        feats += _normalised_hand(results.left_hand_landmarks)
    else:
        feats += [0.0] * 63

    if results.right_hand_landmarks:
        feats += _normalised_hand(results.right_hand_landmarks)
    else:
        feats += [0.0] * 63

    if results.left_hand_landmarks:
        feats += _thumb_distances(results.left_hand_landmarks)
        feats += _joint_angles(results.left_hand_landmarks)
        feats += _fingertip_distances(results.left_hand_landmarks)
    else:
        feats += [0.0] * 50

    if results.right_hand_landmarks:
        feats += _thumb_distances(results.right_hand_landmarks)
        feats += _joint_angles(results.right_hand_landmarks)
        feats += _fingertip_distances(results.right_hand_landmarks)
    else:
        feats += [0.0] * 50

    if results.left_hand_landmarks:
        feats += _fist_disambiguation_features(results.left_hand_landmarks)
    else:
        feats += [0.0] * FIST_GEOMETRY_LEN

    if results.right_hand_landmarks:
        feats += _fist_disambiguation_features(results.right_hand_landmarks)
    else:
        feats += [0.0] * FIST_GEOMETRY_LEN

    assert len(feats) == FEATURE_LEN, f"Expected {FEATURE_LEN} features, got {len(feats)}"
    return feats


def extract_two_hand_features(right_hand_landmarks, left_hand_landmarks) -> list[float]:
    """Deterministically extracts 272-dim features with:
    Slot 0 = right_hand_landmarks (user's physical Right hand)
    Slot 1 = left_hand_landmarks (user's physical Left hand)
    Prevents MediaPipe handedness inversion from swapping slots."""
    feats = []

    if right_hand_landmarks:
        feats += _normalised_hand(right_hand_landmarks)
    else:
        feats += [0.0] * 63

    if left_hand_landmarks:
        feats += _normalised_hand(left_hand_landmarks)
    else:
        feats += [0.0] * 63

    if right_hand_landmarks:
        feats += _thumb_distances(right_hand_landmarks)
        feats += _joint_angles(right_hand_landmarks)
        feats += _fingertip_distances(right_hand_landmarks)
    else:
        feats += [0.0] * 50

    if left_hand_landmarks:
        feats += _thumb_distances(left_hand_landmarks)
        feats += _joint_angles(left_hand_landmarks)
        feats += _fingertip_distances(left_hand_landmarks)
    else:
        feats += [0.0] * 50

    if right_hand_landmarks:
        feats += _fist_disambiguation_features(right_hand_landmarks)
    else:
        feats += [0.0] * FIST_GEOMETRY_LEN

    if left_hand_landmarks:
        feats += _fist_disambiguation_features(left_hand_landmarks)
    else:
        feats += [0.0] * FIST_GEOMETRY_LEN

    assert len(feats) == FEATURE_LEN, f"Expected {FEATURE_LEN} features, got {len(feats)}"
    return feats


class _MirroredHand:
    class _LM:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    def __init__(self, hand_landmarks):
        self.landmark = [
            self._LM(1.0 - lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
        ]


def extract_single_hand_features(hand_landmarks, is_left_hand=False) -> list[float]:
    """Extracts features for a single hand placed in Slot 0 (indices 0..62, 126..175, and 226..248).
    If is_left_hand=True, horizontally mirrors coordinates so the geometry matches
    the model's single-hand training distributions with high accuracy."""
    target = _MirroredHand(hand_landmarks) if is_left_hand else hand_landmarks
    norm_hand = _normalised_hand(target)
    thumb_d = _thumb_distances(target)
    j_ang = _joint_angles(target)
    tip_d = _fingertip_distances(target)
    fist_d = _fist_disambiguation_features(target)
    feats = norm_hand + [0.0] * 63 + thumb_d + j_ang + tip_d + [0.0] * 50 + fist_d + [0.0] * FIST_GEOMETRY_LEN
    assert len(feats) == FEATURE_LEN, f"Expected {FEATURE_LEN} features, got {len(feats)}"
    return feats


def validate_feature_vector(feature_vector) -> np.ndarray:
    """Strictly validates feature vector dimensions for gesture_model.keras.
    Requires shape (1, 252) or (252,). Raises AssertionError if invalid."""
    arr = np.asarray(feature_vector, dtype=np.float32)
    if arr.ndim == 1:
        assert arr.shape == (FEATURE_LEN,), f"Feature dimension mismatch: expected ({FEATURE_LEN},), got {arr.shape}"
        arr = np.expand_dims(arr, axis=0)
    assert arr.shape == (1, FEATURE_LEN), f"Feature dimension mismatch: expected (1, {FEATURE_LEN}), got {arr.shape}"
    assert not np.isnan(arr).any(), "Feature vector contains NaN values"
    assert not np.isinf(arr).any(), "Feature vector contains Inf values"
    return arr





def apply_heuristics(hand_or_res, label, is_left_hand=False, confidence=None, route_mode="STATIC"):
    """
    Stabilizes and disambiguates signs using anatomical finger curl and thumb geometry checks:
    1. Disambiguates N vs T (fist with thumb under index PIP vs thumb between index & middle).
    2. Disambiguates A vs S (fist with thumb upright on side vs thumb folded across front knuckles).
    3. Disambiguates S vs O (fist with curled tucked fingertips vs curved circular "O" loop).
    4. Disambiguates P vs Q vs Z (middle pointing down with index forward vs index pointing sharply down vs index pointing up).
    5. Disambiguates U vs V vs R (parallel fingers side-by-side vs spread apart V vs crossed fingers R).
    """
    if label is None or str(label).strip() in ("", "—", "none", "NONE"):
        return "—"

    lbl_clean = str(label).strip().lower()

    # Extract landmark list if available
    lms = None
    if hasattr(hand_or_res, "landmark"):
        lms = hand_or_res.landmark
    elif hasattr(hand_or_res, "multi_hand_landmarks") and hand_or_res.multi_hand_landmarks:
        lms = hand_or_res.multi_hand_landmarks[0].landmark

    if lms and len(lms) >= 21:
        wrist = np.array([lms[0].x, lms[0].y], dtype=np.float32)
        idx_tip = np.array([lms[8].x, lms[8].y], dtype=np.float32)
        idx_pip = np.array([lms[6].x, lms[6].y], dtype=np.float32)
        mid_tip = np.array([lms[12].x, lms[12].y], dtype=np.float32)
        mid_pip = np.array([lms[10].x, lms[10].y], dtype=np.float32)
        ring_tip = np.array([lms[16].x, lms[16].y], dtype=np.float32)
        ring_pip = np.array([lms[14].x, lms[14].y], dtype=np.float32)
        pky_tip = np.array([lms[20].x, lms[20].y], dtype=np.float32)
        pky_pip = np.array([lms[18].x, lms[18].y], dtype=np.float32)
        thb_tip = np.array([lms[4].x, lms[4].y], dtype=np.float32)

        pky_mcp = np.array([lms[17].x, lms[17].y], dtype=np.float32)
        idx_mcp = np.array([lms[5].x, lms[5].y], dtype=np.float32)
        palm_w = float(max(np.linalg.norm(pky_mcp - idx_mcp), 1e-6))
        palm_scale = float(max(np.linalg.norm(np.array([lms[9].x, lms[9].y], dtype=np.float32) - wrist), 1e-6))

        # Check if outer fingers are curled into palm
        is_curled_idx = (np.linalg.norm(idx_tip - wrist) <= np.linalg.norm(idx_pip - wrist) * 1.25)
        is_curled_mid = (np.linalg.norm(mid_tip - wrist) <= np.linalg.norm(mid_pip - wrist) * 1.20)
        is_curled_ring = (np.linalg.norm(ring_tip - wrist) <= np.linalg.norm(ring_pip - wrist) * 1.20)
        is_curled_pky = (np.linalg.norm(pky_tip - wrist) <= np.linalg.norm(pky_pip - wrist) * 1.20)
        is_ext_pky = (np.linalg.norm(pky_tip - wrist) > np.linalg.norm(pky_pip - wrist) * 1.20 and lms[20].y < lms[18].y)
        is_fist = bool(is_curled_idx and is_curled_mid and is_curled_ring and is_curled_pky)

        # ── 0. Starting Shape Disambiguation ("I" vs "Y" / "YOU") ──
        # In 'I' (and J_START), pinky is extended and index, middle, ring are curled.
        # Thumb is folded across front of fingers: thumb flare ratio dist(4, 5) / dist(0, 9) < 0.38.
        # In 'Y', pinky is extended AND thumb is flared widely outward (flare ratio >= 0.50).
        if is_ext_pky and is_curled_idx and is_curled_mid and is_curled_ring:
            thumb_flare = float(np.linalg.norm(thb_tip - idx_mcp) / palm_scale)
            if thumb_flare < 0.38:
                # Strictly "I" / "J_START" — must NOT be classified as "Y" or "YOU"
                if lbl_clean in ("y", "you", "love", "i love you", "o", "how are you"):
                    return "J_START"
            elif thumb_flare >= 0.55:
                # Thumb is flared wide out -> "Y"
                if lbl_clean in ("j_start", "j", "i"):
                    return "Y"

        # ── 1. Fist Cluster Disambiguation (A, S, T, E, M, N) ──
        if is_fist and lbl_clean in ("a", "s", "t", "e", "m", "n"):
            # Construct normalized 21x3 coordinates for deterministic geometric tie-breaker
            raw_pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in lms], dtype=np.float32)
            sc = float(np.linalg.norm(raw_pts[9] - raw_pts[0])) + 1e-6
            norm_pts = (raw_pts - raw_pts[0]) / sc
            tb_res = resolve_fist_tie_breaker(norm_pts, lbl_clean.upper(), margin=0.5 if (confidence and confidence >= 0.8) else 0.0)
            if tb_res:
                return tb_res

            # Fallback to model label within fist cluster
            return lbl_clean.upper()

        # ── 2. S vs O Disambiguation & Mid-Sweep "O" Suppression ──
        # In 'O', Index Tip (8) and Thumb Tip (4) must touch forming a closed circle: dist(4, 8) < 0.28 * palm_scale.
        # In 'S' (fist), fingertips 8 & 12 are tucked close into palm (< 0.98 of palm scale).
        if lbl_clean in ("s", "o"):
            d_thumb_idx = float(np.linalg.norm(thb_tip - idx_tip) / palm_scale)
            if is_ext_pky:
                # During "J" stroke or "I", pinky is out, cannot be "O"
                return "J_START"
            if d_thumb_idx > 0.32:
                # Index and thumb do not touch; cannot be true "O"
                if is_fist:
                    return "S"
            tuck8 = float(np.linalg.norm(idx_tip - wrist) / palm_scale)
            tuck12 = float(np.linalg.norm(mid_tip - wrist) / palm_scale)
            if tuck8 < 0.98 and tuck12 < 0.98:
                return "S"
            elif tuck8 >= 1.02 and tuck12 >= 1.02 and d_thumb_idx <= 0.32:
                return "O"
            return "S" if lbl_clean == "s" else "O"

        # ── 3. P vs Q vs Z Disambiguation ──
        # In 'P': index points forward/downward (-0.50 <= idx_dy <= 0.85) and middle points DOWN (mid_dy >= 0.75).
        # In 'Q': index points sharply straight down to floor (idx_dy >= 1.20) and middle is curled in palm.
        # In 'Z': index points up or forward-up (idx_dy < 0.35) and middle is curled in palm.
        if lbl_clean in ("p", "q", "z", "z_start", "z_end"):
            idx_dy = float((lms[8].y - lms[5].y) / palm_w)
            mid_dy = float((lms[12].y - lms[9].y) / palm_w)

            # In P, middle finger points downward and index is forward/downward
            is_p_mid_down = (mid_dy >= 0.75 and lms[12].y > lms[9].y + 0.02)
            is_p_idx_fwd = (idx_dy < 0.90)

            if is_p_mid_down and is_p_idx_fwd:
                return "P"
            elif idx_dy >= 1.20 and (mid_dy < 1.0 or lms[8].y > lms[12].y + 0.05):
                return "Q"
            elif idx_dy < 0.35 and mid_dy < 0.70:
                return "Z" if (lbl_clean == "z" or route_mode == "DYNAMIC") else "P"
            return "P" if lbl_clean == "p" else ("Q" if lbl_clean == "q" else "Z")

        # ── 4. U vs V vs R Disambiguation ──
        # ASL anatomy:
        # U: Index + Middle extended straight up, parallel side-by-side. Ring + Pinky curled.
        # V: Index + Middle extended straight up, spread apart in distinct 'V'. Ring + Pinky curled.
        # R: Index + Middle extended straight up, crossed over each other. Ring + Pinky curled.
        is_ext_idx = (np.linalg.norm(idx_tip - wrist) > np.linalg.norm(idx_pip - wrist) * 1.15)
        is_ext_mid = (np.linalg.norm(mid_tip - wrist) > np.linalg.norm(mid_pip - wrist) * 1.15)
        is_curled_ring = (np.linalg.norm(ring_tip - wrist) <= np.linalg.norm(ring_pip - wrist) * 1.30)
        is_curled_pky = (np.linalg.norm(pky_tip - wrist) <= np.linalg.norm(pky_pip - wrist) * 1.30)
        is_two_extended = bool(is_ext_idx and is_ext_mid and is_curled_ring and is_curled_pky)

        if is_two_extended and lbl_clean in ("u", "v", "r"):
            p5 = np.array([lms[5].x, lms[5].y], dtype=np.float32)
            p8 = np.array([lms[8].x, lms[8].y], dtype=np.float32)
            p9 = np.array([lms[9].x, lms[9].y], dtype=np.float32)
            p12 = np.array([lms[12].x, lms[12].y], dtype=np.float32)

            v_knuckle = p9 - p5
            knuckle_len = float(max(np.linalg.norm(v_knuckle), 1e-6))
            u_knuckle = v_knuckle / knuckle_len
            v_tip = p12 - p8
            proj = float(np.dot(v_tip, u_knuckle) / knuckle_len)
            d_tip = float(np.hypot(lms[8].x - lms[12].x, lms[8].y - lms[12].y) / palm_w)

            # Check if 2D line segments cross (Index 5->8 vs Middle 9->12)
            def _ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            has_cross = (_ccw(p5, p9, p12) != _ccw(p8, p9, p12)) and (_ccw(p5, p8, p9) != _ccw(p5, p8, p12))

            # Preserve neural network predictions without spurious overrides:
            if lbl_clean == "u":
                if d_tip >= 0.65 and proj >= 1.40:
                    return "V"
                elif has_cross or proj <= 0.15:
                    return "R"
                return "U"
            elif lbl_clean == "r":
                if d_tip >= 0.65 and proj >= 1.40:
                    return "V"
                elif proj >= 0.70 and d_tip >= 0.32 and not has_cross:
                    return "U"
                return "R"
            elif lbl_clean == "v":
                if d_tip < 0.38 and proj < 0.85:
                    return "U"
                elif has_cross or proj <= 0.15:
                    return "R"
                return "V"

    return str(label).strip().upper()


# =====================================================================
# DYNAMIC GESTURE MOTION BUFFER & MULTI-STAGE RECOGNIZER
# =====================================================================

class LandmarkMotionBuffer:
    """Sliding FIFO sequence buffer (fixed window 16-30 frames) tracking wrist and fingertip velocities (dx/dt, dy/dt)."""
    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames
        self.buffer = collections.deque(maxlen=max_frames)
        self.timestamps = collections.deque(maxlen=max_frames)

    def append(self, landmarks, timestamp: Optional[float] = None):
        t = timestamp if timestamp is not None else time.time()
        if hasattr(landmarks, "landmark"):
            pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in landmarks.landmark], dtype=np.float32)
        elif isinstance(landmarks, (list, tuple)):
            if len(landmarks) > 0 and isinstance(landmarks[0], dict):
                pts = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in landmarks], dtype=np.float32)
            else:
                pts = np.asarray(landmarks, dtype=np.float32)
        else:
            return
        if pts.ndim == 2 and pts.shape[0] >= 21:
            self.buffer.append(pts[:21, :])
            self.timestamps.append(t)

    def clear(self):
        self.buffer.clear()
        self.timestamps.clear()

    def get_velocities(self) -> dict:
        """Computes velocity (dx/dt, dy/dt) and overall magnitude for wrist (0) and index tip (8)."""
        if len(self.buffer) < 2:
            return {
                "wrist": (0.0, 0.0),
                "index_tip": (0.0, 0.0),
                "magnitude": 0.0
            }
        p_now = self.buffer[-1]
        p_prev = self.buffer[-2]
        dt = max(1e-4, self.timestamps[-1] - self.timestamps[-2])

        v_wrist = ((p_now[0, 0] - p_prev[0, 0]) / dt, (p_now[0, 1] - p_prev[0, 1]) / dt)
        v_idx = ((p_now[8, 0] - p_prev[8, 0]) / dt, (p_now[8, 1] - p_prev[8, 1]) / dt)
        mag = float(np.hypot(v_wrist[0], v_wrist[1]) * 0.4 + np.hypot(v_idx[0], v_idx[1]) * 0.6)

        return {
            "wrist": (float(v_wrist[0]), float(v_wrist[1])),
            "index_tip": (float(v_idx[0]), float(v_idx[1])),
            "magnitude": mag
        }

    def get_trajectory(self, landmark_idx: int = 8) -> list:
        return [(float(f[landmark_idx, 0]), float(f[landmark_idx, 1])) for f in self.buffer]


# Dynamic Signs Registry: separate static single-frame classes from dynamic sequence classes
DYNAMIC_SIGNS_REGISTRY = {
    "HOW ARE YOU": {
        "start_label": "HOW ARE YOU_START",
        "end_label": "HOW ARE YOU_END",
        "two_handed": True,
        "max_window_sec": 1.5,
        "min_velocity": 0.05,
    },
    "NICE TO MEET YOU": {
        "start_label": "NICE TO MEET YOU_START",
        "end_label": "NICE TO MEET YOU_END",
        "two_handed": True,
        "max_window_sec": 1.8,
        "min_velocity": 0.05,
    },
    "J": {
        "start_label": "J_START",
        "end_label": "J_END",
        "two_handed": False,
        "max_window_sec": 1.5,
        "min_velocity": 0.04,
    },
    "Z": {
        "start_label": "Z_START",
        "end_label": "Z_END",
        "two_handed": False,
        "max_window_sec": 1.5,
        "min_velocity": 0.04,
    }
}


class MultiStageDynamicGestureStateMachine:
    """
    Multi-stage dynamic gesture state machine:
    1. Trigger Phase (Start Pose): Detect initial pose + motion onset exceeding velocity threshold.
       Starts a countdown window (1.5 seconds max).
    2. Path Verification: Checks that trajectory of dominant hand follows expected movement
       rather than requiring rigid frame hits on every tick.
    3. Release Phase (End Pose): If terminal pose is reached within window and trajectory is valid,
       commits the dynamic sign.
    4. Timeout / Abort: If velocity drops to zero before reaching end pose or 1.5s timer expires,
       aborts smoothly and resets to static sign detection.
    """
    def __init__(self):
        self.state = "IDLE"  # "IDLE", "TRACKING"
        self.active_sign: Optional[str] = None
        self.start_time: float = 0.0
        self.max_duration: float = 1.5
        self.motion_buffer = LandmarkMotionBuffer(max_frames=30)
        self.still_counter: int = 0
        self.trajectory_points: list = []

    def reset(self):
        self.state = "IDLE"
        self.active_sign = None
        self.start_time = 0.0
        self.still_counter = 0
        self.trajectory_points.clear()
        self.motion_buffer.clear()

    def update(
        self,
        landmarks,
        raw_pred: Optional[str] = None,
        confidence: float = 0.0,
        timestamp: Optional[float] = None
    ) -> dict:
        now = timestamp if timestamp is not None else time.time()
        self.motion_buffer.append(landmarks, now)
        vel = self.motion_buffer.get_velocities()
        mag = vel["magnitude"]

        raw_upper = str(raw_pred or "").upper().strip()

        # Phase 1: IDLE -> Trigger Phase
        if self.state == "IDLE":
            for sign_name, cfg in DYNAMIC_SIGNS_REGISTRY.items():
                is_start_hit = (raw_upper == cfg["start_label"] or raw_upper.startswith(sign_name) and raw_upper.endswith("_START"))
                if is_start_hit and mag >= cfg["min_velocity"] and confidence >= 0.40:
                    self.state = "TRACKING"
                    self.active_sign = sign_name
                    self.start_time = now
                    self.max_duration = cfg["max_window_sec"]
                    self.still_counter = 0
                    self.trajectory_points = [vel["index_tip"]]
                    return {
                        "state": "TRACKING",
                        "active_sign": self.active_sign,
                        "remaining_time": self.max_duration,
                        "committed_sign": None,
                        "aborted": False,
                        "message": f"DYNAMIC START: {self.active_sign}"
                    }
            return {
                "state": "IDLE",
                "active_sign": None,
                "remaining_time": 0.0,
                "committed_sign": None,
                "aborted": False,
                "message": ""
            }

        # Phase 2 & 3: TRACKING -> Path Verification & End Pose Release
        elapsed = now - self.start_time
        remaining = max(0.0, self.max_duration - elapsed)
        self.trajectory_points.append(vel["index_tip"])

        # Check Timeout
        if elapsed > self.max_duration or remaining <= 0:
            cancelled_sign = self.active_sign
            self.reset()
            return {
                "state": "IDLE",
                "active_sign": None,
                "remaining_time": 0.0,
                "committed_sign": None,
                "aborted": True,
                "message": f"DYNAMIC TIMEOUT: {cancelled_sign}"
            }

        # Check Still / Premature Stop
        if mag < 0.015:
            self.still_counter += 1
            if self.still_counter > 8 and elapsed < 0.4:
                cancelled_sign = self.active_sign
                self.reset()
                return {
                    "state": "IDLE",
                    "active_sign": None,
                    "remaining_time": 0.0,
                    "committed_sign": None,
                    "aborted": True,
                    "message": f"DYNAMIC ABORTED (Motion Stopped): {cancelled_sign}"
                }
        else:
            self.still_counter = max(0, self.still_counter - 1)

        # Check End Pose Completion
        cfg = DYNAMIC_SIGNS_REGISTRY.get(self.active_sign, {})
        end_label = cfg.get("end_label", f"{self.active_sign}_END")
        is_end_hit = (raw_upper == end_label or raw_upper.endswith("_END") or (self.active_sign in ("HOW ARE YOU", "NICE TO MEET YOU") and raw_upper == "YOU"))

        # Verify minimum motion frames before committing (>= 5 frames or >= 0.35s)
        if is_end_hit and (elapsed >= 0.35 or len(self.trajectory_points) >= 6):
            committed = self.active_sign
            self.reset()
            return {
                "state": "COMMITTED",
                "active_sign": None,
                "remaining_time": 0.0,
                "committed_sign": committed,
                "aborted": False,
                "message": f"DYNAMIC COMPLETED: {committed}"
            }

        return {
            "state": "TRACKING",
            "active_sign": self.active_sign,
            "remaining_time": remaining,
            "committed_sign": None,
            "aborted": False,
            "message": f"TRACKING: {self.active_sign} ({remaining:.1f}s)"
        }
