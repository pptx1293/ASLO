import numpy as np
import itertools

FEATURE_LEN = 252


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


def _fist_disambiguation_features(hand_landmarks) -> list[float]:
    """Computes explicit thumb-to-finger relational geometry to disambiguate
    closed-fist family gestures (A, E, M, N, S, T):
    1. Thumb Tip (id 4) relative to Index MCP (id 5), Middle MCP (id 9), Ring MCP (id 13), and Index Tip (id 8).
    2. Thumb curl and direction vector (id 1 -> id 4) and angle/dot product with Index proximal vector (id 5 -> id 6).
    3. Fingertip tuck depths: dist(8, 0), dist(12, 0), dist(16, 0), dist(20, 0) normalized by palm scale.
    """
    pts = np.array([_coords(lm) for lm in hand_landmarks.landmark])
    scale = float(np.linalg.norm(pts[9] - pts[0])) + 1e-6

    # 1. Thumb Tip relative to knuckles and index tip
    d45 = float(np.linalg.norm(pts[4] - pts[5]) / scale)
    d49 = float(np.linalg.norm(pts[4] - pts[9]) / scale)
    d413 = float(np.linalg.norm(pts[4] - pts[13]) / scale)
    d48 = float(np.linalg.norm(pts[4] - pts[8]) / scale)

    # 2. Thumb curl and direction vector (1 -> 4)
    v_th = (pts[4] - pts[1]) / scale
    v_idx = (pts[6] - pts[5]) / scale

    norm_th = float(np.linalg.norm(v_th)) + 1e-6
    norm_idx = float(np.linalg.norm(v_idx)) + 1e-6
    u_th = v_th / norm_th
    u_idx = v_idx / norm_idx

    dot = float(np.clip(np.dot(u_th, u_idx), -1.0, 1.0))
    angle = float(np.arccos(dot))

    # 3. Finger tuck depths normalized by palm scale
    tuck8 = float(np.linalg.norm(pts[8] - pts[0]) / scale)
    tuck12 = float(np.linalg.norm(pts[12] - pts[0]) / scale)
    tuck16 = float(np.linalg.norm(pts[16] - pts[0]) / scale)
    tuck20 = float(np.linalg.norm(pts[20] - pts[0]) / scale)

    return [
        d45, d49, d413, d48,
        float(v_th[0]), float(v_th[1]), float(v_th[2]),
        dot, angle,
        tuck8, tuck12, tuck16, tuck20
    ]


def _fist_disambiguation_from_norm_coords(coords_63) -> list[float]:
    """Extracts identical 13-dim fist disambiguation features directly from normalized coordinates."""
    pts = np.asarray(coords_63, dtype=np.float32).reshape(21, 3)
    if np.all(pts == 0):
        return [0.0] * 13

    d45 = float(np.linalg.norm(pts[4] - pts[5]))
    d49 = float(np.linalg.norm(pts[4] - pts[9]))
    d413 = float(np.linalg.norm(pts[4] - pts[13]))
    d48 = float(np.linalg.norm(pts[4] - pts[8]))

    v_th = pts[4] - pts[1]
    v_idx = pts[6] - pts[5]

    norm_th = float(np.linalg.norm(v_th)) + 1e-6
    norm_idx = float(np.linalg.norm(v_idx)) + 1e-6
    u_th = v_th / norm_th
    u_idx = v_idx / norm_idx

    dot = float(np.clip(np.dot(u_th, u_idx), -1.0, 1.0))
    angle = float(np.arccos(dot))

    tuck8 = float(np.linalg.norm(pts[8] - pts[0]))
    tuck12 = float(np.linalg.norm(pts[12] - pts[0]))
    tuck16 = float(np.linalg.norm(pts[16] - pts[0]))
    tuck20 = float(np.linalg.norm(pts[20] - pts[0]))

    return [
        d45, d49, d413, d48,
        float(v_th[0]), float(v_th[1]), float(v_th[2]),
        dot, angle,
        tuck8, tuck12, tuck16, tuck20
    ]


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

    # Fist Disambiguation Features: 13 for Hand 0, 13 for Hand 1
    if results.left_hand_landmarks:
        feats += _fist_disambiguation_features(results.left_hand_landmarks)
    else:
        feats += [0.0] * 13

    if results.right_hand_landmarks:
        feats += _fist_disambiguation_features(results.right_hand_landmarks)
    else:
        feats += [0.0] * 13

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
        feats += [0.0] * 13

    if results.right_hand_landmarks:
        feats += _fist_disambiguation_features(results.right_hand_landmarks)
    else:
        feats += [0.0] * 13

    assert len(feats) == FEATURE_LEN, f"Expected {FEATURE_LEN} features, got {len(feats)}"
    return feats


def extract_two_hand_features(right_hand_landmarks, left_hand_landmarks) -> list[float]:
    """Deterministically extracts 252-dim features with:
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
        feats += [0.0] * 13

    if left_hand_landmarks:
        feats += _fist_disambiguation_features(left_hand_landmarks)
    else:
        feats += [0.0] * 13

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
    """Extracts features for a single hand placed in Slot 0 (indices 0..62, 126..175, and 226..238).
    If is_left_hand=True, horizontally mirrors coordinates so the geometry matches
    the model's single-hand training distributions with high accuracy."""
    target = _MirroredHand(hand_landmarks) if is_left_hand else hand_landmarks
    norm_hand = _normalised_hand(target)
    thumb_d = _thumb_distances(target)
    j_ang = _joint_angles(target)
    tip_d = _fingertip_distances(target)
    fist_d = _fist_disambiguation_features(target)
    feats = norm_hand + [0.0] * 63 + thumb_d + j_ang + tip_d + [0.0] * 50 + fist_d + [0.0] * 13
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
        is_curled_mid = (np.linalg.norm(mid_tip - wrist) <= np.linalg.norm(mid_pip - wrist) * 1.20)
        is_curled_ring = (np.linalg.norm(ring_tip - wrist) <= np.linalg.norm(ring_pip - wrist) * 1.20)
        is_curled_pky = (np.linalg.norm(pky_tip - wrist) <= np.linalg.norm(pky_pip - wrist) * 1.20)
        is_fist = bool(is_curled_mid and is_curled_ring and is_curled_pky)

        # ── 1. N vs T Disambiguation ──
        if is_fist and lbl_clean in ("t", "n"):
            d_idx_pip = float(np.linalg.norm(thb_tip - idx_pip) / palm_w)
            if d_idx_pip >= 0.42:
                return "N"
            else:
                return "T"

        # ── 2. A vs S Disambiguation ──
        # In ASL 'A', thumb is resting straight up on the outer side of the index finger.
        # In ASL 'S', thumb is folded horizontally across the front of the curled fingers.
        if is_fist and lbl_clean in ("a", "s"):
            mid_mcp = np.array([lms[9].x, lms[9].y], dtype=np.float32)
            d4_to_9 = float(np.linalg.norm(thb_tip - mid_mcp) / palm_w)
            dy_thb_idx = float((lms[4].y - lms[5].y) / palm_w)  # y is positive downward

            if d4_to_9 >= 0.65 or dy_thb_idx < -0.40:
                return "A"
            elif d4_to_9 < 0.55 and dy_thb_idx > -0.35:
                return "S"
            return "A" if lbl_clean == "a" else "S"

        # ── 3. S vs O Disambiguation ──
        # In 'S' (fist), fingertips 8 & 12 are tucked close into palm (< 0.98 of palm scale).
        # In 'O', fingers arc forward touching thumb to form an open circle (tips >= 1.02 of palm scale).
        if lbl_clean in ("s", "o"):
            tuck8 = float(np.linalg.norm(idx_tip - wrist) / palm_scale)
            tuck12 = float(np.linalg.norm(mid_tip - wrist) / palm_scale)
            if tuck8 < 0.98 and tuck12 < 0.98:
                return "S"
            elif tuck8 >= 1.02 and tuck12 >= 1.02:
                return "O"
            return "S" if lbl_clean == "s" else "O"

        # ── 4. P vs Q vs Z Disambiguation ──
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

        # ── 5. U vs V vs R Disambiguation ──
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
                # Only override U with V if fingertips are spread wide apart (clear V)
                if d_tip >= 0.65 and proj >= 1.40:
                    return "V"
                # Only override U with R if fingers are crossed
                elif has_cross or proj <= 0.15:
                    return "R"
                return "U"
            elif lbl_clean == "r":
                # R: fingers crossed. Do not override with U or V unless clearly spread or parallel
                if d_tip >= 0.65 and proj >= 1.40:
                    return "V"
                elif proj >= 0.70 and d_tip >= 0.32 and not has_cross:
                    return "U"
                return "R"
            elif lbl_clean == "v":
                # Only override V with U if fingertips are held close together
                if d_tip < 0.38 and proj < 0.85:
                    return "U"
                elif has_cross or proj <= 0.15:
                    return "R"
                return "V"

    return str(label).strip().upper()
