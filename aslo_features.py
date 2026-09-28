import numpy as np
import itertools

FEATURE_LEN = 226


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
    return np.array([1.0 - lm.x, lm.y, lm.z], dtype=np.float32)


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
    scale = float(np.linalg.norm(points[0] - points[9]))
    if scale < 1e-6:
        scale = 1.0
    return _normalise_points(points, wrist, scale)


def _thumb_distances(hand_landmarks) -> list[float]:
    points = np.array([_coords(lm) for lm in hand_landmarks.landmark])
    thumb_tip = points[4]
    scale = float(np.linalg.norm(points[0] - points[9]))
    if scale < 1e-6:
        scale = 1.0
        
    diffs = points - thumb_tip
    dists = np.linalg.norm(diffs, axis=1) / scale
    dists = np.delete(dists, 4)
    return dists.tolist()


def _fingertip_distances(hand_landmarks) -> list[float]:
    points = np.array([_coords(lm) for lm in hand_landmarks.landmark])
    scale = float(np.linalg.norm(points[0] - points[9]))
    if scale < 1e-6:
        scale = 1.0
        
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

    return feats


def extract_primary_hand_features(hands_res, dominant_hand="right") -> list[float]:
    """Returns features with the secondary (resting) hand zeroed out to prevent noise in 1-hand gestures."""
    results = _Results(hands_res)

    # In _Results:
    # results.left_hand_landmarks  = user's physical RIGHT hand (mirrored label 'Right')
    # results.right_hand_landmarks = user's physical LEFT hand (mirrored label 'Left')
    if results.left_hand_landmarks and results.right_hand_landmarks:
        if dominant_hand == "left":
            results.left_hand_landmarks = None
        elif dominant_hand == "right":
            results.right_hand_landmarks = None
        else:
            # Auto mode: prioritize the hand with wrist positioned higher in camera (smaller y)
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

    return feats


def extract_two_hand_features(right_hand_landmarks, left_hand_landmarks) -> list[float]:
    """Deterministically extracts 226-dim features with:
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
    """Extracts features for a single hand placed in Slot 0 (indices 0..62 and 126..175).
    If is_left_hand=True, horizontally mirrors coordinates so the geometry matches
    the model's single-hand training distributions with high accuracy."""
    target = _MirroredHand(hand_landmarks) if is_left_hand else hand_landmarks
    norm_hand = _normalised_hand(target)
    thumb_d = _thumb_distances(target)
    j_ang = _joint_angles(target)
    tip_d = _fingertip_distances(target)
    return norm_hand + [0.0] * 63 + thumb_d + j_ang + tip_d + [0.0] * 50


def apply_heuristics(hand_or_res, label, is_left_hand=False, confidence=None, route_mode="STATIC"):
    if hand_or_res is None:
        return str(label).upper()
    label = str(label).lower()

    if hasattr(hand_or_res, "landmark"):
        hand = hand_or_res
    else:
        results = _Results(hand_or_res)
        if results.left_hand_landmarks:
            hand = results.left_hand_landmarks
            is_left_hand = False
        elif results.right_hand_landmarks:
            hand = results.right_hand_landmarks
            is_left_hand = True
        else:
            return label.upper()

    target = _MirroredHand(hand) if is_left_hand else hand
    pts = [_coords(lm) for lm in target.landmark]
    wrist = pts[0]
    scale = max(float(np.linalg.norm((pt - wrist)[:2])) for pt in pts)
    if scale < 1e-6:
        scale = 1.0

    pts_norm = [(pt - wrist) / scale for pt in pts]

    # Finger extension ratios (tip distance from wrist vs MCP distance from wrist)
    idx_ext = float(np.linalg.norm(pts[8][:2] - wrist[:2]) / (np.linalg.norm(pts[5][:2] - wrist[:2]) + 1e-6))
    mid_ext = float(np.linalg.norm(pts[12][:2] - wrist[:2]) / (np.linalg.norm(pts[9][:2] - wrist[:2]) + 1e-6))
    ring_ext = float(np.linalg.norm(pts[16][:2] - wrist[:2]) / (np.linalg.norm(pts[13][:2] - wrist[:2]) + 1e-6))
    pky_ext = float(np.linalg.norm(pts[20][:2] - wrist[:2]) / (np.linalg.norm(pts[17][:2] - wrist[:2]) + 1e-6))

    palm_scale = float(np.linalg.norm(pts[9][:2] - wrist[:2]) + 1e-6)
    d_thb_idx_pip = float(np.linalg.norm(pts_norm[4][:2] - pts_norm[6][:2]))
    d_thb_mid_pip = float(np.linalg.norm(pts_norm[4][:2] - pts_norm[10][:2]))
    dy_thb_idx_pip = float(pts_norm[4][1] - pts_norm[6][1])  # y is positive downward

    # ── 1. ANATOMICAL DISAMBIGUATION: X vs P vs Q vs Z / D ───────────────────
    # Distinguishing metrics:
    # 1. pip_bend_deg: 2D angle between MCP->PIP and PIP->TIP vectors
    #    In X: Index finger is crooked into a hook (pip_bend_deg >= 20.0°, mean ~52°).
    #    In Z/D: Index finger is straight upright (pip_bend_deg < 18.0°, mean ~3°).
    # 2. idx_dy = pts_norm[8][1] - pts_norm[5][1]:
    #    In X: index tip is upward relative to knuckle (idx_dy <= 0.12).
    #    In P: index points forward/downward and middle finger points downward.
    #    In Q: index points sharply downward towards floor (idx_dy > 0.35).
    # 3. Middle finger extension & direction:
    #    In X: middle finger is tightly folded in a fist (mid_ext < 1.20, pts[12][1] < pts[0][1]).
    #    In P: middle finger is extended downward (pts[12][1] > pts[9][1] + 0.03).

    v_knuckle_pip = pts[6][:2] - pts[5][:2]
    v_pip_tip = pts[8][:2] - pts[6][:2]
    cos_pip = np.dot(v_knuckle_pip, v_pip_tip) / (np.linalg.norm(v_knuckle_pip) * np.linalg.norm(v_pip_tip) + 1e-6)
    pip_bend_deg = float(np.arccos(np.clip(cos_pip, -1.0, 1.0)) * 180 / np.pi)

    is_upright = (pts[5][1] < pts[0][1] - 0.02)
    is_fist_others = (mid_ext < 1.25 and ring_ext < 1.25 and pky_ext < 1.25)
    idx_tip_pip = float(np.linalg.norm(pts_norm[8][:2] - pts_norm[6][:2]))
    idx_dy = float(pts_norm[8][1] - pts_norm[5][1])

    # In X: hand is upright, middle/ring/pinky in fist, index is crooked (pip_bend_deg >= 20.0 or idx_tip_pip <= 0.22)
    # and not pointing to floor (idx_dy <= 0.12)
    is_hooked = (pip_bend_deg >= 20.0 or (idx_tip_pip <= 0.22 and idx_ext < 1.70))
    is_middle_curled = (pts[12][1] < pts[0][1] or mid_ext < 1.10)
    is_x_anatomy = (is_upright and is_fist_others and is_hooked and idx_dy <= 0.12 and is_middle_curled)

    lbl_clean = label.lower()
    if lbl_clean in ["x", "p", "q", "z", "z_start", "z_end", "d"]:
        # Hooked index upright in a fist -> DEFINITIVELY X (protected from low-light P/Q/Z jitter)
        if is_x_anatomy:
            return "X"

        # Sharp downward pointing index -> Q
        if idx_dy > 0.35:
            return "Q"

        # Downward pointing index or extended downward middle finger -> P
        if idx_dy > 0.08 or (not is_fist_others and mid_ext >= 1.05 and pts[12][1] > pts[9][1]):
            return "P"

        # Straight upright index finger -> D (static) or Z (dynamic)
        if idx_ext >= 1.20 and pip_bend_deg < 20.0:
            return "D" if route_mode == "STATIC" else "Z"

        # If raw prediction was X and hand is upright with other fingers in fist
        if lbl_clean == "x" and is_upright and is_fist_others:
            return "X"

        # Preserve authentic P / Q / Z predictions
        if lbl_clean == "p":
            return "P"
        if lbl_clean == "q":
            return "Q"
        if lbl_clean in ["z", "z_start", "z_end"]:
            return "D" if route_mode == "STATIC" else "Z"

    # ── 2. ANATOMICAL DISAMBIGUATION: ME vs YOU ──────────────────────────────
    # Both ME and YOU feature single-hand extended index with other fingers in fist.
    # IN YOU: Index points OUTWARD/FORWARD towards camera/interlocutor (dx >= 0.015).
    # IN ME: Index points INWARD towards signer's own chest (dx < 0.012).
    if lbl_clean in ["me", "you", "d"] and is_fist_others and idx_ext >= 1.15:
        dx_idx = pts[8][0] - pts[5][0]
        if dx_idx < 0.012:
            return "ME"
        elif dx_idx >= 0.015:
            return "YOU"

    # ── 3. TWO-FINGER & THREE-FINGER FAMILY: H vs U vs V vs R vs W vs K ───
    # ASL anatomy:
    # H: Index + Middle extended HORIZONTALLY (sideways across body/chest). Ring & Pinky curled.
    # W: Exactly 3 fingers extended VERTICALLY UP: Index + Middle + Ring. Pinky curled.
    # U: Index + Middle extended VERTICALLY UP, held together side-by-side. Ring & Pinky curled.
    # V: Index + Middle extended VERTICALLY UP, spread apart in 'V' shape. Ring & Pinky curled.
    # R: Index + Middle extended VERTICALLY UP, crossed over each other. Ring & Pinky curled.
    # K: Index + Middle extended VERTICALLY UP with thumb upright between knuckles.
    is_idx_mid_ext = (idx_ext >= 1.05 and mid_ext >= 1.05)
    is_two_finger_family = is_idx_mid_ext and (
        lbl_clean in ["u", "v", "r", "w", "k", "h", "d"]
        or (ring_ext < 1.35 and pky_ext < 1.30)
    )

    if is_two_finger_family:
        v_idx = pts_norm[8][:2] - pts_norm[5][:2]
        v_mid = pts_norm[12][:2] - pts_norm[9][:2]
        v_ring = pts_norm[16][:2] - pts_norm[13][:2]

        # Horizontal orientation test (pointing sideways across body):
        # In H, index/middle fingers point sideways: |dx| dominates or |dy| is small.
        # In U/V/R/W, fingers point strictly upward: dy < -0.15 and |dy| > 1.1 * |dx|.
        idx_dx, idx_dy = abs(float(v_idx[0])), float(v_idx[1])
        mid_dx, mid_dy = abs(float(v_mid[0])), float(v_mid[1])

        is_horizontal = (
            (idx_dx >= abs(idx_dy) * 0.70 or mid_dx >= abs(mid_dy) * 0.70)
            and (idx_dy > -0.22 or idx_dx > abs(idx_dy))
        )
        is_strictly_upright = (idx_dy < -0.20 and abs(idx_dy) > idx_dx * 1.15)

        # 3.1: H Detection
        # Hand is pointing horizontally/sideways, index and middle extended, pinky curled
        if pky_ext < 1.30 and (is_horizontal or (lbl_clean == "h" and not is_strictly_upright)):
            return "H"

        # 3.2: W Detection (Ring finger extended alongside index and middle, pointing UP)
        is_ring_upright = (
            ring_ext >= 1.20
            and pts[16][1] < pts[13][1] - 0.03
            and v_ring[1] < -0.12
            and not is_horizontal
        )
        if is_ring_upright and is_idx_mid_ext and pky_ext < 1.25:
            return "W"

        # Ring finger is curled into palm: Must be U, V, R, or K (Upright hand)
        if ring_ext < 1.30 and pky_ext < 1.25:
            # Knuckle vector 5->9
            vk = pts_norm[9][:2] - pts_norm[5][:2]
            d_k = float(np.linalg.norm(vk))
            vk_u = vk / (d_k + 1e-6)

            # Tip vector 8->12
            vt = pts_norm[12][:2] - pts_norm[8][:2]
            d_t = float(np.linalg.norm(vt))

            proj_ratio = float(np.dot(vt, vk_u)) / (d_k + 1e-6)
            d_ratio = d_t / (d_k + 1e-6)

            u_idx = v_idx / (np.linalg.norm(v_idx) + 1e-6)
            u_mid = v_mid / (np.linalg.norm(v_mid) + 1e-6)
            cos_ang = np.clip(np.dot(u_idx, u_mid), -1.0, 1.0)
            ang_deg = float(np.arccos(cos_ang) * 180.0 / np.pi)

            # 3.3: R Detection (Fingers crossed over each other)
            is_crossed = (
                proj_ratio < 0.0
                or (lbl_clean == "r" and proj_ratio <= 0.60)
                or (proj_ratio <= 0.35 and d_ratio <= 1.0)
                or (d_t <= 0.055 and proj_ratio <= 0.40)
            )
            if is_crossed:
                return "R"

            # 3.4: V Detection (Fingers spread wide apart in a distinct 'V')
            if not is_crossed and proj_ratio > 0.60 and d_t >= 0.115 and (d_ratio >= 1.65 or proj_ratio >= 1.40 or ang_deg >= 6.5):
                return "V"
            if not is_crossed and lbl_clean == "v" and d_t >= 0.105 and d_ratio >= 1.50 and proj_ratio >= 1.30:
                return "V"

            # 3.5: K Detection (Thumb upright between index and middle knuckles)
            if lbl_clean == "k" and pts[4][1] < pts[9][1]:
                return "K"

            # 3.6: Fallback for horizontal hand if not caught earlier
            if is_horizontal or lbl_clean == "h":
                return "H"

            # 3.7: U Detection (Default for parallel, side-by-side uncrossed upright index+middle)
            return "U"

    # ── HELLO vs B DISAMBIGUATION ─────────────────────────────────────────────
    # Both have 4 fingers (index, middle, ring, pinky) extended vertically.
    # In 'HELLO': Hand is raised higher near the temple/head (wrist_y <= 0.38)
    #             and thumb is extended outward/open (thumb tip to pinky mcp distance >= 0.26).
    # In 'B': Hand is held at chest/mid-level (wrist_y > 0.38)
    #        and thumb is folded tightly across front of palm (thumb tip to pinky mcp < 0.26).
    if label in ["b", "hello"]:
        all_four_up = (idx_ext >= 1.15 and mid_ext >= 1.15 and ring_ext >= 1.15 and pky_ext >= 1.15)
        if all_four_up:
            d_thb_pky_norm = float(np.linalg.norm(pts_norm[4][:2] - pts_norm[17][:2]))
            wrist_y = float(pts[0][1])
            if wrist_y <= 0.38 or d_thb_pky_norm >= 0.26:
                return "HELLO"
            else:
                return "B"

    # ── ME vs START vs HOW ARE YOU Disambiguation ────────────────────────────
    # In 'ME', the user points the index finger towards their chest/body (downward/inward, idx_dy > 0.15)
    # while the other fingers (middle, ring, pinky) are folded or curled.
    # In 'START' or 'HOW ARE YOU_START', both hands are open and moving horizontally.
    if label in ["me", "start", "how are you_start"]:
        idx_downward = float(pts[8][1] - pts[5][1]) # y positive downward
        pky_folded = (pky_ext < 1.25)
        mid_folded = (mid_ext < 1.30)
        if idx_downward > 0.15 and pky_folded and mid_folded:
            return "ME"

    # ── 4. E vs O DISAMBIGUATION ───────────────────────────────────────────────
    if label in ["e", "o"]:
        if idx_ext < 1.05 and mid_ext < 1.05 and ring_ext < 1.05:
            return "E"
        elif idx_ext >= 1.05 and mid_ext >= 1.05:
            return "O"

    # ── 5. HARD ANATOMICAL CHECK: 'A' vs 'Y' ──────────────────────────────────
    if label == "y" and pky_ext < 1.15:
        if dy_thb_idx_pip >= 0.10:
            return "S"
        elif d_thb_idx_pip < 0.15:
            return "T"
        elif dy_thb_idx_pip < 0.02 and d_thb_mid_pip > 0.18:
            return "A"
        return "A"
    elif label == "a" and pky_ext > 1.35 and mid_ext < 1.20 and ring_ext < 1.20:
        return "Y"

    # ── 6. HARD ANATOMICAL CHECK: 'I love you' vs 'J_START' / 'Y' / 'E' / 'A' ─
    if "love" in label and (idx_ext < 1.25 or pky_ext < 1.25):
        if pky_ext > 1.30 and idx_ext < 1.15 and mid_ext < 1.15:
            d_pip_raw = float(np.linalg.norm(pts[4][:2] - pts[6][:2]) / palm_scale)
            if d_pip_raw > 0.45:
                return "Y"
            else:
                return "J_START"
        if idx_ext < 1.05 and mid_ext < 1.05 and ring_ext < 1.05 and pky_ext < 1.05:
            if dy_thb_idx_pip >= 0.0 and d_thb_idx_pip < 0.18:
                return "E"
            elif dy_thb_idx_pip < 0.02 and d_thb_mid_pip > 0.18:
                return "A"
            elif dy_thb_idx_pip >= 0.10:
                return "S"
            return "E"
        if idx_ext < 1.20 and mid_ext < 1.20 and ring_ext < 1.20 and pky_ext < 1.20:
            if dy_thb_idx_pip < 0.02 and d_thb_mid_pip > 0.18:
                return "A"
            elif dy_thb_idx_pip >= 0.10:
                return "S"
            elif d_thb_idx_pip < 0.15:
                return "T"
            return "A"

    # ── 7. J_START DETECTION (Pinky extended, index/mid/ring folded, thumb curled) ──
    d_pip_raw = float(np.linalg.norm(pts[4][:2] - pts[6][:2]) / palm_scale)
    if label == "y" and pky_ext > 1.30 and idx_ext < 1.15 and mid_ext < 1.15 and d_pip_raw <= 0.45:
        return "J_START"

    # ── 8. ANATOMICAL FIST DISAMBIGUATION (A vs S vs T vs N vs E) ─────────────
    all_fist_fingers = (idx_ext < 1.25 and mid_ext < 1.25 and ring_ext < 1.25 and pky_ext < 1.25)
    if all_fist_fingers and label in ["a", "s", "t", "n", "m", "e"]:
        if label == "e":
            return "E"
        if dy_thb_idx_pip < 0.02 and d_thb_mid_pip > 0.18:
            return "A"
        if label in ["a", "s"] and dy_thb_idx_pip >= 0.10:
            return "S"
        if label in ["n", "t"]:
            if d_thb_idx_pip < 0.15:
                return "T"
            else:
                return "N"

    # ── 9. CONFIDENCE CUTOFF FOR OTHER SIGNS ────────────────────────────────────
    if confidence is not None and confidence >= 0.85:
        return label.upper()

    if label in ["n", "t"]:
        if d_thb_idx_pip < 0.15:
            return "T"
        else:
            return "N"

    return label.upper()
