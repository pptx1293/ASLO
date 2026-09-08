import numpy as np
import itertools

FEATURE_LEN = 226


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


def extract_primary_hand_features(hands_res) -> list[float]:
    """Returns features with the secondary (resting) hand zeroed out to prevent noise in 1-hand gestures."""
    results = _Results(hands_res)

    if results.left_hand_landmarks and results.right_hand_landmarks:
        y_l = results.left_hand_landmarks.landmark[0].y
        y_r = results.right_hand_landmarks.landmark[0].y
        if y_l < y_r:
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


def apply_heuristics(hands_res, label):
    results = _Results(hands_res)
    label = str(label).lower()

    if label in ["n", "t"] and (
        results.left_hand_landmarks or results.right_hand_landmarks
    ):
        hand = (
            results.left_hand_landmarks
            if results.left_hand_landmarks
            else results.right_hand_landmarks
        )
        pts = [_coords(lm) for lm in hand.landmark]
        wrist = pts[0]
        scale = max(float(np.linalg.norm((pt - wrist)[:2])) for pt in pts)
        if scale < 1e-6:
            scale = 1.0

        pts_norm = [(pt - wrist) / scale for pt in pts]
        d_idx_pip = float(np.linalg.norm(pts_norm[4][:2] - pts_norm[6][:2]))
        d_thumb_idx_tip = float(np.linalg.norm(pts_norm[4][:2] - pts_norm[8][:2]))

        if d_idx_pip <= 0.16:
            label = "t"
        else:
            if d_thumb_idx_tip <= 0.28:
                label = "t"
            else:
                label = "n"

    elif label in ["u", "v", "r"] and (
        results.left_hand_landmarks or results.right_hand_landmarks
    ):
        hand = (
            results.left_hand_landmarks
            if results.left_hand_landmarks
            else results.right_hand_landmarks
        )
        pts = [_coords(lm) for lm in hand.landmark]
        wrist = pts[0]
        scale = max(float(np.linalg.norm((pt - wrist)[:2])) for pt in pts)
        if scale < 1e-6:
            scale = 1.0

        pts_norm = [(pt - wrist) / scale for pt in pts]

        x_dir = pts_norm[5][:2] - pts_norm[17][:2]
        n_dir = np.linalg.norm(x_dir)
        if n_dir > 1e-6:
            x_dir = x_dir / n_dir
        else:
            x_dir = np.array([1.0, 0.0], dtype=np.float32)

        x_idx = float(np.dot(pts_norm[8][:2], x_dir))
        x_mid = float(np.dot(pts_norm[12][:2], x_dir))
        proj_diff = x_idx - x_mid

        if proj_diff < -0.01:
            label = "r"
        else:
            d_idx_mid = float(np.linalg.norm(pts_norm[8][:2] - pts_norm[12][:2]))
            if d_idx_mid > 0.12:
                label = "v"
            else:
                label = "u"

    return label.upper()
