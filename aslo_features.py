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
    """
    Pass-through to neural network prediction.
    Manual heuristic overrides are completely disabled to ensure 100% clean,
    un-hijacked predictions from the trained neural network (98.2% test accuracy).
    """
    if label is None or str(label).strip() in ("", "—", "none", "NONE"):
        return "—"
    return str(label).strip().upper()
