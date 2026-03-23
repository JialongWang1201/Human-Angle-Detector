"""
Unit tests for MachineVision.py algorithm changes.
No hardware required — tests run on any machine with mediapipe installed.
"""
import math
import types
import unittest


# ---------------------------------------------------------------------------
# Inline the functions under test so we don't import PiCamera2 / RPi.GPIO
# ---------------------------------------------------------------------------

TORSO_VISIBILITY_MIN = 0.55
HFOV = 88.0


def get_torso_center_x(landmarks, pose_module, frame_width):
    left_specs = (
        (pose_module.PoseLandmark.LEFT_SHOULDER, 1.4),
        (pose_module.PoseLandmark.LEFT_HIP, 1.0),
    )
    right_specs = (
        (pose_module.PoseLandmark.RIGHT_SHOULDER, 1.4),
        (pose_module.PoseLandmark.RIGHT_HIP, 1.0),
    )
    weighted_x = 0.0
    total_weight = 0.0
    left_visible = 0
    right_visible = 0

    for landmark_enum, base_weight in left_specs:
        landmark = landmarks[landmark_enum.value]
        if landmark.visibility < TORSO_VISIBILITY_MIN:
            continue
        weight = base_weight * (landmark.visibility ** 2)
        weighted_x += landmark.x * frame_width * weight
        total_weight += weight
        left_visible += 1

    for landmark_enum, base_weight in right_specs:
        landmark = landmarks[landmark_enum.value]
        if landmark.visibility < TORSO_VISIBILITY_MIN:
            continue
        weight = base_weight * (landmark.visibility ** 2)
        weighted_x += landmark.x * frame_width * weight
        total_weight += weight
        right_visible += 1

    if left_visible == 0 or right_visible == 0 or total_weight <= 0.0:
        return None

    return int(weighted_x / total_weight)


def pixel_offset_to_angle_deg(pixel_offset, frame_width):
    if frame_width <= 0:
        return 0.0
    focal_pixels = (frame_width / 2.0) / max(math.tan(math.radians(HFOV / 2.0)), 1e-6)
    return math.degrees(math.atan2(pixel_offset, focal_pixels))


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_pose_module():
    """
    Build a pure mock of mp.solutions.pose — no mediapipe required.
    Landmark indices match the real MediaPipe spec:
      LEFT_SHOULDER=11, RIGHT_SHOULDER=12, LEFT_HIP=23, RIGHT_HIP=24
    """
    def make_lm_enum(value):
        lm = types.SimpleNamespace()
        lm.value = value
        return lm

    PoseLandmark = types.SimpleNamespace(
        LEFT_SHOULDER=make_lm_enum(11),
        RIGHT_SHOULDER=make_lm_enum(12),
        LEFT_HIP=make_lm_enum(23),
        RIGHT_HIP=make_lm_enum(24),
        LEFT_WRIST=make_lm_enum(15),
        RIGHT_WRIST=make_lm_enum(16),
    )
    pose_module = types.SimpleNamespace(PoseLandmark=PoseLandmark)
    return pose_module


def _make_landmark(x, y, visibility):
    lm = types.SimpleNamespace()
    lm.x = x
    lm.y = y
    lm.visibility = visibility
    return lm


def _make_landmarks(overrides: dict):
    """
    Build a 33-element landmark list (all invisible by default).
    overrides: {landmark_value: (x, y, visibility)}
    """
    lms = [_make_landmark(0.5, 0.5, 0.0) for _ in range(33)]
    for idx, (x, y, vis) in overrides.items():
        lms[idx] = _make_landmark(x, y, vis)
    return lms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetTorsoCenterX(unittest.TestCase):

    def setUp(self):
        self.mp_pose = _make_pose_module()
        self.W = 640  # frame width

    def _lm(self, **kwargs):
        """kwargs: landmark_name=(x, y, vis)"""
        mp = self.mp_pose
        mapping = {
            "LEFT_SHOULDER":  mp.PoseLandmark.LEFT_SHOULDER.value,
            "RIGHT_SHOULDER": mp.PoseLandmark.RIGHT_SHOULDER.value,
            "LEFT_HIP":       mp.PoseLandmark.LEFT_HIP.value,
            "RIGHT_HIP":      mp.PoseLandmark.RIGHT_HIP.value,
        }
        overrides = {mapping[k]: v for k, v in kwargs.items()}
        return _make_landmarks(overrides)

    # ---- Happy path --------------------------------------------------------

    def test_centered_person_returns_half_width(self):
        """Person perfectly centred → result should be ~320."""
        lms = self._lm(
            LEFT_SHOULDER=(0.5, 0.3, 0.9),
            RIGHT_SHOULDER=(0.5, 0.3, 0.9),
            LEFT_HIP=(0.5, 0.7, 0.9),
            RIGHT_HIP=(0.5, 0.7, 0.9),
        )
        result = get_torso_center_x(lms, self.mp_pose, self.W)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 320, delta=5)

    def test_person_left_of_center(self):
        """Person at x=0.25 → result should be <320."""
        lms = self._lm(
            LEFT_SHOULDER=(0.2, 0.3, 0.9),
            RIGHT_SHOULDER=(0.3, 0.3, 0.9),
            LEFT_HIP=(0.2, 0.7, 0.9),
            RIGHT_HIP=(0.3, 0.7, 0.9),
        )
        result = get_torso_center_x(lms, self.mp_pose, self.W)
        self.assertIsNotNone(result)
        self.assertLess(result, 320)

    # ---- Fix verification: same-side occlusion ----------------------------

    def test_only_left_side_visible_returns_none(self):
        """BUG FIX: only left landmarks visible → must return None (was returning biased value before)."""
        lms = self._lm(
            LEFT_SHOULDER=(0.2, 0.3, 0.9),
            LEFT_HIP=(0.2, 0.7, 0.9),
            # right side invisible
            RIGHT_SHOULDER=(0.8, 0.3, 0.1),
            RIGHT_HIP=(0.8, 0.7, 0.1),
        )
        result = get_torso_center_x(lms, self.mp_pose, self.W)
        self.assertIsNone(result, "Should return None when only one side visible")

    def test_only_right_side_visible_returns_none(self):
        """BUG FIX: only right landmarks visible → must return None."""
        lms = self._lm(
            LEFT_SHOULDER=(0.2, 0.3, 0.1),
            LEFT_HIP=(0.2, 0.7, 0.1),
            RIGHT_SHOULDER=(0.8, 0.3, 0.9),
            RIGHT_HIP=(0.8, 0.7, 0.9),
        )
        result = get_torso_center_x(lms, self.mp_pose, self.W)
        self.assertIsNone(result, "Should return None when only one side visible")

    def test_one_from_each_side_is_enough(self):
        """One left + one right landmark visible → valid result."""
        lms = self._lm(
            LEFT_SHOULDER=(0.4, 0.3, 0.9),
            RIGHT_HIP=(0.6, 0.7, 0.9),
        )
        result = get_torso_center_x(lms, self.mp_pose, self.W)
        self.assertIsNotNone(result)

    def test_all_invisible_returns_none(self):
        lms = _make_landmarks({})
        result = get_torso_center_x(lms, self.mp_pose, self.W)
        self.assertIsNone(result)


class TestPixelOffsetToAngle(unittest.TestCase):

    def test_zero_offset_is_zero_degrees(self):
        self.assertAlmostEqual(pixel_offset_to_angle_deg(0, 640), 0.0, places=5)

    def test_positive_offset_positive_angle(self):
        angle = pixel_offset_to_angle_deg(100, 640)
        self.assertGreater(angle, 0)

    def test_negative_offset_negative_angle(self):
        angle = pixel_offset_to_angle_deg(-100, 640)
        self.assertLess(angle, 0)

    def test_full_right_edge_equals_hfov_half(self):
        """Pixel at right edge (offset = width/2) → angle == HFOV/2 by definition."""
        angle = pixel_offset_to_angle_deg(320, 640)
        self.assertAlmostEqual(angle, HFOV / 2, places=4)

    def test_inside_edge_less_than_hfov_half(self):
        """atan is sublinear: offset < width/2 → angle < HFOV/2."""
        angle = pixel_offset_to_angle_deg(200, 640)
        self.assertLess(angle, HFOV / 2)
        self.assertGreater(angle, 0)

    def test_zero_frame_width_returns_zero(self):
        self.assertEqual(pixel_offset_to_angle_deg(100, 0), 0.0)

    def test_symmetry(self):
        """Left and right offsets give equal magnitude."""
        a1 = pixel_offset_to_angle_deg(150, 640)
        a2 = pixel_offset_to_angle_deg(-150, 640)
        self.assertAlmostEqual(a1, -a2, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
