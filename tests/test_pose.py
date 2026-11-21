"""Self-check for the pose maths and the session bookkeeping.

Run from the repository root:  python tests/test_pose.py
No webcam, no MediaPipe model needed.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crom.pose import as_rotation, euler_zyx_deg, relative_rotation  # noqa: E402
from crom.protocol import STEPS, Protocol  # noqa: E402
from crom.session import (  # noqa: E402
    MOVEMENTS, Session, channels, dominant, means, symmetry,
)


def rot(axis: str, deg: float) -> np.ndarray:
    """Elementary rotation matrix about x, y or z (degrees)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def close(a, b, tol=1e-6) -> bool:
    return abs(a - b) <= tol


def test_euler_roundtrip() -> None:
    for pitch, yaw, roll in [(0, 0, 0), (30, 0, 0), (0, 25, 0), (0, 0, 15),
                             (30, 20, 10), (-45, -35, -20), (60, 5, -30)]:
        R = rot("z", roll) @ rot("y", yaw) @ rot("x", pitch)
        p, y, r = euler_zyx_deg(R)
        assert close(p, pitch, 1e-6) and close(y, yaw, 1e-6) and close(r, roll, 1e-6), (
            f"expected {(pitch, yaw, roll)}, got {(p, y, r)}"
        )
    print("ok  euler extraction round-trips known rotations")


def test_calibration_is_relative() -> None:
    """A head turned 25 deg from an already-tilted neutral must read 25 deg."""
    neutral = rot("z", 12) @ rot("y", -8) @ rot("x", 15)   # arbitrary starting pose
    current = neutral @ rot("y", 25)                        # +25 yaw from neutral
    p, y, r = euler_zyx_deg(relative_rotation(neutral, current))
    assert close(y, 25, 1e-6) and close(p, 0, 1e-6) and close(r, 0, 1e-6), (p, y, r)

    # Moving the whole rig (camera rotated) must not change the relative angle.
    camera_offset = rot("x", 20) @ rot("z", -10)
    p, y, r = euler_zyx_deg(
        relative_rotation(camera_offset @ neutral, camera_offset @ current)
    )
    assert close(y, 25, 1e-6), y
    print("ok  angles are relative to the calibrated neutral pose")


def test_matrix_cleanup() -> None:
    """Noisy / scaled input matrices are re-orthonormalised before use."""
    R = rot("y", 18) * 1.0004
    p, y, r = euler_zyx_deg(R)
    assert close(y, 18, 1e-6), y
    assert close(float(np.linalg.det(as_rotation(R))), 1.0, 1e-9)
    print("ok  as_rotation repairs scale/drift")


def test_channels_and_signs() -> None:
    ch = channels(20, -30, 10)  # +pitch down, -yaw (patient's right), +roll right tilt
    assert close(ch["Flexion"], 20) and close(ch["Extension"], 0)
    assert close(ch["Right rotation"], 30) and close(ch["Left rotation"], 0)
    assert close(ch["Right lateral flexion"], 10) and close(ch["Left lateral flexion"], 0)

    ch = channels(-40, 15, -5)
    assert close(ch["Extension"], 40) and close(ch["Left rotation"], 15)
    assert close(ch["Left lateral flexion"], 5)

    assert dominant({m: 0.1 for m in MOVEMENTS})[0] == "Neutral"
    assert dominant(ch)[0] == "Extension"
    print("ok  six channels map to the right directions")


def test_session_peaks_trials_export() -> None:
    s = Session()
    s.update(channels(10, 0, 0))
    s.update(channels(35, 0, 0))   # peak grows
    s.update(channels(12, 0, 0))   # ...and does not shrink
    assert close(s.peaks["Flexion"], 35), s.peaks

    row = s.record()
    assert row["trial"] == 1 and close(row["Flexion"], 35)
    assert close(s.peaks["Flexion"], 35), "record() keeps the peaks on screen"

    with tempfile.TemporaryDirectory() as tmp:
        path = s.export_csv(tmp, meta={"patient": "P-001"})
        text = path.read_text(encoding="utf-8")
        assert path.suffix == ".csv" and "Flexion" in text
        assert "# patient: P-001" in text
        lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
        assert len(lines) == 2, lines  # header + one trial

    s.reset()
    assert not s.trials and not s.has_data()
    print("ok  peaks, trials and CSV export")


def test_symmetry_and_means() -> None:
    assert symmetry({}) == {"Rotation": 0.0, "Lateral flexion": 0.0}
    sym = symmetry({
        "Left rotation": 60.0, "Right rotation": 45.0,
        "Left lateral flexion": 30.0, "Right lateral flexion": 30.0,
    })
    assert close(sym["Rotation"], 75.0) and close(sym["Lateral flexion"], 100.0)

    trials = [{m: 10.0 for m in MOVEMENTS}, {m: 20.0 for m in MOVEMENTS}]
    avg = means(trials)
    assert close(avg["Flexion"], 15.0) and close(avg["Left rotation"], 15.0)
    assert close(means([])["Flexion"], 0.0)
    print("ok  symmetry and trial means")


def test_protocol_advances_on_held_end_range() -> None:
    p = Protocol()
    assert p.total == len(STEPS) == 6
    assert p.step.movement == "Flexion"

    ch = {m: 0.0 for m in MOVEMENTS}
    ch["Flexion"] = 30.0
    assert not p.update(ch, 0.0)   # reaching end range
    assert not p.update(ch, 1.0)   # not held long enough yet
    assert p.update(ch, 1.5)       # held >= 1.2 s -> advance
    assert p.step.movement == "Extension"

    neutral = {m: 0.0 for m in MOVEMENTS}
    assert not p.update(neutral, 5.0), "a step with no movement must not advance"

    p.reset()
    for _ in range(p.total):
        p.next()
    assert p.done and p.step is None
    assert not p.update(neutral, 99.0)
    print("ok  guided protocol advances on a held end range")


if __name__ == "__main__":
    test_euler_roundtrip()
    test_calibration_is_relative()
    test_matrix_cleanup()
    test_channels_and_signs()
    test_session_peaks_trials_export()
    test_symmetry_and_means()
    test_protocol_advances_on_held_end_range()
    print("\nAll self-checks passed.")
