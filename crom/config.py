"""Tunable constants.

The sign constants at the bottom are the only values you may need to change.
They depend on the camera and on MediaPipe's canonical-face axes; see README
"Verify the directions" (30 seconds with a mirror).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
REPORTS_DIR = ROOT / "reports"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Mirror the preview only. Landmarks and angles are always computed on the raw
# frame, so mirroring cannot flip the sign of a measurement.
MIRROR_PREVIEW = True

# Smoothing: exponential moving average on the three angles (0..1, lower = smoother).
SMOOTHING_ALPHA = 0.25
# A single-frame change larger than this is treated as a tracking glitch and ignored.
MAX_JUMP_DEG = 30.0
# How long the last good angles are kept after the face is lost.
FACE_LOST_TIMEOUT_S = 0.4
# Frames averaged to build the neutral reference.
CALIBRATION_FRAMES = 12

# Degrees of change between frames above which the head is shown as "MOVING".
# Display only - it never gates a measurement.
MOVE_THRESHOLD_DEG = 0.5
# Frames used for the face-detection rate shown in the panel.
FACE_HISTORY = 60

# Guided protocol: hold the end range this long before the step auto-advances,
# and ignore steps where the patient never moved past this angle.
PROTOCOL_HOLD_S = 1.2
PROTOCOL_MIN_DEG = 5.0

# Raw angle -> clinical channel signs (positive = the movement named on the right).
# Flip a value if the app reports the opposite side / direction.
PITCH_SIGN = 1    # + pitch = looking down  -> Flexion
YAW_SIGN = -1     # + yaw   = turning to the patient's right
ROLL_SIGN = 1     # + roll  = tilting toward the patient's right shoulder
