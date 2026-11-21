"""All OpenCV drawing. The UI renders the state it is given; it owns no logic."""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import config
from .session import DIRECTIONS, MOVEMENTS, dominant

WINDOW = "CROM - Cervical Range of Motion (research prototype)"

CANVAS_W, CANVAS_H = 1280, 720
VIDEO_W = 860
PANEL_X = VIDEO_W
PAD = 20

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (26, 26, 30)
PANEL_BG = (36, 36, 42)
FG = (238, 238, 238)
MUTED = (150, 150, 160)
ACCENT = (250, 200, 90)
GOOD = (130, 220, 130)
WARN = (90, 170, 255)
BTN = (62, 62, 72)
BTN_OFF = (46, 46, 52)

# A readable subset of the 468-point face mesh.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
        267, 0, 37, 39, 40, 185]
NOSE = [168, 6, 197, 195, 5, 4, 1, 2]


def _text(img, s, x, y, scale=0.5, color=FG, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def _text_right(img, s, x_right, y, scale=0.5, color=FG, thick=1):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    _text(img, s, x_right - tw, y, scale, color, thick)


class Ui:
    def __init__(self) -> None:
        self.buttons: dict[str, tuple[int, int, int, int]] = {}

    # -- public ------------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        *,
        pose,
        live: dict[str, float],
        calibrated: bool,
        measuring: bool,
        session,
        message: str = "",
        dashboard: bool = False,
    ) -> np.ndarray:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
        canvas[:, :VIDEO_W] = self._video_tile(frame, pose.landmarks if pose else None)

        if pose is not None and pose.stale:
            self._banner(canvas, "FACE LOST - hold still, keep your face in view", WARN)
        elif pose is not None and pose.angles is None and not dashboard:
            self._banner(canvas, "Searching for a face...", MUTED)

        if message:
            _text(canvas, message, 16, CANVAS_H - 18, 0.55, ACCENT)

        if dashboard:
            self._dashboard(canvas, session)
        else:
            self._panel(canvas, pose, live, calibrated, measuring, session)
        return canvas

    def hit(self, x: int, y: int) -> str | None:
        for key, (bx, by, bw, bh) in self.buttons.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return key
        return None

    # -- pieces ------------------------------------------------------------
    def _video_tile(self, frame: np.ndarray, landmarks) -> np.ndarray:
        tile = np.full((CANVAS_H, VIDEO_W, 3), BG, np.uint8)
        h, w = frame.shape[:2]
        scale = min(VIDEO_W / w, CANVAS_H / h)
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        if landmarks:
            pts = np.array([(lm.x * nw, lm.y * nh) for lm in landmarks], np.int32)
            r = max(1, int(nw / 400))
            for chain, color in (
                (FACE_OVAL, (120, 200, 140)),
                (LEFT_EYE, ACCENT), (RIGHT_EYE, ACCENT),
                (LIPS, (120, 160, 250)), (NOSE, (200, 200, 200)),
            ):
                cv2.polylines(img, [pts[chain]], False, color, max(1, r - 1), cv2.LINE_AA)
            for i in (1, 10, 152, 33, 263, 61, 291, 199):
                cv2.circle(img, tuple(pts[i]), r, (255, 255, 255), -1, cv2.LINE_AA)

        if config.MIRROR_PREVIEW:
            img = cv2.flip(img, 1)

        ox, oy = (VIDEO_W - nw) // 2, (CANVAS_H - nh) // 2
        tile[oy:oy + nh, ox:ox + nw] = img
        return tile

    def _banner(self, canvas, text, color):
        cv2.rectangle(canvas, (0, 0), (VIDEO_W, 44), (20, 20, 24), -1)
        _text(canvas, text, 16, 30, 0.65, color, 2)

    def _panel(self, canvas, pose, live, calibrated, measuring, session) -> None:
        x = PANEL_X + PAD
        cv2.rectangle(canvas, (PANEL_X, 0), (CANVAS_W, CANVAS_H), PANEL_BG, -1)

        _text(canvas, "CROM prototype", x, 38, 0.8, ACCENT, 2)
        if not calibrated:
            _text(canvas, "Neutral: NOT calibrated", x, 68, 0.5, WARN)
        else:
            _text(canvas, "Neutral: calibrated", x, 68, 0.5, GOOD)
        if measuring:
            _text(canvas, "ASSESSING", x + 200, 68, 0.5, GOOD)

        label = "Angles vs neutral (deg)" if calibrated else "Absolute pose (deg)"
        _text(canvas, label, x, 100, 0.45, MUTED)
        angles = pose.angles if pose is not None else None
        for i, (name, key) in enumerate((("Pitch", 0), ("Yaw", 1), ("Roll", 2))):
            y = 128 + i * 26
            _text(canvas, name, x, y, 0.5, MUTED)
            value = "--" if angles is None else f"{angles[key]:+.1f}"
            _text(canvas, value, x + 110, y, 0.55, FG)

        _text(canvas, "Current movement", x, 216, 0.45, MUTED)
        name, value = dominant(live) if calibrated else ("Neutral", 0.0)
        _text(canvas, f"{name}  {value:.1f} deg", x, 244, 0.65, FG if name == "Neutral" else ACCENT, 2)

        _text(canvas, "Peak range of motion (deg)", x, 282, 0.45, MUTED)
        for i, m in enumerate(MOVEMENTS):
            y = 310 + i * 22
            _text(canvas, m, x, y, 0.45, MUTED)
            _text_right(canvas, f"{session.peaks[m]:.1f}", PANEL_X + 380, y, 0.45, FG)

        if session.trials:
            _text(canvas, f"Trials recorded: {len(session.trials)}", x, 448 - 8, 0.45, GOOD)

        self._buttons(canvas, calibrated, measuring)

    def _buttons(self, canvas, calibrated, measuring) -> None:
        x0 = PANEL_X + PAD
        x1 = x0 + 185
        rows = [448, 496, 544]
        spec = [
            ("calibrate", "Calibrate  [C]", True),
            ("start", "Stop  [S]" if measuring else "Start  [S]", calibrated),
            ("record", "Record  [R]", calibrated),
            ("reset", "Reset  [X]", True),
            ("export", "Export CSV  [E]", calibrated),
            ("dashboard", "Dashboard  [D]", True),
        ]
        self.buttons = {}
        for i, (key, text, enabled) in enumerate(spec):
            row, col = divmod(i, 2)
            rect = (x0 if col == 0 else x1, rows[row], 175, 40)
            self.buttons[key] = rect
            self._button(canvas, rect, text, enabled)

        self.buttons["quit"] = (x0, 596, 360, 40)
        self._button(canvas, self.buttons["quit"], "Quit  [Q]", True)

        _text(canvas, "Research prototype: head-pose estimate,", x0, 678, 0.4, MUTED)
        _text(canvas, "not a validated clinical measurement.", x0, 696, 0.4, MUTED)

    def _button(self, canvas, rect, text, enabled=True) -> None:
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), BTN if enabled else BTN_OFF, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 92), 1)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
        _text(canvas, text, x + (w - tw) // 2, y + (h + th) // 2, 0.42,
              FG if enabled else (110, 110, 120))

    def _dashboard(self, canvas, session) -> None:
        canvas[:] = BG
        _text(canvas, "Results dashboard", 70, 90, 1.0, ACCENT, 2)
        _text(canvas, f"Trials recorded: {len(session.trials)}   "
                      f"Generated {time.strftime('%Y-%m-%d %H:%M')}", 70, 122, 0.5, MUTED)

        cols = (70, 430, 640, 900)
        _text(canvas, "Movement", cols[0], 180, 0.55, MUTED)
        _text(canvas, "Peak (deg)", cols[1], 180, 0.55, MUTED)
        _text(canvas, "Last trial", cols[2], 180, 0.55, MUTED)
        _text(canvas, "What it means", cols[3], 180, 0.55, MUTED)
        cv2.line(canvas, (70, 194), (CANVAS_W - 70, 194), (80, 80, 92), 1)

        last = session.trials[-1] if session.trials else None
        for i, m in enumerate(MOVEMENTS):
            y = 228 + i * 34
            _text(canvas, m, cols[0], y, 0.6, FG)
            _text(canvas, f"{session.peaks[m]:.1f}", cols[1], y, 0.6, ACCENT)
            _text(canvas, "--" if last is None else f"{last[m]:.1f}", cols[2], y, 0.6, FG)
            _text(canvas, DIRECTIONS[m], cols[3], y, 0.45, MUTED)

        _text(canvas, "Peak = maximum reached since Start/Record. Values are relative to the calibrated neutral pose.",
              70, 470, 0.45, MUTED)
        _text(canvas, "This is a head-pose estimate, not a validated cervical ROM measurement.",
              70, 496, 0.45, WARN)

        self.buttons = {
            "export": (70, 560, 250, 48),
            "reset": (350, 560, 250, 48),
            "dashboard": (630, 560, 250, 48),
        }
        self._button(canvas, self.buttons["export"], "Export CSV  [E]")
        self._button(canvas, self.buttons["reset"], "Reset  [X]")
        self._button(canvas, self.buttons["dashboard"], "Close  [D]")
