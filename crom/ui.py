"""All OpenCV drawing. The UI renders a `ViewState`; it owns no measurement logic."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .session import DIRECTIONS, MOVEMENTS, Session, dominant, means, symmetry

WINDOW = "CROM - Cervical Range of Motion (research prototype)"

CANVAS_W, CANVAS_H = 1280, 720
VIDEO_W = 860
PANEL_X = VIDEO_W
PAD = 20
X0 = PANEL_X + PAD
X1 = X0 + 185
RIGHT = PANEL_X + 380

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
DOTS = (1, 10, 152, 33, 263, 61, 291, 199)


@dataclass
class ViewState:
    """Everything the screen needs for one frame."""

    pose: object | None = None
    live: dict[str, float] = field(default_factory=lambda: {m: 0.0 for m in MOVEMENTS})
    calibrated: bool = False
    measuring: bool = False
    session: Session = field(default_factory=Session)
    message: str = ""
    dashboard: bool = False
    fps: float = 0.0
    face_ratio: float = 0.0
    moving: bool = False
    protocol: object | None = None
    mirror: bool = True


def _text(img, s, x, y, scale=0.5, color=FG, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def _text_right(img, s, x_right, y, scale=0.5, color=FG, thick=1):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    _text(img, s, x_right - tw, y, scale, color, thick)


class Ui:
    def __init__(self) -> None:
        self.buttons: dict[str, tuple[int, int, int, int]] = {}

    # -- public ------------------------------------------------------------
    def render(self, frame: np.ndarray, state: ViewState) -> np.ndarray:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
        canvas[:, :VIDEO_W] = self._video_tile(
            frame, state.pose.landmarks if state.pose else None, state.mirror
        )

        if state.pose is not None and state.pose.stale:
            self._banner(canvas, "FACE LOST - hold still, keep your face in view", WARN)
        elif state.pose is not None and state.pose.angles is None and not state.dashboard:
            self._banner(canvas, "Searching for a face...", MUTED)

        if state.message:
            _text(canvas, state.message, 16, CANVAS_H - 18, 0.55, ACCENT)

        if state.dashboard:
            self._dashboard(canvas, state)
        else:
            self._panel(canvas, state)
        return canvas

    def hit(self, x: int, y: int) -> str | None:
        for key, (bx, by, bw, bh) in self.buttons.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return key
        return None

    # -- pieces ------------------------------------------------------------
    def _video_tile(self, frame: np.ndarray, landmarks, mirror: bool) -> np.ndarray:
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
            for i in DOTS:
                cv2.circle(img, tuple(pts[i]), r, (255, 255, 255), -1, cv2.LINE_AA)

        if mirror:
            img = cv2.flip(img, 1)

        ox, oy = (VIDEO_W - nw) // 2, (CANVAS_H - nh) // 2
        tile[oy:oy + nh, ox:ox + nw] = img
        return tile

    def _banner(self, canvas, text, color):
        cv2.rectangle(canvas, (0, 0), (VIDEO_W, 44), (20, 20, 24), -1)
        _text(canvas, text, 16, 30, 0.65, color, 2)

    def _panel(self, canvas, s: ViewState) -> None:
        cv2.rectangle(canvas, (PANEL_X, 0), (CANVAS_W, CANVAS_H), PANEL_BG, -1)
        session = s.session

        _text(canvas, "CROM prototype", X0, 38, 0.8, ACCENT, 2)
        _text(canvas, "Neutral: calibrated" if s.calibrated else "Neutral: NOT calibrated",
              X0, 62, 0.5, GOOD if s.calibrated else WARN)
        if s.measuring:
            _text(canvas, "ASSESSING", X0 + 200, 62, 0.5, GOOD)

        hold = "HOLD" if not s.moving else "MOVING"
        _text(canvas, f"{s.fps:4.0f} fps   face {s.face_ratio:3.0f}%", X0, 86, 0.42, MUTED)
        _text_right(canvas, hold, RIGHT, 86, 0.42, GOOD if not s.moving else ACCENT)

        _text(canvas, "Angles vs neutral (deg)" if s.calibrated else "Absolute pose (deg)",
              X0, 112, 0.42, MUTED)
        angles = s.pose.angles if s.pose is not None else None
        for i, name in enumerate(("Pitch", "Yaw", "Roll")):
            y = 138 + i * 24
            _text(canvas, name, X0, y, 0.5, MUTED)
            _text(canvas, "--" if angles is None else f"{angles[i]:+.1f}", X0 + 110, y, 0.55, FG)

        self._movement_block(canvas, s)
        self._peaks(canvas, s)

        if session.trials:
            _text(canvas, f"Trials recorded: {len(session.trials)}", X0, 434, 0.42, GOOD)

        self._buttons(canvas, s)
        _text(canvas, "Research prototype: head-pose estimate,", X0, 678, 0.4, MUTED)
        _text(canvas, "not a validated clinical measurement.", X0, 696, 0.4, MUTED)

    def _movement_block(self, canvas, s: ViewState) -> None:
        p = s.protocol
        if p is not None:
            _text(canvas, "Protocol  step %d/%d" % (min(p.index + 1, p.total), p.total),
                  X0, 212, 0.42, MUTED)
            if p.done:
                _text(canvas, "Protocol complete", X0, 238, 0.65, GOOD, 2)
                return
            _text(canvas, p.step.movement, X0, 238, 0.6, ACCENT, 2)
            _text_right(canvas, f"{s.live[p.step.movement]:.1f} deg", RIGHT, 238, 0.55, FG)
            _text(canvas, p.step.instruction, X0, 258, 0.4, MUTED)
            frac = p.index / p.total
            cv2.rectangle(canvas, (X0, 265), (RIGHT, 268), (60, 60, 70), -1)
            cv2.rectangle(canvas, (X0, 265), (X0 + int((RIGHT - X0) * frac), 268), ACCENT, -1)
            return

        _text(canvas, "Current movement", X0, 212, 0.42, MUTED)
        name, value = dominant(s.live) if s.calibrated else ("Neutral", 0.0)
        _text(canvas, f"{name}  {value:.1f} deg", X0, 238, 0.6,
              FG if name == "Neutral" else ACCENT, 2)

    def _peaks(self, canvas, s: ViewState) -> None:
        _text(canvas, "Peak range of motion (deg)", X0, 292, 0.42, MUTED)
        for i, m in enumerate(MOVEMENTS):
            y = 316 + i * 20
            _text(canvas, m, X0, y, 0.42, MUTED)
            _text_right(canvas, f"{s.session.peaks[m]:.1f}", RIGHT, y, 0.42, FG)

    def _buttons(self, canvas, s: ViewState) -> None:
        rows = (456, 502, 548, 594)
        spec = (
            ("calibrate", "Calibrate  [C]", True),
            ("start", "Stop  [S]" if s.measuring else "Start  [S]", s.calibrated),
            ("record", "Record  [R]", s.calibrated),
            ("reset", "Reset  [X]", True),
            ("export", "Export CSV  [E]", s.calibrated),
            ("dashboard", "Dashboard  [D]", True),
            ("protocol", "Protocol  [G]", s.calibrated),
            ("quit", "Quit  [Q]", True),
        )
        self.buttons = {}
        for i, (key, text, enabled) in enumerate(spec):
            row, col = divmod(i, 2)
            rect = (X0 if col == 0 else X1, rows[row], 175, 40)
            self.buttons[key] = rect
            self._button(canvas, rect, text, enabled)

    def _button(self, canvas, rect, text, enabled=True) -> None:
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), BTN if enabled else BTN_OFF, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 92), 1)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
        _text(canvas, text, x + (w - tw) // 2, y + (h + th) // 2, 0.42,
              FG if enabled else (110, 110, 120))

    def _dashboard(self, canvas, s: ViewState) -> None:
        canvas[:] = BG
        session = s.session
        _text(canvas, "Results dashboard", 70, 90, 1.0, ACCENT, 2)
        _text(canvas, f"Trials recorded: {len(session.trials)}", 70, 122, 0.5, MUTED)

        cols = (70, 360, 500, 640, 820)
        for label, x in zip(("Movement", "Peak", "Mean", "Last trial", "What it means"), cols):
            _text(canvas, label, x, 180, 0.5, MUTED)
        cv2.line(canvas, (70, 194), (CANVAS_W - 70, 194), (80, 80, 92), 1)

        avg = means(session.trials)
        last = session.trials[-1] if session.trials else None
        for i, m in enumerate(MOVEMENTS):
            y = 228 + i * 34
            _text(canvas, m, cols[0], y, 0.6, FG)
            _text(canvas, f"{session.peaks[m]:.1f}", cols[1], y, 0.6, ACCENT)
            _text(canvas, "--" if not session.trials else f"{avg[m]:.1f}", cols[2], y, 0.6, FG)
            _text(canvas, "--" if last is None else f"{last[m]:.1f}", cols[3], y, 0.6, FG)
            _text(canvas, DIRECTIONS[m], cols[4], y, 0.45, MUTED)

        ref = last if last else session.peaks
        sym = symmetry(ref)
        parts = []
        for label, left, right in (
            ("rotation", "Left rotation", "Right rotation"),
            ("lateral flexion", "Left lateral flexion", "Right lateral flexion"),
        ):
            measured = min(ref[left], ref[right]) >= 5.0
            parts.append(f"{label} " + (f"{sym[label]:.0f}%" if measured else "n/a"))
        _text(canvas, "L/R symmetry (100% = equal sides):", 70, 450, 0.45, MUTED)
        _text(canvas, "    ".join(parts), 400, 450, 0.45, FG)

        _text(canvas, "Peak = maximum reached since Start/Record, relative to the calibrated neutral pose.",
              70, 490, 0.42, MUTED)
        _text(canvas, "Head-pose estimate, not a validated cervical ROM measurement.",
              70, 514, 0.42, WARN)

        self.buttons = {
            "export": (70, 570, 250, 48),
            "reset": (350, 570, 250, 48),
            "dashboard": (630, 570, 250, 48),
        }
        self._button(canvas, self.buttons["export"], "Export CSV  [E]")
        self._button(canvas, self.buttons["reset"], "Reset  [X]")
        self._button(canvas, self.buttons["dashboard"], "Close  [D]")
