"""Assessment workflow: camera loop, buttons, keyboard shortcuts, export."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2

from . import __version__, config
from .pose import HeadPoseTracker
from .protocol import Protocol
from .session import MOVEMENTS, Session, channels
from .ui import WINDOW, Ui, ViewState


class CromApp:
    def __init__(self, args: argparse.Namespace) -> None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
        self.cap = cv2.VideoCapture(args.camera, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise SystemExit(
                f"Cannot open camera {args.camera}. Close other apps using it, "
                "or pass --camera 1."
            )

        self.tracker = HeadPoseTracker(args.model)
        self.session = Session()
        self.ui = Ui()
        self.protocol: Protocol | None = Protocol() if args.guided else None

        self.patient = args.patient
        self.notes = args.notes
        self.camera_index = args.camera
        self.pose_shape: tuple[int, int] | None = None
        self.mirror = config.MIRROR_PREVIEW and not args.no_mirror
        self.signs = {
            "pitch_sign": config.PITCH_SIGN,
            "yaw_sign": config.YAW_SIGN * (-1 if args.swap_lr else 1),
            "roll_sign": config.ROLL_SIGN * (-1 if args.swap_lr else 1),
        }

        self.measuring = False
        self.dashboard = False
        self.pose = None
        self.live = {m: 0.0 for m in MOVEMENTS}
        self.message = ""
        self.message_until = 0.0

        self._fps = 0.0
        self._last_frame_t: float | None = None
        self._face = deque(maxlen=config.FACE_HISTORY)
        self._prev_angles: tuple[float, float, float] | None = None
        self._moving = False
        self._start = time.perf_counter()

    # -- helpers -----------------------------------------------------------
    def notify(self, text: str) -> None:
        self.message = text
        self.message_until = time.perf_counter() + 3.0
        print(text)

    def action(self, key: str) -> bool:
        """Run one action. Returns False to quit."""
        if key == "quit":
            return False
        if key == "calibrate":
            if self.tracker.calibrate():
                self.session.reset_peaks()
                self.measuring = False
                if self.protocol:
                    self.protocol.reset()
                self.notify("Neutral position recorded as 0 deg.")
            else:
                self.notify("Face not detected yet - look at the camera, then retry.")
        elif key == "start":
            if not self.tracker.has_neutral:
                self.notify("Calibrate the neutral position first.")
            elif self.measuring:
                self.measuring = False
                self.notify("Assessment stopped - peaks kept for review/export.")
            else:
                self.measuring = True
                self.session.reset_peaks()
                if self.protocol:
                    self.protocol.reset()
                    self.notify(f"Protocol started: {self.protocol.step.movement}.")
                else:
                    self.notify("Assessment started.")
        elif key == "record":
            if not self.tracker.has_neutral:
                self.notify("Calibrate first.")
            else:
                row = self.session.record()
                self.notify(f"Trial {row['trial']} recorded ({len(self.session.trials)} total).")
        elif key == "reset":
            self.session.reset()
            self.tracker.clear_neutral()
            self.measuring = False
            if self.protocol:
                self.protocol.reset()
            self.notify("Reset: calibration and trials cleared.")
        elif key == "export":
            if not self.tracker.has_neutral:
                self.notify("Nothing to export - calibrate and assess first.")
            else:
                self.notify(f"Saved {self.session.export_csv(meta=self._meta())}")
        elif key == "dashboard":
            self.dashboard = not self.dashboard
        elif key == "protocol":
            self.protocol = None if self.protocol else Protocol()
            self.notify(
                "Guided protocol on - press Start." if self.protocol
                else "Guided protocol off."
            )
        elif key == "mirror":
            self.mirror = not self.mirror
            self.notify(f"Preview mirroring {'on' if self.mirror else 'off'}.")
        elif key == "skip":
            if self.protocol and self.measuring:
                self.protocol.next()
                if self.protocol.done:
                    self.measuring = False
                    self.notify("Protocol complete - review the peaks, then Record.")
                else:
                    self.notify(f"Next: {self.protocol.step.movement}.")
        return True

    def _meta(self) -> dict[str, str]:
        height, width = self.pose_shape or (0, 0)
        meta = {
            "patient": self.patient or "anonymous",
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "camera": f"{self.camera_index} at {width}x{height}",
            "software": f"crom {__version__}",
        }
        if self.notes:
            meta["notes"] = self.notes
        return meta

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    self.notify("Camera read failed - stopping.")
                    break
                self.pose_shape = frame.shape[:2]

                now = time.perf_counter()
                if self._last_frame_t is not None:
                    dt = max(now - self._last_frame_t, 1e-3)
                    self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
                self._last_frame_t = now

                ts = int((now - self._start) * 1000)
                self.pose = self.tracker.process(frame, ts)
                self._face.append(1 if self.pose.landmarks else 0)
                self._update_angles(now)

                if self.message and now > self.message_until:
                    self.message = ""

                canvas = self.ui.render(frame, self._view())
                cv2.imshow(WINDOW, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if not self._on_key(key):
                    break
        finally:
            self.cap.release()
            self.tracker.close()
            cv2.destroyAllWindows()

    def _update_angles(self, now: float) -> None:
        angles = self.pose.angles
        if angles is None:
            self._prev_angles = None
            self._moving = False
            return
        self._moving = self._prev_angles is not None and max(
            abs(a - b) for a, b in zip(angles, self._prev_angles)
        ) > config.MOVE_THRESHOLD_DEG
        self.live = channels(*angles, **self.signs)
        if self.measuring:
            self.session.update(self.live)
            if self.protocol and self.protocol.update(self.live, now):
                if self.protocol.done:
                    self.measuring = False
                    self.notify("Protocol complete - review the peaks, then Record.")
                else:
                    self.notify(f"Next: {self.protocol.step.movement}.")
        self._prev_angles = angles

    def _view(self) -> ViewState:
        return ViewState(
            pose=self.pose,
            live=self.live,
            calibrated=self.tracker.has_neutral,
            measuring=self.measuring,
            session=self.session,
            message=self.message,
            dashboard=self.dashboard,
            fps=self._fps,
            face_ratio=100.0 * sum(self._face) / len(self._face) if self._face else 0.0,
            moving=self._moving,
            protocol=self.protocol,
            mirror=self.mirror,
        )

    def _on_key(self, key: int) -> bool:
        shortcuts = {
            ord("c"): "calibrate", ord("s"): "start", ord("r"): "record",
            ord("x"): "reset", ord("e"): "export", ord("d"): "dashboard",
            ord("g"): "protocol", ord("m"): "mirror", ord(" "): "skip",
        }
        action = shortcuts.get(key)
        return self.action(action) if action else True

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            key = self.ui.hit(x, y)
            if key and not self.action(key):
                raise SystemExit(0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CROM measurement research prototype")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="webcam index (default: %(default)s)")
    parser.add_argument("--model", default=None, help="path to face_landmarker.task")
    parser.add_argument("--swap-lr", action="store_true",
                        help="flip left/right if the reported side is mirrored")
    parser.add_argument("--no-mirror", action="store_true",
                        help="show the raw (unmirrored) camera preview")
    parser.add_argument("--guided", action="store_true",
                        help="start with the guided six-step protocol enabled")
    parser.add_argument("--patient", default="", help="patient/reference id for the CSV")
    parser.add_argument("--notes", default="", help="free-text notes for the CSV")
    args = parser.parse_args(argv)

    print(f"CROM {__version__} - head-pose estimate, not a clinical measurement.")
    print("C calibrate | S start/stop | R record | X reset | E export | D dashboard")
    print("G guided protocol | Space next step | M mirror | Q quit")
    CromApp(args).run()


if __name__ == "__main__":
    main()
