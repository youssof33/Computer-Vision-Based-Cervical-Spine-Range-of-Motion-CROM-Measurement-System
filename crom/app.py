"""Assessment workflow: camera loop, buttons, keyboard shortcuts, export."""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from . import config
from .pose import HeadPoseTracker
from .session import MOVEMENTS, Session, channels
from .ui import WINDOW, Ui


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
                self.session.reset_peaks()  # a new assessment starts from zero
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
            self.notify("Reset: calibration and trials cleared.")
        elif key == "export":
            if not self.tracker.has_neutral:
                self.notify("Nothing to export - calibrate and assess first.")
            else:
                path = self.session.export_csv()
                self.notify(f"Saved {path}")
        elif key == "dashboard":
            self.dashboard = not self.dashboard
        return True

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

                ts = int((time.perf_counter() - self._start) * 1000)
                self.pose = self.tracker.process(frame, ts)
                if self.pose.angles is not None:
                    self.live = channels(*self.pose.angles, **self.signs)
                    if self.measuring:
                        self.session.update(self.live)

                if self.message and time.perf_counter() > self.message_until:
                    self.message = ""

                canvas = self.ui.render(
                    frame,
                    pose=self.pose,
                    live=self.live,
                    calibrated=self.tracker.has_neutral,
                    measuring=self.measuring,
                    session=self.session,
                    message=self.message,
                    dashboard=self.dashboard,
                )
                cv2.imshow(WINDOW, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                shortcut = {ord("c"): "calibrate", ord("s"): "start", ord("r"): "record",
                            ord("x"): "reset", ord("e"): "export", ord("d"): "dashboard"}.get(key)
                if shortcut and not self.action(shortcut):
                    break
        finally:
            self.cap.release()
            self.tracker.close()
            cv2.destroyAllWindows()

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
    args = parser.parse_args(argv)

    print("CROM research prototype - head-pose estimate, not a clinical measurement.")
    print("Keys: C calibrate | S start/stop | R record trial | X reset | E export | D dashboard | Q quit")
    CromApp(args).run()


if __name__ == "__main__":
    main()
