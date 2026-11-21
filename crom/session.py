"""Assessment state: six movement channels, peak tracking, trials, CSV export."""

from __future__ import annotations

import csv
import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

from . import config

MOVEMENTS = (
    "Flexion",
    "Extension",
    "Right lateral flexion",
    "Left lateral flexion",
    "Right rotation",
    "Left rotation",
)

# What each movement means, for the dashboard.
DIRECTIONS = {
    "Flexion": "chin toward chest",
    "Extension": "look up",
    "Right lateral flexion": "right ear to right shoulder",
    "Left lateral flexion": "left ear to left shoulder",
    "Right rotation": "turn head right",
    "Left rotation": "turn head left",
}


def channels(
    pitch: float,
    yaw: float,
    roll: float,
    *,
    pitch_sign: float | None = None,
    yaw_sign: float | None = None,
    roll_sign: float | None = None,
) -> dict[str, float]:
    """Split the signed pose angles into the six CROM channels (degrees, >= 0)."""
    ps = config.PITCH_SIGN if pitch_sign is None else pitch_sign
    ys = config.YAW_SIGN if yaw_sign is None else yaw_sign
    rs = config.ROLL_SIGN if roll_sign is None else roll_sign
    p, y, r = ps * pitch, ys * yaw, rs * roll
    return {
        "Flexion": max(p, 0.0),
        "Extension": max(-p, 0.0),
        "Right lateral flexion": max(r, 0.0),
        "Left lateral flexion": max(-r, 0.0),
        "Right rotation": max(y, 0.0),
        "Left rotation": max(-y, 0.0),
    }


def dominant(ch: dict[str, float], threshold: float = 0.5) -> tuple[str, float]:
    """Largest active channel, or ('Neutral', 0) when the head is near neutral."""
    name = max(ch, key=lambda k: ch[k])
    return (name, ch[name]) if ch[name] >= threshold else ("Neutral", 0.0)


@dataclass
class Session:
    """One assessment. Peaks are the maximum reached since the last reset/record."""

    peaks: dict[str, float] = field(
        default_factory=lambda: {m: 0.0 for m in MOVEMENTS}
    )
    trials: list[dict] = field(default_factory=list)

    def update(self, ch: dict[str, float]) -> None:
        for name, value in ch.items():
            if value > self.peaks[name]:
                self.peaks[name] = value

    def record(self) -> dict:
        """Snapshot the current peaks as a trial and start a fresh set of peaks."""
        row = {
            "trial": len(self.trials) + 1,
            "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **{m: round(self.peaks[m], 1) for m in MOVEMENTS},
        }
        self.trials.append(row)
        self.reset_peaks()
        return row

    def reset_peaks(self) -> None:
        self.peaks = {m: 0.0 for m in MOVEMENTS}

    def reset(self) -> None:
        self.reset_peaks()
        self.trials.clear()

    def has_data(self) -> bool:
        return bool(self.trials) or any(v > 0.05 for v in self.peaks.values())

    # -- reporting ---------------------------------------------------------
    def rows(self) -> list[dict]:
        """Trials, or the un-recorded current peaks as a single row."""
        if self.trials:
            return self.trials
        return [
            {
                "trial": "current",
                "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **{m: round(self.peaks[m], 1) for m in MOVEMENTS},
            }
        ]

    def export_csv(self, directory: Path | str | None = None) -> Path:
        """Write the summary table. One row per trial, one column per movement."""
        out_dir = Path(directory or config.REPORTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"crom_report_{_dt.datetime.now():%Y%m%d_%H%M%S}.csv"

        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write("# CROM research prototype - head-pose estimate, not a validated clinical measurement.\n")
            fh.write("# Angles in degrees, relative to the calibrated neutral position.\n")
            writer = csv.DictWriter(fh, fieldnames=["trial", "timestamp", *MOVEMENTS])
            writer.writeheader()
            writer.writerows(self.rows())
        return path
