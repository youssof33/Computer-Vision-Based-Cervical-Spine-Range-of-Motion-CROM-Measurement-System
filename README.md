# CROM — Computer-Vision Cervical Range of Motion (research prototype)

Measures head rotation from a normal laptop webcam and reports the six standard
cervical movements — flexion, extension, left/right lateral flexion, left/right
rotation — in degrees, **relative to a calibrated neutral position**.

> **This is a head-pose estimator, not a validated cervical ROM instrument.**
> The numbers describe how the *head* is oriented. They are not a measurement of
> individual vertebral motion, they have not been compared against a CROM device
> or goniometer, and they must not be used for diagnosis or clinical decisions.
> See [Validation path](#validation-path) for how to get there.

---

## What it does

| Step | Behaviour |
|---|---|
| 1 | Live webcam feed with face-mesh overlay |
| 2 | **Calibrate** records the patient's neutral forward-facing pose as 0° |
| 3 | **Start** begins the assessment; every frame updates six movement channels |
| 4 | The peak (maximum) angle reached in each direction is tracked live |
| 5 | **Record** snapshots the six peaks as a trial and starts a fresh set |
| 6 | **Dashboard** shows all six movements; **Export** writes a CSV report |

Left/right and up/down are derived from a 3D rotation, not from 2D pixel motion
of the nose.

---

## Install (Windows)

Requirements: Python **3.9 – 3.12** (MediaPipe has no 3.13 wheels yet) and a webcam.

**Option A — one click:** double-click `run.bat`. It creates `.venv`, installs
dependencies, and starts the app.

**Option B — manual:**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The first run downloads `models/face_landmarker.task` (~3.7 MB) automatically.
Offline? Download it from
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
and put it in `models/`, or pass `--model path\to\file.task`.

Check the maths without a camera:

```bat
python tests/test_pose.py
```

---

## Run

```bat
python main.py                 :: default camera
python main.py --camera 1      :: second webcam
python main.py --swap-lr       :: swap left/right labels
```

### Controls (buttons or keys)

| Key | Action |
|---|---|
| `C` | Calibrate neutral position (0°) |
| `S` | Start / stop assessment |
| `R` | Record the current peaks as a trial |
| `X` | Reset calibration, peaks and trials |
| `E` | Export CSV to `reports/` |
| `D` | Open / close the results dashboard |
| `Q` / `Esc` | Quit |

### Assessment procedure

1. Sit the patient upright, shoulders relaxed, **shoulders stationary** for the
   whole session. Camera roughly at eye level, 50–80 cm away, face fully visible.
2. Ask them to look straight ahead at the camera and sit still for ~1 second.
   Press **Calibrate**. The panel must say `Neutral: calibrated`.
3. Press **Start**, then have them perform, slowly and to end range:
   flexion (chin to chest), extension (look up), right lateral flexion, left
   lateral flexion, right rotation, left rotation. Return to neutral between
   movements so the channels do not overlap.
4. Press **Stop**. Review the six peaks, press **Record** to store the trial
   (do this once per trial if you want several), then **Export CSV**.

The live panel shows the current dominant movement so you can confirm each
movement is being seen before it counts.

---

## Verify the directions (30 seconds, do this once per camera)

Sign conventions depend on the camera and on MediaPipe's canonical-face axes.
After calibrating, look **down** and check that `Flexion` grows; turn your head
to **your right** and check `Right rotation` grows; tilt toward **your right
shoulder** and check `Right lateral flexion` grows.

If a direction is inverted, flip the matching constant in `crom/config.py`:

```python
PITCH_SIGN = 1   # + pitch = looking down
YAW_SIGN = -1    # + yaw   = patient's right
ROLL_SIGN = 1    # + roll  = patient's right
```

If only left/right are swapped, `--swap-lr` flips yaw and roll at runtime.

Also check that the patient's own right side appears on the **left** of the
preview — that is a correct non-mirrored camera view. The preview is mirrored
for the patient's comfort (`MIRROR_PREVIEW` in `config.py`); angles are always
computed on the unmirrored frame, so mirroring never changes a measurement.

---

## How the angles are computed

MediaPipe's Face Landmarker returns, per frame, a 4×4 **facial transformation
matrix** that maps the canonical face model into camera coordinates.

1. Keep the rotation block `R`, re-orthonormalised with an SVD (removes scale
   drift and small numerical error).
2. At calibration, average the last 12 rotation matrices and project back to a
   rotation → `R_neutral`.
3. Per frame compute the **relative** rotation, expressed in the neutral head
   frame: `R_rel = R_neutralᵀ · R_current`. This removes camera placement and
   the patient's natural resting posture from the measurement.
4. Decompose with a ZYX Euler convention:
   `R_rel = Rz(roll) · Ry(yaw) · Rx(pitch)`, extracting

   ```
   yaw   = asin(-R[2,0])
   pitch = atan2(R[2,1], R[2,2])
   roll  = atan2(R[1,0], R[0,0])
   ```

   Pitch is a nod (about the camera's x axis), yaw a turn (y), roll a tilt (z).
   This is an *intrinsic* ZYX decomposition, i.e. rotations about the moving
   head axes — the natural reading for cervical motion.
5. Split the signed angles into the six non-negative channels
   (`Flexion = max(pitch, 0)`, `Extension = max(-pitch, 0)`, …) and track peaks.

**Robustness**

- Exponential moving average on the angles (`SMOOTHING_ALPHA = 0.25`).
- Frame-to-frame jumps over `MAX_JUMP_DEG = 30°` are treated as tracking
  glitches and ignored.
- If the face is lost, the last good angles are held for 0.4 s, then the panel
  shows `FACE LOST` and peak tracking stops instead of drifting.
- Calibration requires several recent detections, so a one-frame false positive
  cannot become the zero reference.

Known limits: yaw and pitch are not perfectly independent near ±90° (Euler
gimbal), so keep the head inside a normal clinical range; a single camera cannot
see pure axial rotation of the head, so "rotation" is really the visible
combination of yaw and a little roll; landmark noise is roughly ±1–2° after
smoothing, which is small next to a typical 40–60° range but not negligible.

---

## Output

`reports/crom_report_YYYYmmdd_HHMMSS.csv`

```
# CROM research prototype - head-pose estimate, not a validated clinical measurement.
# Angles in degrees, relative to the calibrated neutral position.
trial,timestamp,Flexion,Extension,Right lateral flexion,Left lateral flexion,Right rotation,Left rotation
1,2026-11-21 14:32:05,48.2,51.7,32.4,30.9,63.1,66.8
```

One row per recorded trial. If you export without recording, the current peaks
are written as a single `current` row. The two `#` lines are the caveat that
travels with the data; `pandas.read_csv(path, comment="#")` skips them.

CSV only — a PDF report was not implemented (it would add a rendering
dependency for no extra data).

---

## Project layout

```
main.py              entry point
run.bat              Windows one-click setup + run
crom/
  config.py          camera, smoothing, and the three sign constants
  pose.py            MediaPipe wrapper, rotation maths, calibration, smoothing
  session.py         six channels, peaks, trials, CSV export
  ui.py              all OpenCV drawing (video tile, panel, dashboard)
  app.py             camera loop, buttons, keyboard, workflow
tests/test_pose.py   assert-based self-check, no camera needed
models/              face_landmarker.task (downloaded on first run)
reports/             exported CSV files
```

---

## Validation path

To turn this into something that could be compared with a clinical instrument:

1. **Protocol.** Same seat, same camera distance, same instruction script for
   every subject; three repetitions per movement; record which repetition is
   which.
2. **Reference standard.** Measure the same movements with a CROM device or a
   gravity/inclinometer goniometer by a trained examiner, ideally blinded to the
   video result.
3. **Paired data.** Export one CSV per subject/trial and join it to the
   reference table on subject + trial id.
4. **Analysis.** Bland–Altman bias and limits of agreement, ICC(2,1), standard
   error of measurement, and per-movement breakdown. Report the camera model,
   resolution, distance, lighting, and glasses/beard/makeup conditions.
5. **Then** decide thresholds and whether the error is acceptable for the
   intended use.

Until that exists, treat every number as a *head-orientation estimate in
degrees*, useful for tracking progress within one session on one setup.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot open camera 0` | Close Zoom/Teams/Camera app; try `--camera 1`; check Windows camera privacy settings |
| No face found | More light on the face, remove backlight, sit 50–80 cm away, remove mask |
| Angles jitter | Lower `SMOOTHING_ALPHA` (e.g. 0.15), keep the laptop still, improve lighting |
| Values lag behind fast movement | Raise `SMOOTHING_ALPHA` (e.g. 0.4) |
| Left/right reversed | `--swap-lr`, or flip `YAW_SIGN` / `ROLL_SIGN` in `config.py` |
| Up/down reversed | Flip `PITCH_SIGN` in `config.py` |
| `pip install mediapipe` fails | Use Python 3.9–3.12 (64-bit), then `pip install --upgrade pip` |
| Low FPS | Lower `FRAME_WIDTH`/`FRAME_HEIGHT` in `config.py`, close other apps |
