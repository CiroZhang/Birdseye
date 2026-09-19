# Birdseye

Monocular 3D reconstruction and analysis of badminton rallies from single-camera broadcast video —
court geometry, player pose, hit detection, and shuttle trajectory, each stage driven by our own
detectors rather than manual annotation.

## Demo

```bash
cd demo && ./serve.sh
```

Opens a local page with three interactive views: live pose detection on broadcast footage,
self-calibrated 3D player reconstruction validated against a real drone dataset, and physics-based
shuttle trajectory reconstruction per shot.

## Results

| Stage | Metric | Ours | Best published baseline |
|---|---|---|---|
| Court calibration | Mean IoU / MPE | **0.9875 / 2.45px** | 0.97 IoU (MonoTrack, CVPRW 2022) |
| Hit detection (who + when) | F1 | **92.7%** | 86.2% (Chen et al., Sensors 2024) |
| Shuttle trajectory | Median reprojection error | 11.9px (9,135 shots) | — (no directly comparable baseline) |

Full method and evaluation detail lives in each stage's `MANIFEST.md`. Different rows above are
evaluated on different published test sets — see each section's manifest for the exact comparison
basis.

## Pipeline

1. **Court** (`1:Court/`) — MonoTrack's court-line detector finds 22 court-plane points per video;
   a RANSAC joint solve recovers camera focal length + pose from those points alone. The net is not
   detected directly — its 3D position is reprojected from known court geometry through the solved
   camera.
2. **Player** (`2:Player/`) — YOLOv8x + YOLOv8x-pose locate both players, filtered to the real court
   boundary using the calibrated camera. VIMO (from TRAM) reconstructs 3D pose per frame, fed our
   camera directly instead of its own SLAM estimate; a ray-cast ground correction removes monocular
   depth drift.
3. **Action** (`3:Action/`) — A transformer over 3D skeleton motion (root-relative position,
   velocity, acceleration, both players) predicts per-frame hit probability and shot type, trained
   on BFMD's hit-timing labels.
4. **Shuttle** (`4:Shuttle/`) — A physics-based (gravity + quadratic drag) trajectory fit per shot,
   anchored at both players' hand positions at the bounding hits to resolve monocular depth
   ambiguity.

## Setup

Requires two sibling repos alongside this one (same parent directory):

```
Badminton LLM/
├── Project/            (this repo)
├── Dataset/BFMD/        BFMD_data — annotations + video
└── monotrack/            court-detection binary (build/detect)
```

Paths resolve relative to each script by default; override with `BFMD_DATA_ROOT` /
`BFMD_SELFDETECTED_ROOT` env vars if your layout differs.

```bash
python3 main.py [match_name]   # runs the full pipeline end to end on one match
```

## Repo layout

```
1:Court/        camera calibration
2:Player/       player detection + 3D pose
3:Action/       hit detection / shot classification
4:Shuttle/      shuttle trajectory reconstruction
5.Evalutation/  supplementary calibration/validation scripts
demo/           interactive demo site
result/         evaluation outputs
```
