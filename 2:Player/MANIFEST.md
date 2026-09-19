# 2:Player -- file guide

- **run_vimo_batch_v2.py** -- the original validated method (48-match dataset, match1/5/9 etc.):
  hybrid YOLOv8x (box) + YOLOv8x-pose (ankle keypoints) player detection with court-boundary
  filtering (rejects referee/crowd), + VIMO 3D reconstruction fed our own calibrated camera.

- **run_vimo_bfmd_match.py** -- SUPERSEDED. The original BFMD shortcut: took BFMD's own annotated
  `player_bbox` labels directly as VIMO's input boxes instead of running our own detector. Kept for
  history/comparison; no longer the active BFMD pipeline.

- **detect_players_bfmd.py** -- THE ACTIVE METHOD for BFMD. Same hybrid YOLOv8x + YOLOv8x-pose
  scheme as run_vimo_batch_v2.py, adapted to BFMD's per-rally structure, filtered against the
  self-detected camera (`1:Court/all_cameras_self_detected.json`). Run on all 12 BFMD matches:
  96.8-99.3% top/bottom detection coverage per match. Needs a separate venv
  (`/scratch/ciro/yolo_env` on Killarney) -- kept isolated from the VIMO venv on purpose, see below.

- **run_vimo_bfmd_selfdetect.py** -- THE ACTIVE METHOD for BFMD 3D reconstruction. Takes
  detect_players_bfmd.py's boxes + the self-detected camera, runs VIMO, and automatically applies
  the real ground/foot correction (below) to every rally -- not just a demo. Run on all 12 matches:
  median foot height ~0.06m across 172,928 sampled values (essentially grounded).

- **ground_correct_demo.py** -- superseded by the `ground_correct()` function now built directly
  into run_vimo_bfmd_selfdetect.py (same algorithm: ray-cast a real, pixel-stable 2D ankle keypoint
  onto the floor plane Z=0, rigid-shift the body, MAD-reject outliers, interpolate, floor+height
  clamp). Kept for reference/history -- it's what the algorithm was first prototyped and validated on
  (one demo rally) before being generalized to the full pipeline.

**Environment note**: `detect_players_bfmd.py` (YOLOv8/ultralytics/cv2) runs in an isolated venv
(`/scratch/ciro/yolo_env`) deliberately kept separate from the VIMO venv (`tram/.venv`) -- installing
opencv-python-headless once accidentally upgraded numpy in the shared VIMO venv from the pinned
1.23.2 to 2.4.6, which could have silently broken the validated VIMO/SMPL pipeline. Caught and
reverted before it touched anything real; the isolated venv exists specifically so this can't happen
again. ffmpeg gotchas hit and fixed while building this: the cluster's `module load ffmpeg` build
cannot decode AV1 (used by VICTOR China Open + one YONEX video) -- use the system Gentoo binary at
`/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/ffmpeg` instead; and any ffmpeg encoder
fails on the cluster's login node specifically (thread/process ulimit, not a real bug) -- add
`-threads 1`, only needed outside an actual SBATCH job.

**Status: DONE (2026-09-09)** -- the "did we actually test our method" gap is closed. All 12 BFMD
matches now have real self-detected boxes + self-detected-camera VIMO reconstruction + real ground
correction. See `3:Action/` for what this changed about the trained model's accuracy.
