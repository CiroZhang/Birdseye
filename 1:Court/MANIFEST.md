# 1:Court -- file guide

**Housekeeping note (2026-09-09)**: this folder had accumulated ~90MB of dead exploratory output
from before the final method was settled -- `monotrack_predictions/`, `monotrack_predictions_FINAL/`,
`monotrack_predictions_multiframe/`, `monotrack_predictions_net_corrected/` (iterative attempts),
`vggt_predictions/` + `analysis_images/` (78MB of net-detection check images from the abandoned
VGGT-based net-detection approach -- VGGT net detection was found too unreliable, see readme.md),
`FINAL_RESULTS/`, and 3 loose comparison CSVs. None of it was read by any script actually in use
(verified: `calibrate_from_own_detections.py`, the script that produced
`all_cameras_self_detected.json`, reads from `monotrack_bfmd_detections/` instead, which hadn't even
been synced here until now). All deleted; `monotrack_bfmd_detections/` synced in to replace it.

## The actual pipeline (see readme.md section 1 for the full method writeup)
- **court_detection.py** -- `court_detection()` (fits camera intrinsics/pose from 2D<->3D court
  point correspondences) and `fit_self_calibrated()` (the real method: RANSAC + joint focal/pose
  solve on 22 non-pole flat court points, net position constructed/reprojected from known court
  geometry afterward, never fit to detected pole points directly). Imported by both scripts below.
- **monotrack_bfmd_detections/** -- raw MonoTrack court-detection binary output (`.out` files, one
  per match, 24 points each in a fixed order) for all 12 BFMD matches -- the actual input to our
  self-calibration, never BFMD's own annotations.
- **calibrate_from_own_detections.py** -- THE ACTIVE METHOD. Reads `monotrack_bfmd_detections/`,
  calls `fit_self_calibrated()`, and separately measures reprojection error against BFMD's own
  annotated court corners (used only as an honest external check, never as calibration input).
  Produced **all_cameras_self_detected.json** (the camera used by every downstream section):
  22/22 inliers, det(R)=1.0000, 1.5-3.2px mean / 3.1-5.7px max reprojection error, 0.9875 mean IoU
  vs BFMD's own corners, 2.45px pooled MPE.
- **all_cameras_self_detected.json** -- the result: per-match camera (K, R, t), used by 2:Player,
  3:Action, and 4:Shuttle.

## Superseded (kept for comparison, not the active method)
- **calibrate_match.py** -- the original shortcut: takes BFMD's own annotated court+net corners
  directly as calibration correspondence points, instead of running our own detector. Kept only
  because its output IS the ground truth calibrate_from_own_detections.py checks itself against --
  not because its own calibration method is still used.

## Fixed (2026-09-19) -- hardcoded paths
Both `calibrate_match.py` and `calibrate_from_own_detections.py` used to have a dead local session
scratchpad path hardcoded (`BASE = "/private/tmp/claude-501/..."`) -- that path no longer exists
anywhere, so neither script ran at all. Both now resolve `BASE` relative to this file
(`../../Dataset/BFMD/BFMD_data`, overridable via the `BFMD_DATA_ROOT` env var) and
`calibrate_from_own_detections.py`'s `DET_DIR` resolves to the real local
`monotrack_bfmd_detections/` sitting in this same folder. Verified: no `/private/tmp/claude` or
other dead-session paths remain anywhere in `Project/`.

## Known issue -- duplicated, drifted module
`court_detection.py` also exists as a second, independently-edited copy in `5.Evalutation/`
(different file size, so it has drifted, not just been copied) -- there is one canonical
`fit_self_calibrated()` here and a second implementation there that may no longer agree exactly.
Not resolved as part of this pass; worth consolidating into one shared module before relying on
both giving identical results.
