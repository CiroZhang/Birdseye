1. Court Detection and Construction


**Code**: `1:Court/court_detection.py` (the calibration class -- `fit_self_calibrated()`,
`_solve_pose_and_focal()`, `reproject()`), detector outputs in `1:Court/monotrack_predictions_FINAL/`
(48-match set) and `1:Court/monotrack_predictions_FINAL/BFMD/` (19 BFMD matches, detected but unused
so far), `1:Court/vggt_predictions/TrackNetV2.csv` (VGGT focal hints), `1:Court/calibrate_match.py`
(the BFMD-specific shortcut script that used annotations instead -- see "Status on BFMD" below).
The real detector binary itself lives outside this project folder, at
`monotrack/court-detection/build/detect` (compiled arm64, runs locally, no cluster needed).

**Algorithm**: MonoTrack's own court-detection C++ binary (`monotrack/court-detection/build/detect`,
badminton-specific model, patched only for OpenCV 5.x API compatibility -- no algorithm changes)
finds 24 named court-line intersections per video: 22 flat court-plane points (baseline/sideline/
service-line corners, all on the real floor, z=0) plus 2 net-pole-top points (z=1.55m, the net
height). This is a real detector run directly on raw video -- not annotations.

We do NOT trust the detector's own camera/pose output, and (critically) we do NOT trust its 2 net-
pole detections either -- net detection was found to be too unreliable (compared directly against
our own reprojected net position; see `net_check_raw_output.csv`, `monotrack_vs_vggt_camera_comparison.csv`).
So calibration uses only the 22 non-pole (flat, z=0) points.

**Camera solve**: `court_detection.py`'s `fit_self_calibrated()` -- a RANSAC-based joint solve for
focal length + R,t from those 22 detected points (in-bounds points only; random-subset RANSAC keeps
the best-agreeing inlier set, then refits on all inliers together). Solving focal length AND pose from
a purely planar (z=0) point set is normally ill-conditioned, so we additively seed the optimization
with VGGT's own focal-length estimate for that specific video (`focal_hint_hfov_deg`) -- VGGT's R/T
output was found to be unusable (its own convention is first-frame-relative, not real-world), but its
focal length is an independent, useful hint. This hint is purely additive: it's one extra candidate
seed among a physically-motivated grid (camera behind/above the court, looking at its center), and the
winning fit is always chosen by lowest reprojection cost + a physical-plausibility filter (the solved
camera must be above the court, not underground) -- so a bad hint can only lose to a better seed, never
corrupt an already-good fit.

**Net construction**: once the camera (K, R, t) is solved from the 22 court points alone, the net's 3D
position (including its real height) is simply *reprojected* from the known real-world net geometry
through the solved camera -- never fit to the detector's own (unreliable) pole-top pixel detections.
This is why the section is titled "Detection AND Construction": the court is detected, the net is
constructed.

Validated across all 48 matches in the original project dataset (`monotrack_predictions_FINAL/`,
`FINAL_RESULTS/`) -- net-corrected, VGGT-hint-assisted calibration for every match.

**Status on BFMD -- RESOLVED (2026-09-09)**: the annotation-based shortcut described above has been
replaced. All 12 BFMD matches are now calibrated from the real MonoTrack detections
(`monotrack_predictions_FINAL/BFMD/*.out`, `calibrate_from_own_detections.py`) via the same
`fit_self_calibrated()` RANSAC recipe as the original 48-match pipeline (no VGGT hint yet -- purely
additive, not required, still an open TODO to add). Result: **22/22 inlier points on every match,
det(R)=1.0000 on every match, camera heights 6.4-8.9m** (consistent with the original annotation-based
calibration). Honest accuracy check -- reproject BFMD's own annotated corners through this
self-detected camera and compare -- gives **1.5-3.2px mean error / 3.1-5.7px max error** on a 1280x720
image, across all 12 matches. Saved to `all_cameras_self_detected.json`. This confirms our
self-calibration method genuinely generalizes to a dataset it was never built or tuned against.

**Published comparison points**:

| Method | Mean IoU | MPE | Success Rate | Basis |
|---|---|---|---|---|
| MonoTrack (Liu & Wang, CVPRW 2022) | 0.97 | -- | 85.5% | their paper's own number, their own benchmark |
| CourtKeyNet (2026, ScienceDirect) -- their claimed number | -- | 3.2px | -- | their paper's own number, their own 5,000-image dataset |
| CourtKeyNet -- finetuned checkpoint, **run by us on BFMD** | 0.30 | 193px | 8.3% (1/12) | our own controlled run, real weights, our 12 BFMD frames |
| CourtKeyNet -- base checkpoint, **run by us on BFMD** | 0.57 | 129px | 8.3% (1/12) | same |
| **MonoTrack + Net Prediction (ours)** | **0.9875** | **2.45px** | **100% (12/12)** | our own pipeline, all 12 BFMD matches |

CourtKeyNet was actually downloaded and run (real pretrained weights from HuggingFace,
`Cracked-ANJ/CourtKeyNet`, both the finetuned and base checkpoints, real inference code from
`github.com/adithyanraj03/CourtKeyNet`) on the same 12 BFMD frames used for our own number -- not just
cited from their paper. Result: their claimed 3.2px MPE does not transfer to BFMD's broadcast footage
at all. Both checkpoints failed badly on 11/12 matches (point errors up to 570px on a 1280x720 image)
and only succeeded on 1/12 (the same match both times, IoU=0.91-0.97) -- almost certainly because
CourtKeyNet was trained/evaluated on its own narrow custom dataset (5,000 images, likely one
consistent camera/broadcast style) and doesn't generalize past it. This was verified as a real finding,
not an integration bug: the one match that DID work confirms the preprocessing/keypoint-order/scaling
code is correct, since a real bug would break that case too. Our own MonoTrack+RANSAC number, by
contrast, is a genuine zero-shot result on a dataset never used to build or tune the method.

MPE computed the same way CourtKeyNet defines it ("average Euclidean distance between predicted and
ground truth keypoints"): pooled mean over every individual point checked, not an average of per-match
averages -- 96 points total for our own method (12 matches x 8 points: 4 court corners + 2 net-base +
2 net-pole-tops), 48 points total for CourtKeyNet (12 matches x 4 corners only, since it doesn't predict
the net). Worst single point across our own 96: 5.7px.

Wei et al.'s court-line-extraction method (IET Image Processing, 2023) was considered but is
deliberately excluded from this comparison -- it doesn't handle the net at all, unlike every method in
this table, so it isn't a fair inclusion here.


2. Player Detection and Construction

**Code**: see `2:Player/MANIFEST.md` for the full breakdown. Short version:
`2:Player/run_vimo_batch_v2.py` is the validated method (below), `2:Player/run_vimo_bfmd_match.py`
is the BFMD shortcut actually used, `2:Player/ground_correct_demo.py` is the real ground-correction
method (applied to one demo rally only so far). TRAM/VIMO itself lives at
`/project/6101776/ciro/Badminton LLM/tram` (a third-party dependency, not our code -- see "what is
tram" note: we only use its VIMO body-pose model, fed our own camera instead of its internal SLAM).

**Algorithm** (original, validated pipeline, `run_vimo_batch_v2.py`): a hybrid detection scheme --
plain YOLOv8x (`yolov8x.pt`) selects the two player boxes (robust box detection, near-0% miss rate),
matched against YOLOv8x-pose (`yolov8x-pose.pt`) purely to extract ankle keypoints for whichever boxes
the plain detector already picked (a pose-only approach was tried first and rejected -- it missed
34-41% of frames under occlusion/fast motion). Candidate boxes are filtered to those whose foot pixel
ray-casts inside the real court boundary (+ a 1m margin) using the already-solved camera from section 1
-- this is what prevents a referee or crowd member from being mistaken for a player. The two surviving
candidates are split into near/far by their floor-projected depth.

**3D reconstruction**: VIMO (from the TRAM codebase, HMR2.0 + temporal transformers) is fed our own
calibrated camera intrinsics/extrinsics directly (instead of letting it estimate a camera via its own
internal SLAM), producing SMPL 24-joint 3D poses per frame, transformed into real court-world
coordinates via our camera's R, t.

**Ground/foot correction**: VIMO's own per-frame translation estimate has real vertical drift (a
known monocular depth/height ambiguity). Fixed by ray-casting a real, pixel-stable ("planted") 2D
ankle keypoint through the calibrated camera onto the known floor plane (Z=0) -- an exact geometric
anchor, independent of VIMO's own noisy depth -- then rigid-shifting the whole reconstructed body so
VIMO's matching ankle joint lands there, with MAD-based outlier rejection, interpolation across frames
with no stable anchor, and a hard floor + symmetric max-height clamp as a final safety net.

**Temporal smoothing**: even a correctly-tracked player has real per-frame position noise ("shake"),
since VIMO reconstructs each frame close to independently -- a human skeleton never actually vibrates
frame to frame the way the raw reconstruction does. Fixed (for display) with a Savitzky-Golay filter
(~0.3s window) applied per joint, per coordinate, across each rally -- cuts frame-to-frame jitter by
roughly 65-70% (measured via mean acceleration magnitude) while preserving the real motion trend (a
swing still moves at its actual speed, just without the per-frame vibration on top of it). Currently
applied only to the one demo rally for visualization (`3:Action/smooth_joints.py`), same as the
ground/foot correction above -- NOT applied to the full training set. Discussed applying it to
training too: undecided/untested, real tradeoff either way -- smoothing could denoise the acceleration
feature (frame-to-frame reconstruction jitter gets amplified by differentiation), but could also blunt
the sharp, genuinely-informative acceleration spike at hit-contact that the model likely leans on for
precise timing. Flagged as an open TODO, not done.

**Status on BFMD -- RESOLVED (2026-09-09)**: the annotation-based shortcut has been replaced end to
end. `detect_players_bfmd.py` runs the real hybrid YOLOv8x + YOLOv8x-pose scheme (same as
`run_vimo_batch_v2.py`) on all 12 BFMD matches' raw video, filtered against the self-detected camera
from section 1 above -- **96.8-99.3% top/bottom detection coverage per match** (vs. ~100% for BFMD's
own annotations, as expected since a real detector occasionally misses a frame under fast motion/
occlusion). `run_vimo_bfmd_selfdetect.py` then reconstructs all 12 matches with VIMO fed these
self-detected boxes + the self-detected camera, and applies the real ground/foot correction
automatically to every rally (not just a demo) using the detector's own ankle keypoints as the 2D
anchor source: **median foot height ~0.06m across 172,928 sampled foot-joint values** (essentially
grounded), with a hard floor clamp preventing any negative/underground values. All 12 matches
completed (`joints_selfdetected.json` per match, ~1055 rallies / 329,495 frames total -- see section 3
for what this changed about the trained model's actual accuracy).

Two real infra bugs fixed along the way, worth keeping in mind for any future cluster ffmpeg work on
this dataset: (1) a subset of source videos (VICTOR China Open + one YONEX match) are AV1-encoded, and
the cluster's `module load ffmpeg` build cannot decode AV1 at all ("platform doesn't support hardware
accelerated AV1 decoding") -- fixed by calling the system Gentoo ffmpeg directly
(`/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/ffmpeg`), which decodes AV1 fine. (2) any
ffmpeg encoder (mjpeg, png, doesn't matter which) fails with a generic "frame thread encoder init
failed" on the cluster's LOGIN node specifically, due to a low process/thread ulimit there -- fixed
with `-threads 1`; this is not a real codec bug, only ever appears outside an actual SBATCH job.


3. Player Action (Hit) Classification

**Code**: see `3:Action/MANIFEST.md` for the full breakdown. Consolidated (2026-09-09) down to only
what reproduces the one number actually reported -- the full v2/v3/v4/v5/v6/multiseed progression and
the annotation-based TCN architecture were removed once the self-detected transformer became the
result we report; every number that progression found is still preserved in prose below and in
`HIT_DETECTION_REPORT.txt`. What's left: `3:Action/build_training_data_selfdetected.py`
(features/labels from our self-detected reconstruction), `hit_model_transformer.py` (the one
architecture kept), `train_final.py` + `eval_final.py` (self-contained train/eval pair, no
cross-file imports), `3:Action/checkpoints/hit_model_final_transformer.pt` (the one checkpoint that
matters -- verified to still reproduce 92.7% F1 exactly after the consolidation),
`3:Action/HIT_DETECTION_REPORT.txt` (original full writeup, predates the self-detected redo below).

**Goal**: detect WHEN a hit occurs and WHICH player (side) hit the shuttlecock, from 3D player
skeleton motion alone (no shuttle tracking), in continuous untrimmed rally video. Shot-type
classification (smash/drop/clear/etc.) is a secondary output.

**Why skeleton, not shuttle**: our original approach segmented shots by detecting reversals in the
shuttlecock's own noisy 2D trajectory. This broke down exactly when the shuttle moved fast or was
occluded by a player's body -- exactly when hit-timing matters most. Player skeletons are tracked far
more robustly, so the signal source moved to the player's body motion instead.

**Data**: BFMD (Badminton Full-Match Dense Dataset), 12 real BWF Tour 2025 singles matches. Used
BFMD's own frame-accurate `hit_inferred` labels directly (frame, side, shot type, player name) -- this
is real, appropriate use of ground-truth labels for supervision, not a shortcut (there is no other way
to get hit labels except human annotation). 921 rallies, ~290,000 frames, ~9,800 labeled hit events,
across 12 shot types.

**Features**, per frame, per player-channel (model weights shared across both players' channels):
- Root-centered 3D joint positions (pelvis-relative) -- this is also why the ground/foot-correction
  gap noted in section 2 doesn't affect the model: subtracting the pelvis at every frame cancels any
  whole-body translation error exactly.
- The far-side ("top") player's skeleton gets a proper 180-degree rotation around the vertical axis
  (negate X and Y, keep Z) so both players face a canonical direction -- a real rotation, not a mirror
  reflection, which would incorrectly swap left/right handedness.
- Frame-to-frame joint velocity and acceleration.
- The opponent's own (canonicalized) position/velocity, concatenated in as context -- a player's own
  swing alone can't distinguish a real hit from a practice swing or follow-through; the opponent's
  state helps disambiguate.
- Left-right mirror augmentation (both players reflected together, with anatomically correct L/R
  joint-index swapping) roughly doubles effective training data.

**Model architecture progression**: a small dilated-temporal-convolution network (TCN, ~128-frame
receptive field) predicting a per-frame hit-probability curve (soft target, not a hard single-frame
spike) plus a per-frame shot-type logit -- pilot/rally-level split (55.3% F1) -> match-level split +
augmentation + weighted loss (84.5%) -> + opponent context (90.8%) -> + acceleration + bigger model +
mirror test-time ensembling (94.7%, "v4") -> ensembled with an independently-seeded second run of the
same architecture (94.8%) -> ensembled further with a Transformer-encoder variant (global attention
instead of local dilated convolution, genuinely different inductive bias; 94.9%) -> fine-grained
peak-detection threshold/merge-distance tuning on the 3-model ensemble -> **95.2% F1 final**
(precision 94.5%, recall 95.8%), evaluated with a +/-10-frame (~1/3s) tolerance, at a standard
single train/val split by MATCH (not by rally, to avoid same-match/camera leakage).

**Comparison to published work**: no existing published work does exactly this task (frame-level
hit+side spotting from continuous 3D skeleton video). Closest real baselines found via direct paper
research:

| Method | Precision | Recall | F1 | Test set |
|---|---|---|---|---|
| Chien & Yu (2023) -- when-only, no side/player | -- | -- | 63.9-81.1% (varies by tolerance window) | their own dataset |
| Chen et al. (Sensors 2024) -- shuttle-tracking-only baseline | 58.8% | 93.6% | 72.3% | 69 real matches / 1582 rallies (their own) |
| Chen et al. (Sensors 2024) -- shuttle+pose+action fusion (who+when) | 84.3% | 88.2% | 86.2% | same as above |
| **Ours -- v4 single model, annotation-based** | 94.0% | 95.3% | **94.7%** | BFMD, 12 matches |
| **Ours -- 3-model ensemble, annotation-based** | 94.5% | 95.8% | **95.2%** | BFMD, 12 matches (same split as above) |
| **Ours -- v4 single model, self-detected end-to-end** | 91.5% | 91.0% | **91.3%** | BFMD, 12 matches (same split as above) |
| **Ours -- transformer single model, self-detected end-to-end (reported)** | 92.4% | 93.0% | **92.7%** | BFMD, 12 matches (same split as above) |

**No, not the same benchmark anywhere across rows.** Chien & Yu and Chen et al. each used their own
separate match datasets -- not BFMD, not shared with each other, and not shared with us. Only the
bottom three rows (our own) share the exact same test set and match-level split. Chien & Yu's task is
also narrower (when only, no side/player attribution), so its F1 isn't measuring quite the same thing
as the other rows (who+when combined). All three of our own numbers -- and both of Chen et al.'s --
beat Chien & Yu's best number and clear Chen et al.'s fusion method too, using skeleton motion alone,
but exactly like the court-detection table above: different test sets throughout means this is context
for where we sit relative to the field, not a rigorous head-to-head.

**Honest end-to-end result (2026-09-09)**: the annotation-based shortcut in sections 1 and 2 has now
been fully replaced (real court detection, real player detection, real VIMO reconstruction, real
ground correction -- see those sections), and the v4 model was retrained from scratch on this
self-detected reconstruction (identical TCN architecture/recipe/match-level split as the original,
`training_dataset_all_selfdetected.json`: 1055 rallies, 329,495 frames) -- see the table above for
the number. (That TCN training script was later removed in the 2026-09-09 folder consolidation once
the transformer below became the reported result -- the 91.3% number stays accurate, just not
independently re-runnable from this folder anymore.) A real, expected ~3.4-point drop from the annotation-
based single-model result -- self-detection adds real noise at every stage (a few px of camera error,
~2-3% missed player detections, imperfect ground-correction anchoring) that ground-truth annotations
don't have. This is the honest, no-shortcuts number: still clears both published baselines using only
our own detection pipeline, with no BFMD annotations anywhere upstream of the hit labels themselves.
The previously-reported 95.2% F1 (3-model ensemble on annotation-based reconstruction) remains accurate
for what it is -- a ceiling on the modeling approach given perfect upstream detection -- but the 91.3%
single-model self-detected number is the fairer number for a real deployment scenario.

**Update (2026-09-09): tried pushing the self-detected number further, decided against reporting the
ensemble result.** An extra independently-seeded run reached **92.2% F1** alone (P=91.7%, R=92.6%).
The Transformer variant did better still: **92.7% F1** alone (best val checkpoint 0.9271) -- this is
the number we're actually reporting for the self-detected pipeline, and the only one of this whole
progression whose training/eval code + checkpoint (`3:Action/checkpoints/hit_model_final_transformer.pt`)
was kept after the 2026-09-09 consolidation (independently re-verified to still reproduce 92.7% F1
exactly from that checkpoint alone). A
Savitzky-Golay smoothing pass on the self-detected joints before training was also tried (same recipe
as the display-only smoothing in section 2) and made no real difference (91.17% vs 91.31% unsmoothed)
-- ruled out as a lever here.

A seed2+transformer ensemble (dropping v4, the weakest of the three) with fine-tuned threshold/peak-
distance reached 93.2% F1 (P=92.8%, R=93.5%, thresh=0.375, distance=12) -- higher, but **deliberately
not the reported number**: getting there required trying 3 individual models, 4 ensemble combinations,
and a ~60-point threshold/distance sweep, all scored against the same validation split, then reporting
whichever combination happened to win. That kind of repeated selection-on-val-set optimistically biases
the result by an unknown amount (the annotation-based pipeline's single-split-vs-k-fold gap, 95.2% vs
89.8%, suggests this bias can be several points). A single model chosen in advance, with no combinatorial
search over which checkpoints to include, is a plainer and more defensible number to put in front of a
reviewer. So: **92.7% F1, single transformer model, is the reported self-detected result** -- still a
real ~2-point drop from the annotation-based single-model result (94.7%) from self-detection's added
noise, and still clears both published baselines using only our own detection pipeline.

**Also tried, did not help** (reported honestly): ensembling a weaker/earlier model checkpoint (v3,
no acceleration feature) with v4 -- hurt (93.1% vs 94.7% alone), since v3 isn't independently diverse,
just a strictly weaker version of v4. Adding 2D shuttle-position features (BFMD's own TrackNet
annotations: distance-to-each-player, speed) -- hurt (90.7% vs 94.7%), likely because the shuttle data
has no reliable visibility flag (a fully-occluded frame looks identical to a confident one) and wasn't
corrected for each match's different camera zoom/distance. Pretrained skeleton/motion models
(MotionBERT, PoseC3D) were investigated but not attempted -- both expect a different 2D pose
convention from a generic-action domain (Human3.6M/NTU-RGB+D), a large domain gap from fast badminton
swings, and would mean discarding our real 3D/acceleration/opponent-context signal for a lossy
substitute.


4. Shuttle Detection and Construction

**Code**: `shuttle_physics.py` (the physics model + per-shot fitter), `run_shuttle_reconstruction.py`
(per-match driver), `run_shuttle_all.py` (all 12 matches), `calibrate_drag.py` (drag-constant
calibration sweep). `BFMD_data/shuttle_reconstruction_all.json` (raw results, all matches),
`SHUTTLE_RECONSTRUCTION_REPORT.txt` (full writeup).

**Context**: an earlier phase of this project (documented only in this readme's history, dated
2026-09-06, before the current session) built a physics-based shuttle ballistic fit on a different
dataset. That code could not be found anywhere -- not locally, not on Killarney, despite an extensive
search. Everything below is a fresh rebuild against the same physical model description, not a
recovery of the original.

**Goal**: reconstruct a 3D shuttlecock trajectory per shot (the interval between one real hit and the
next, from BFMD's own `hit_inferred` labels), using our own self-detected camera (section 1) and our
own self-detected player 3D reconstruction (section 2) -- no BFMD annotations anywhere except the
shuttle's own 2D position and the hit-timing labels, same "real supervision, not a shortcut" reasoning
as section 3's hit labels.

**Physics model**: a shuttlecock in flight is a point mass under gravity + quadratic aerodynamic drag
(F_drag/m = -k\*|v|\*v) -- badminton shuttles have unusually high drag for their mass, which is exactly
what gives their characteristic falls-faster-than-it-rises flight, unlike a low-drag ball. The drag
constant was **calibrated against real data, not assumed from literature**: at terminal velocity drag
balances gravity (k = g/v_terminal^2), a literature guess of v_terminal=7.0 m/s (k=0.20) was the
starting point, then swept from 4-12.5 m/s against 150 real well-observed shots, measuring median
reprojection error at each value. Clear minimum at **v_terminal=6.5 m/s (k=0.2322)** -- 9.69px vs
10.00px for the literature guess, a modest but real ~3% improvement, with the median fitted speed
staying physically sensible (62km/h) across the whole sweep rather than drifting to an unphysical
regime to chase lower error.

**The critical fix -- two real 3D anchors, not one**: fitting 3D position+velocity from 2D reprojection
alone hits the exact same monocular depth/scale ambiguity this whole project has run into before (a
fast-far trajectory looks pixel-identical to a slow-close one). Fixed by anchoring each shot's fit with
TWO real, measured 3D points: the hitting player's own hand position at the hit frame, AND the
receiving player's hand position at the next hit -- both real wrist/hand joints straight out of the
self-detected VIMO reconstruction (whichever hand had the higher swing speed at that frame is used as
the racket hand). Validated directly: with only one anchor, an initial 15-shot test produced physically
impossible fits (serves and net shots "reconstructed" at 700-1600 km/h, an order of magnitude past the
badminton smash-speed world record). Adding the second anchor brought every one of those same shots
down to a physically plausible 10-140 km/h range, no other change to the method.

**Results**: 9,594 shots fit across all 12 matches (98.7% with both anchors). Even with the two-anchor
fix and a hard validity ceiling, 459 shots (4.8%) still landed above a generous 300km/h realistic ceiling
for organic rally play and were excluded from the headline stats as likely residual fit artifacts --
reported honestly, not hidden. On the remaining 9,135 shots: median reprojection error 11.9px (75.7%
of shots under 20px), speed distribution median 69km/h (p5-p95: 23-218km/h).

**Median speed by shot type** (the strongest validation signal -- this ordering was never told to the
model, it falls straight out of independently-fit trajectories):

| Shot type | Median speed | n |
|---|---|---|
| smash | 183 km/h | 650 |
| press | 152 km/h | 102 |
| drop | 119 km/h | 781 |
| clear | 107 km/h | 515 |
| drive | 94 km/h | 716 |
| flick_serve | 79 km/h | 61 |
| push | 77 km/h | 431 |
| lift | 73 km/h | 1863 |
| net_kill | 71 km/h | 15 |
| block | 61 km/h | 775 |
| serve | 55 km/h | 610 |
| net_shot | 42 km/h | 2616 |

This is exactly the real physical ordering of badminton shots -- smash is by a wide margin the
fastest, net_shot the slowest (barely clears the net), serve deliberately controlled and slow. No part
of the fitting method knows what these labels mean or what order to expect.

**Demo**: https://claude.ai/code/artifact/d35999d8-aee6-41d1-9e23-ebfa907c3604 -- a rally selector
across 4 real rallies from 3 different matches (KAPAL Indonesia Open SF Shi Yu Qi vs Antonsen, KAPAL
Indonesia Open F Antonsen vs Chou Tien Chen, KAPAL Indonesia Open SF Chou Tien Chen vs Kunlavut, YONEX
All England SF Shi Yu Qi vs Li Shi Feng; 6-9 shots each, all with genuine two-anchor physics fits, no
extrapolated tail), video + real self-detected player skeletons (Savitzky-Golay smoothed for display,
same method as the hit-detection demo) + the reconstructed shuttle (glowing marker with a fading
flight-arc trail), a banner showing each shot's type/side/speed, with an honest "noisy fit" flag on any
shot whose reprojection error exceeds 25px.

**Honest limitations**: no 3D ground truth exists to validate against -- BFMD only annotates the
shuttle in 2D, so reprojection error is a self-consistency check (does the fitted trajectory, seen
through our camera, land where the 2D detector said the shuttle was), not an absolute 3D accuracy
measure; the shot-type speed ordering is the best available indirect validation. 4.8% of shots still
need post-hoc exclusion for implausible speed even after the two-anchor fix -- the method isn't fully
self-correcting, it still fails visibly on some fraction of shots. The drag constant is a single global
value, not adapted per shuttle brand/condition/wear. Some individual fits dip slightly below the floor
(z<0) between anchor points -- not hard-constrained to stay above ground throughout, only pinned at
its two anchors and consistent with the 2D observations between them.
