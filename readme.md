# Birdseye
### Self-Detected End-to-End 3D Reconstruction and Analysis of Badminton Rallies

---

<p align="center">
  <a href="https://cirozhang.github.io/Birdseye/">
    <img src="https://img.shields.io/badge/▶_LIVE_DEMO-View_the_reconstruction_in_action-2ea44f?style=for-the-badge" alt="Live Demo">
  </a>
</p>

<p align="center"><b>cirozhang.github.io/Birdseye</b></p>

---

## 1. Court Detection and Construction

MonoTrack's own court-line detector (Liu & Wang, 2022 [1]) finds 22 court-plane points + 2 net-pole
points per video and estimates a camera from them. During testing we found its net-pole detection
to be unreliable, so instead we use a VGGT [2] focal-length estimate as an additive seed for a
RANSAC joint focal+pose solve over MonoTrack's 22 flat (non-pole) court points only and never its net
detections. To evaluate the result, we then re-project the net using the solved camera and the
known real net height, instead of trusting MonoTrack's own net-pole pixels. The following table
shows the improvement this gives specifically on net position, on BFMD [8]:

| Method | Mean IoU | MPE |
|---|---|---|
| MonoTrack (net detection) | 0.9518 | 4.6px |
| MonoTrack + our net projection | **0.9622** | **2.9px** |

## 2. Player Detection and 3D Reconstruction

Birdseye uses a hybrid YOLOv8x + YOLOv8x-pose [9] scheme to locate both players, then, using the
calibrated camera from Section 1, reconstructs 3D SMPL pose per frame with VIMO (from TRAM — Wang
et al., ECCV 2024 [4]). To avoid picking up referees or spectators, any detected player outside the
court boundary is rejected. A ray-cast ground correction then removes the vertical drift inherent
to monocular depth estimation, anchoring each player's reconstructed foot to the known floor plane
using a stable 2D ankle keypoint.

On the badminton doubles dataset (Ding et al., 2023 [7]), which provides synchronized top-view and
back-view drone footage, Birdseye reconstructs 3D player position from the back view alone and
compares it against the top view's real annotated ground truth. Pooled across 20 rallies, median localization error is **0.39m** (mean 0.45m). Given that the average arm span of an adult male is 
roughly 1.8m, a meaningful share of this error is plausibly body-scale noise such as a raised arm 
alone can shift a player's "position" by a large fraction of that. Thus the following table shows the 
percentage of frames within several tolerance thresholds:

| Tolerance | % of frames within |
|---|---|
| 0.5m | 66.6% |
| 0.8m | 88.6% |
| 1.0m | 93.2% |

## 3. Hit Detection and Shot Classification

To further demonstrate the effectiveness of Birdseye's representation, we use a transformer encoder
(6 heads, 4 layers, 192-dim) over 3D skeleton motion — root-relative joint position, velocity,
acceleration, for both players, with the far player canonically rotated 180° — to predict per-frame
hit probability and shot type (12 classes), trained on BFMD's real hit-timing labels (the only
manual annotation used anywhere in this pipeline).

**Train/test split:** match-level holdout, not rally-level (to prevent same-match/camera leakage) —
1,055 rallies across all 12 BFMD matches, with 217 rallies from 2 held-out matches used purely for
validation.

We train two versions: a single-player variant that mimics the case where only one player is
visible, and a two-player variant specialized for singles play. *[results table pending]*

| Method | Precision (%) | Recall (%) | F1 (%) | Stroke accuracy (%) |
|---|---:|---:|---:|---:|
| Chien–Yu† | 69.2 | 97.9 | 81.1 | — |
| TrackNet† | 58.8 | 93.6 | 72.3 | 38.8 |
| Trajectory + action† | 84.3 | 88.2 | 86.2 | 54.1 |
| Ours (full-court) | **92.4** | **93.0** | **92.7** | **78.1** |
| Ours (single-player) | 91.5 | 91.0 | 91.3 | 75.9 |

† Results reported by the respective original papers; they were not re-evaluated under our protocol.

## Datasets

- **BFMD** — Ning Ding et al., *BFMD: A Full-Match Badminton Dense Dataset for Dense Shot
  Captioning*, arXiv:2603.25533 (2026) [8]. 19 total matches (singles + doubles), 1,687 rallies; we
  use the 12 singles matches with usable hit-timing ground truth.
- **Drone doubles dataset** — Ning Ding et al., *Estimation of control area in badminton doubles
  with pose information from top and back view drone videos*, Multimedia Tools and Applications
  (2023) [7]. 39 games, 1,347 rallies total; we validate against 4 rallies (see Section 2).


## References

1. P. Liu & J.-H. Wang. "MonoTrack: Shuttle Trajectory Reconstruction From Monocular Badminton
   Video." *CVPR Workshops (CVPRW)*, 2022.
   [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.html)
2. J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny. "VGGT: Visual Geometry
   Grounded Transformer." *CVPR* (Best Paper Award), 2025. [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)
3. A. N. Raj & Prethija G. "CourtKeyNet: A novel octave-based architecture for precision badminton
   court detection with geometric constraints." *Machine Learning with Applications*, 2026.
   DOI: [10.1016/j.mlwa.2026.100884](https://www.sciencedirect.com/science/article/pii/S2666827026000496)
4. Y. Wang, Z. Wang, L. Liu, K. Daniilidis. "TRAM: Global Trajectory and Motion of 3D Humans from
   in-the-Wild Videos." *ECCV*, 2024. [arXiv:2403.17346](https://arxiv.org/abs/2403.17346)
5. Y.-H. Chien & F. Yu. "Automated Hit-frame Detection for Badminton Match Analysis." arXiv:2307.16000, 2023.
6. Y.-H. Hsu, C.-C. Yu, H.-Y. Cheng. "Enhancing Badminton Game Analysis: An Approach to Shot
   Refinement via a Fusion of Shuttlecock Tracking and Hit Detection from Monocular Camera."
   *Sensors*, 24(13), 4372, 2024. [mdpi.com](https://www.mdpi.com/1424-8220/24/13/4372)
7. N. Ding, K. Takeda, W. Jin, Y. Bei, K. Fujii. "Estimation of control area in badminton doubles
   with pose information from top and back view drone videos." *Multimedia Tools and Applications*, 2023.
   [arXiv:2305.04247](https://arxiv.org/abs/2305.04247)
8. N. Ding et al. "BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning."
   arXiv:2603.25533, 2026. [github.com/Ning-D/BFMD](https://github.com/Ning-D/BFMD)
9. YOLOv8 — G. Jocher, A. Chaurasia, J. Qiu. *Ultralytics YOLOv8*, 2023.

---

