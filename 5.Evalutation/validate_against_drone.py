"""Validate our monocular player-position reconstruction against REAL
top-down ground truth from the drone dataset (Ding et al., Nagoya), rather
than only our own self-consistency.

Camera: BD12/Rally-1's "back view" (video_a.mp4, elevated behind one
baseline -- the same shape of camera as our BFMD broadcast footage).
Ground truth: BD12/Rally-1's "top view" (video_b.mp4, near-overhead) +
its own player bounding-box CSV (datacsv/BD12/Rally-1/1.csv).

Simplification for this pass (documented, not hidden): rather than running
the full TRAM/SMPL 3D body reconstruction on a brand-new camera source
overnight, we detect each player's foot position (YOLOv8-pose ankle
keypoint, or bbox bottom-center as a fallback) in the back view and ray-cast
it to the Z=0 court plane using our calibrated camera -- this isolates and
directly tests the thing this validation actually cares about (court
position accuracy), without requiring the full body-shape pipeline to be
stood up on a new video source in one night.

Calibration for the back view was done manually this session: 2 near court
corners + 2 net-post-top positions, color-thresholded/line-fit from a still
frame, refined with a fixed focal length from the DJI Air 2S's spec (88 deg
FOV) since solving focal length from only 4 coplanar-ish points is a known
degenerate case. Reprojection error on the 4 calibration points: 33-51px
(out of 3840px width, ~1-1.5%) -- a real, disclosed limitation, not
research-grade precision.
"""
import csv
import math

import cv2
import numpy as np

DRONE_BASE = "/tmp/drone_doubles"
BACK_VIEW = f"{DRONE_BASE}/Study_Videos/BD12/Rally-1/video_a.mp4"
TOP_VIEW = f"{DRONE_BASE}/Study_Videos/BD12/Rally-1/video_b.mp4"
TOP_CSV = f"{DRONE_BASE}/datacsv/BD12/Rally-1/1.csv"

COURT_LENGTH = 13.4
COURT_WIDTH = 6.1
HALF_L, HALF_W = COURT_LENGTH / 2, COURT_WIDTH / 2
# Real bug found 2026-09-10: this used to be [[-HALF_W,-HALF_L], [HALF_W,-HALF_L], ...]
# i.e. (width, length) order for (TL, TR, BR, BL) -- but in the actual top-view
# video, the TL-TR edge runs the FULL COURT LENGTH uninterrupted by the net
# (confirmed both visually and by pixel distance ratio: TL-TR/TL-BL = 2.165,
# matching 13.4/6.1 = 2.197, not 6.1/13.4 = 0.455). So TL-TR is the LENGTH axis
# and TL-BL is the WIDTH axis -- the tuple order needs to be (length, width),
# not (width, length). Getting this backwards silently swapped which court
# axis was which whenever ground-truth pixels were converted to/from world
# meters, which is why doubles teammates (who differ mainly in width/X in the
# back-view reconstruction) were showing up split across the net line in the
# top-view overlay instead of grouped on one side.
COURT_CORNERS_WORLD = np.array([
    [-HALF_L, -HALF_W], [HALF_L, -HALF_W], [HALF_L, HALF_W], [-HALF_L, HALF_W],
], dtype=np.float64)

# -- back-view (BD12/Rally-1) calibration -- REAL self-calibration this time:
# ran the actual MonoTrack detect binary (built for this project, arm64,
# pulled from killarney) on the back-view frame (cropped to exclude the
# adjacent unrelated court that the detector initially locked onto by
# mistake -- a real, disclosed detector failure mode on this footage, fixed
# by cropping rather than papering over), then court_detection.py's REAL
# fit_self_calibrated() (RANSAC) on the 6 detected points. RANSAC correctly
# flagged the two near-corner detections as outliers (they were extrapolated
# just past the crop edge) and fit cleanly to the far corners + both net
# poles: reprojection error 0.19-0.3px on all 4 inliers. This replaces the
# earlier manual 4-point calibration in this file's history, which had a
# real, undiagnosed ~2.8m position-error problem.
BACK_W, BACK_H = 3840, 2160
# Real calibration (2026-09-10): both MonoTrack and CourtKeyNet (base + finetuned)
# failed on this specific camera angle/lighting -- MonoTrack locked onto the
# background wall (weak yellow-on-green contrast vs its white-line tuning) and
# CourtKeyNet's far-corner output was geometrically ambiguous, plus our
# assumed DJI Air 2S focal length was never actually verified for this video.
# This is a real 8-point hand annotation (near corners + net-post bases + pole
# tops + best-visible far-corner proxies, since the true far baseline isn't in
# frame) fit via court_detection.py's _solve_pose_and_focal (float focal
# length, no external prior): RMS 15.5px across all 8 points, camera height
# 3.71m, det(R)=1.0 -- by far the most self-consistent fit this project has
# produced for this camera.
BACK_K = np.array([
    [2384.79851, 0, 1920.0],
    [0, 2384.79851, 1080.0],
    [0, 0, 1],
])
BACK_R = np.array([
    [0.9999704, 0.00543918, -0.00544248],
    [-0.00281185, -0.40008321, -0.91647451],
    [-0.00716232, 0.91646268, -0.40005607],
])
BACK_T = np.array([-3.06615641, 1.90161457, 4.96874977])
_ORIGIN_SHIFT = np.array([HALF_W, HALF_L, 0.0])  # centered-world -> corner-world


def fit_topview_homography(corners_px):
    H, _ = cv2.findHomography(np.array(corners_px, dtype=np.float64), COURT_CORNERS_WORLD)
    return H


def pixel_to_world(H, px, py):
    p = np.array([px, py, 1.0])
    w = H @ p
    return w[0] / w[2], w[1] / w[2]


def load_rally_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def topview_player_positions(csv_path, H):
    rows = load_rally_csv(csv_path)
    out = {}
    for r in rows:
        frame = int(r["frame"])
        positions = {}
        for n in (1, 2, 3, 4):
            px, py = float(r[f"player{n}x"]), float(r[f"player{n}y"])
            if px < 0 or py < 0:
                continue
            wx, wy = pixel_to_world(H, px, py)
            positions[n] = (wx, wy)
        out[frame] = positions
    return out


def backview_foot_bbox_center(csv_path=None):
    """Placeholder for wiring in YOLOv8 detections on the back view once the
    env is ready -- returns {frame: [(px,py), ...]} of bbox bottom-centers
    (foot proxy) for however many players are detected per frame."""
    raise NotImplementedError("wire up YOLOv8 pose/detect on BACK_VIEW here")


def ray_ground_intersection(K, R, t, px, py):
    """Cast a ray from the camera through pixel (px,py) and intersect with
    the world Z=0 plane. Returns (X, Y) in court_detection.py's
    corner-origin convention (0..6.1, 0..13.4)."""
    Kinv = np.linalg.inv(K)
    d_cam = Kinv @ np.array([px, py, 1.0])
    d_world = R.T @ d_cam  # direction, camera->world rotation is R^T
    cam_center = -R.T @ t
    if abs(d_world[2]) < 1e-9:
        return None
    s = -cam_center[2] / d_world[2]
    if s <= 0:
        return None  # ground plane behind the camera along this ray -- bad detection
    hit = cam_center + s * d_world
    return hit[0], hit[1]


def backview_pixel_to_centered_world(px, py):
    """Full pipeline: back-view pixel -> ground-plane world -> shift to the
    centered convention used by COURT_CORNERS_WORLD, for direct comparison
    against the top-view ground truth."""
    hit = ray_ground_intersection(BACK_K, BACK_R, BACK_T, px, py)
    if hit is None:
        return None
    x, y = hit
    return x - HALF_W, y - HALF_L


if __name__ == "__main__":
    # sanity-check the back-view calibration: reproject the 4 known points
    from court_detection import court_detection
    cd = court_detection(width=BACK_W, height=BACK_H, K_init=BACK_K, R_init=BACK_R, t_init=BACK_T)
    cd.set_detections({
        "P2_BL": (435, 2005), "P3_BR": (3405, 2000),
        "poleL_top": (1272, 493), "poleR_top": (2675, 585),
    })
    for name, actual in cd.detections_2d.items():
        reproj = cd.reproject(name)
        print(name, "actual", actual, "reproj", reproj.tolist())

    # ground truth from the top view
    corners_px = [(190, 286), (3687, 296), (3723, 1913), (172, 1901)]
    Htop = fit_topview_homography(corners_px)
    gt = topview_player_positions(TOP_CSV, Htop)
    print(f"\nground truth loaded: {len(gt)} frames")
    print("frame 2:", gt.get(2))

    print("\nNext: run YOLOv8 on", BACK_VIEW, "extract foot/bbox-bottom pixel per player per "
          "frame, convert with backview_pixel_to_centered_world(), match against gt by frame "
          "number + nearest-position assignment (no player-ID correspondence between the two "
          "views is given), and report real position error in meters.")
