"""Real ground/foot correction for the hit-detection demo rally, following
the same method already validated in this project's earlier player-3D-
reconstruction work (Project/METHODS.md, "Foot/ground correction v1 -> v2"
/ run_vimo_batch_v2.py): ray-cast a real 2D ankle keypoint (BFMD's own pose
annotation, not VIMO's own depth guess) through the calibrated camera onto
the known floor plane (Z=0) whenever that ankle is pixel-stable ("planted"),
then rigid-shift the WHOLE reconstructed body so VIMO's matching ankle joint
lands on that anchor -- MAD-rejecting outlier anchors, interpolating the
shift across frames with no stable anchor, then a hard floor clamp + a
symmetric max-height clamp as a final safety net.

This replaces the median-height-subtraction hack used earlier in this
session, which just re-centered the existing (noisy) VIMO heights instead of
anchoring them to anything real.
"""
import json

import numpy as np

BASE = "BFMD_data"
MATCH = "KAPAL-API-Indonesia-Open-2025-Shi-Yu-Qi-CHN-1-vs.-Anders-Antonsen-DEN-3-SF"
RALLY_TAG = "g1_r004"
POSE_KEY = "rally_004_12821_13237"
START, END = 12821, 13237
STABLE_PIX_THRESH = 6.0   # px/frame, same threshold as run_vimo_batch_v2.py
MIN_ANKLE_CONF = 0.3
MAD_REJECT_K = 5.0
MAX_FOOT_HEIGHT = 1.1     # m, symmetric safety clamp for residual bad frames

cam = json.load(open(f"{BASE}/prepped/{MATCH}/camera.json"))
K = np.array(cam["K"]); R = np.array(cam["R"]); t = np.array(cam["t"])


def floor_xy(u, v):
    """Ray-cast pixel (u,v) through the calibrated camera onto the real
    floor plane Z=0 -- exact geometric answer, independent of VIMO's own
    depth estimate, whenever the corresponding foot is actually on the floor."""
    Kinv = np.linalg.inv(K)
    ray_cam = Kinv @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam
    cam_center = -R.T @ t
    if abs(ray_world[2]) < 1e-9:
        return None
    s = -cam_center[2] / ray_world[2]
    if s <= 0:
        return None
    point = cam_center + s * ray_world
    return np.array([point[0], point[1], 0.0])


joints_src = json.load(open(f"{BASE}/{MATCH}_joints.json"))
rally = joints_src["rallies"][RALLY_TAG]
assert rally["start"] == START and rally["end"] == END
n_frames = END - START + 1

pose_data = json.load(open(f"{BASE}/annotations/pose/{MATCH}.json"))
pose_frames = pose_data["videos"][POSE_KEY]["frames"]
W, H = pose_data["videos"][POSE_KEY]["width"], pose_data["videos"][POSE_KEY]["height"]

bbox_by_frame = json.load(open(f"{BASE}/prepped/{MATCH}/bbox_by_frame.json"))


def bbox_center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def correct_player(side, joint_idx):
    """side: 'top'/'bottom'. joint_idx: SMPL ankle joint index (7=L,8=R) --
    used only as a fallback anchor target; the real anchor picks whichever
    ankle (L=15/R=16 in COCO) has higher pose confidence per frame."""
    joints = np.array(joints_src["rallies"][RALLY_TAG]["joints"][side])  # (n_frames, 24, 3)

    # 1. per-frame 2D ankle pixel (best-confidence of L/R), matched to this
    # side via bbox-center proximity to BFMD's own top/bottom box for that frame
    ankle_px = np.full((n_frames, 2), np.nan)
    for local_f in range(n_frames):
        global_f = START + local_f
        fb = bbox_by_frame.get(str(global_f), {})
        if side not in fb:
            continue
        target_box = np.array(fb[side])
        target_c = bbox_center(target_box)

        frame_poses = pose_frames.get(str(global_f))
        if frame_poses is None:
            continue
        best_j, best_d = None, np.inf
        for j, p in enumerate(frame_poses):
            d = np.linalg.norm(bbox_center(np.array(p["bbox"])) - target_c)
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None or best_d > 60.0:  # not a plausible match
            continue
        kpts = frame_poses[best_j]["keypoints"]
        l_ankle, r_ankle = kpts[15], kpts[16]
        chosen = l_ankle if l_ankle[2] >= r_ankle[2] else r_ankle
        if chosen[2] < MIN_ANKLE_CONF:
            continue
        ankle_px[local_f] = [chosen[0] * W, chosen[1] * H]

    # 2. stability: ankle barely moved from the previous valid frame
    stable = np.zeros(n_frames, dtype=bool)
    last_valid_px, last_valid_f = None, None
    for f in range(n_frames):
        if np.isnan(ankle_px[f]).any():
            continue
        if last_valid_px is not None and (f - last_valid_f) == 1:
            if np.linalg.norm(ankle_px[f] - last_valid_px) < STABLE_PIX_THRESH:
                stable[f] = True
        last_valid_px, last_valid_f = ankle_px[f], f

    # 3. candidate rigid shift at each stable, VIMO-reconstructed frame
    shift_f, shift_vec = [], []
    for f in np.where(stable)[0]:
        if np.isnan(joints[f, 7]).any():  # VIMO didn't reconstruct this frame
            continue
        anchor = floor_xy(*ankle_px[f])
        if anchor is None:
            continue
        vimo_ankle = joints[f, 7] if joints[f, 7, 2] < joints[f, 8, 2] else joints[f, 8]
        # (use whichever SMPL ankle sits lower this frame as the VIMO-side
        # planted foot -- L/R identity of the 2D keypoint isn't guaranteed
        # to match SMPL's L/R labeling frame-to-frame, only that SOME foot
        # is planted)
        shift_f.append(f)
        shift_vec.append(anchor - vimo_ankle)
    shift_f = np.array(shift_f)
    shift_vec = np.array(shift_vec) if shift_vec else np.zeros((0, 3))

    # 4. MAD-based outlier rejection on the candidate shifts
    if len(shift_vec) >= 4:
        med = np.median(shift_vec, axis=0)
        mad = np.median(np.abs(shift_vec - med), axis=0) + 1e-6
        keep = np.all(np.abs(shift_vec - med) < MAD_REJECT_K * mad, axis=1)
        shift_f, shift_vec = shift_f[keep], shift_vec[keep]

    # 5. interpolate the shift across every frame (hold nearest at the edges)
    if len(shift_f) == 0:
        print(f"  {side}: NO stable anchors found, cannot ground-correct")
        full_shift = np.zeros((n_frames, 3))
    else:
        full_shift = np.stack([
            np.interp(np.arange(n_frames), shift_f, shift_vec[:, c])
            for c in range(3)
        ], axis=1)
        print(f"  {side}: {len(shift_f)}/{n_frames} stable anchor frames "
              f"(median shift {np.median(shift_vec, axis=0).round(3).tolist()})")

    corrected = joints + full_shift[:, None, :]

    # 6. hard floor clamp + symmetric max-height clamp, per frame
    for f in range(n_frames):
        foot_z = [corrected[f, idx, 2] for idx in (7, 8, 10, 11) if corrected[f, idx, 2] == corrected[f, idx, 2]]
        if not foot_z:
            continue
        min_z = min(foot_z)
        if min_z < 0:
            corrected[f, :, 2] -= min_z          # lift out of the floor
        elif min_z > MAX_FOOT_HEIGHT:
            corrected[f, :, 2] -= (min_z - MAX_FOOT_HEIGHT)  # implausible height, pull back down

    return corrected


print(f"grounding {RALLY_TAG} ({MATCH})...")
top_corrected = correct_player("top", 7)
bottom_corrected = correct_player("bottom", 8)

for name, arr in [("top", top_corrected), ("bottom", bottom_corrected)]:
    foot = np.concatenate([arr[:, 7, 2:3], arr[:, 8, 2:3], arr[:, 10, 2:3], arr[:, 11, 2:3]], axis=1)
    foot = foot[~np.isnan(foot)]
    print(f"{name}: foot z min={foot.min():.3f} median={np.median(foot):.3f} "
          f"p95={np.percentile(foot,95):.3f} max={foot.max():.3f}")

demo = json.load(open("viz/demo_data.json"))
demo["top"] = top_corrected.tolist()
demo["bottom"] = bottom_corrected.tolist()
json.dump(demo, open("viz/demo_data.json", "w"))
print("saved viz/demo_data.json")
