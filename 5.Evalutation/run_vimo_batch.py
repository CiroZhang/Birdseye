"""Generic version of run_vimo_doubles.py, parameterized by ROOT dir (passed
as sys.argv[1]) instead of hardcoded, so the same real VIMO reconstruction +
ground-correction logic runs unchanged across every rally sharing the fixed
camera. Expects {ROOT}/frames/*.jpg, {ROOT}/backview_camera.json,
{ROOT}/tracked_players.json; writes {ROOT}/joints.json.
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "/project/6101776/ciro/Badminton LLM/tram")
from lib.models import get_hmr_vimo

ROOT = sys.argv[1]
STABLE_PIX_THRESH = 6.0
MAD_REJECT_K = 5.0
MAX_FOOT_HEIGHT = 1.1

cam = json.load(open(f"{ROOT}/backview_camera.json"))
K, R, t = np.array(cam["K"]), np.array(cam["R"]), np.array(cam["t"])
tracks = json.load(open(f"{ROOT}/tracked_players.json"))

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
BONES = [(i, p) for i, p in enumerate(SMPL_PARENTS) if p >= 0]


def floor_xyz(u, v):
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


def ground_correct(joints, ankle_px):
    n = joints.shape[0]
    px = np.nanmean(ankle_px, axis=1)
    stable = np.zeros(n, dtype=bool)
    last_valid_px, last_valid_f = None, None
    for f in range(n):
        if np.isnan(px[f]).any():
            continue
        if last_valid_px is not None and (f - last_valid_f) == 1:
            if np.linalg.norm(px[f] - last_valid_px) < STABLE_PIX_THRESH:
                stable[f] = True
        last_valid_px, last_valid_f = px[f], f

    shift_f, shift_vec = [], []
    for f in np.where(stable)[0]:
        if np.isnan(joints[f, 7]).any():
            continue
        anchor = floor_xyz(*px[f])
        if anchor is None:
            continue
        vimo_ankle = joints[f, 7] if joints[f, 7, 2] < joints[f, 8, 2] else joints[f, 8]
        shift_f.append(f)
        shift_vec.append(anchor - vimo_ankle)
    shift_f = np.array(shift_f)
    shift_vec = np.array(shift_vec) if shift_vec else np.zeros((0, 3))

    if len(shift_vec) >= 4:
        med = np.median(shift_vec, axis=0)
        mad = np.median(np.abs(shift_vec - med), axis=0) + 1e-6
        keep = np.all(np.abs(shift_vec - med) < MAD_REJECT_K * mad, axis=1)
        shift_f, shift_vec = shift_f[keep], shift_vec[keep]

    if len(shift_f) == 0:
        full_shift = np.zeros((n, 3))
    else:
        full_shift = np.stack([np.interp(np.arange(n), shift_f, shift_vec[:, c]) for c in range(3)], axis=1)

    corrected = joints + full_shift[:, None, :]
    for f in range(n):
        foot_z = [corrected[f, idx, 2] for idx in (7, 8, 10, 11) if corrected[f, idx, 2] == corrected[f, idx, 2]]
        if not foot_z:
            continue
        min_z = min(foot_z)
        if min_z < 0:
            corrected[f, :, 2] -= min_z
        elif min_z > MAX_FOOT_HEIGHT:
            corrected[f, :, 2] -= (min_z - MAX_FOOT_HEIGHT)
    return corrected, len(shift_f)


print("loading VIMO...")
model = get_hmr_vimo(checkpoint="/project/6101776/ciro/Badminton LLM/tram/data/pretrain/vimo_checkpoint.pth.tar")
device = next(model.parameters()).device
img_focal = float(K[0, 0])
img_center = np.array([K[0, 2], K[1, 2]], dtype=np.float32)

imgfiles = sorted(glob.glob(f"{ROOT}/frames/*.jpg"))
n_frames = len(imgfiles)
print(f"{n_frames} frames, {len(tracks)} tracked players")

out = {"bones": BONES, "camera": cam, "players": {}}
for name, tdata in tracks.items():
    boxes = np.array(tdata["boxes"], dtype=np.float32)
    valid = np.array(tdata["valid"], dtype=bool)
    ankle = np.array(tdata["ankle"], dtype=np.float32)
    n_valid = int(valid.sum())
    print(f"\n{name}: {n_valid}/{n_frames} valid boxes")
    if n_valid < 8:
        print("  too few valid boxes, skipping")
        continue
    frame_full = np.arange(n_frames)
    res = model.inference(np.array(imgfiles), boxes, valid=valid, frame=frame_full,
                           img_focal=img_focal, img_center=img_center)
    if res is None:
        print("  inference returned None")
        continue
    smpl_out = model.smpl.query({"pred_rotmat": res["pred_rotmat"].to(device),
                                  "pred_shape": res["pred_shape"].to(device)},
                                  default_smpl=True)
    j3d = smpl_out.joints[:, :24].cpu()
    j3d_cam = (j3d + res["pred_trans"]).numpy()
    j3d_world = np.zeros_like(j3d_cam)
    for f in range(j3d_cam.shape[0]):
        j3d_world[f] = (R.T @ (j3d_cam[f] - t).T).T
    recon_frame_idx = res["frame"].numpy().astype(int)
    j3d_world_full = np.full((n_frames, 24, 3), np.nan)
    j3d_world_full[recon_frame_idx] = j3d_world
    j3d_world_full, n_anchors = ground_correct(j3d_world_full, ankle)
    out["players"][name] = j3d_world_full.tolist()
    print(f"  {len(recon_frame_idx)}/{n_frames} reconstructed, {n_anchors} ground anchors")

json.dump(out, open(f"{ROOT}/joints.json", "w"))
print(f"\nDONE, saved {ROOT}/joints.json")
