"""BFMD VIMO reconstruction using OUR OWN detections end to end: camera from
all_cameras_self_detected.json (MonoTrack + fit_self_calibrated, not
annotations), player boxes from boxes_selfdetected.json (YOLOv8x hybrid
detection, not BFMD's player_bbox annotations) -- and, since we now have real
detected ankle keypoints instead of annotation-derived ones, the real
ground/foot correction (ray-cast a pixel-stable ankle through the camera onto
the floor plane, rigid-shift the body, MAD-reject outliers, interpolate,
floor+height clamp) is applied automatically for every rally, not just a demo.

Expects, under /scratch/ciro/bfmd_all/<match_name>/:
  video.mp4, rallies.json, boxes_selfdetected.json
  and /scratch/ciro/bfmd_all/all_cameras_self_detected.json (shared)
Resume-safe: reloads joints_selfdetected.json if present, skips rallies already done.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/project/6101776/ciro/Badminton LLM/tram")
from lib.models import get_hmr_vimo

MATCH = sys.argv[1]
ROOT = f"/scratch/ciro/bfmd_all/{MATCH}"
FPS = 30
OUT_PATH = f"{ROOT}/joints_selfdetected.json"
STABLE_PIX_THRESH = 6.0
MAD_REJECT_K = 5.0
MAX_FOOT_HEIGHT = 1.1

cam_all = json.load(open("/scratch/ciro/bfmd_all/all_cameras_self_detected.json"))
cam = cam_all[MATCH]
K = np.array(cam["K"]); R = np.array(cam["R"]); t = np.array(cam["t"])

rallies = json.load(open(f"{ROOT}/rallies.json"))
boxes_data = json.load(open(f"{ROOT}/boxes_selfdetected.json"))

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
    """joints: (n_frames, 24, 3) world coords, possibly NaN. ankle_px:
    (n_frames, 2, 2) [L/R, xy] pixel coords, possibly NaN. Real v2-style
    correction: ray-cast a pixel-stable ankle onto the floor, rigid-shift the
    whole body so VIMO's own (lower) ankle joint lands there, MAD-reject
    outlier anchors, interpolate the shift, then a hard floor + max-height
    clamp as a final safety net."""
    n = joints.shape[0]
    px = np.nanmean(ankle_px, axis=1)  # avg L/R pixel as one 2D anchor source, (n,2)

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


if os.path.exists(OUT_PATH):
    out_data = json.load(open(OUT_PATH))
    print(f"resuming: {len(out_data['rallies'])} rallies already done")
else:
    out_data = {"bones": BONES, "camera": {"K": K.tolist(), "R": R.tolist(), "t": t.tolist()}, "rallies": {}}

print(f"loading VIMO for {MATCH}...")
model = get_hmr_vimo(checkpoint="/project/6101776/ciro/Badminton LLM/tram/data/pretrain/vimo_checkpoint.pth.tar")
device = next(model.parameters()).device

img_focal = float(K[0, 0])
img_center = np.array([K[0, 2], K[1, 2]], dtype=np.float32)

FRAMES_DIR = f"{ROOT}/frames_sd"
os.makedirs(FRAMES_DIR, exist_ok=True)

for ridx, r in enumerate(rallies):
    tag = f"g{r['game']}_r{r['rally']:03d}"
    if tag in out_data["rallies"]:
        continue
    if tag not in boxes_data:
        print(f"  {tag}: no self-detected boxes, skipping")
        continue
    start, end = r["start"], r["end"]
    n_frames = end - start + 1
    print(f"\n[{ridx+1}/{len(rallies)}] {MATCH} rally {tag}: frames {start}-{end} ({n_frames} frames)", flush=True)

    pattern_dir = f"{FRAMES_DIR}/{tag}"
    os.makedirs(pattern_dir, exist_ok=True)
    existing = sorted(glob.glob(f"{pattern_dir}/*.jpg"))
    if len(existing) != n_frames:
        start_t = start / FPS
        cmd = ["/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/ffmpeg", "-y", "-ss", f"{start_t:.6f}", "-i", f"{ROOT}/video.mp4",
               "-frames:v", str(n_frames), "-qscale:v", "2", "-strict", "unofficial", "-threads", "1",
               f"{pattern_dir}/%06d.jpg", "-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd, check=True)
    imgfiles = sorted(glob.glob(f"{pattern_dir}/*.jpg"))
    if len(imgfiles) != n_frames:
        print(f"  WARNING: expected {n_frames} frames, got {len(imgfiles)}, skipping rally")
        continue

    rb = boxes_data[tag]
    boxes = {"top": np.array(rb["top_boxes"], dtype=np.float32),
             "bottom": np.array(rb["bottom_boxes"], dtype=np.float32)}
    valid = {"top": np.array(rb["top_valid"], dtype=bool),
             "bottom": np.array(rb["bottom_valid"], dtype=bool)}
    ankle = {"top": np.array(rb["top_ankle"], dtype=np.float32),
             "bottom": np.array(rb["bottom_ankle"], dtype=np.float32)}

    rally_out = {}
    for side in ["top", "bottom"]:
        n_valid = int(valid[side].sum())
        if n_valid < 8:
            print(f"  {side}: only {n_valid} valid boxes, skipping")
            continue
        frame_full = np.arange(n_frames)
        res = model.inference(np.array(imgfiles), boxes[side], valid=valid[side], frame=frame_full,
                               img_focal=img_focal, img_center=img_center)
        if res is None:
            print(f"  {side}: inference returned None")
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

        j3d_world_full, n_anchors = ground_correct(j3d_world_full, ankle[side])

        rally_out[side] = j3d_world_full.tolist()
        print(f"  {side}: {len(recon_frame_idx)}/{n_frames} reconstructed, "
              f"{n_anchors} ground anchors", flush=True)

    if rally_out:
        out_data["rallies"][tag] = {"start": start, "end": end, "joints": rally_out}
        json.dump(out_data, open(OUT_PATH, "w"))

    for f in imgfiles:
        os.remove(f)

print(f"\n{MATCH}: ALL DONE")
