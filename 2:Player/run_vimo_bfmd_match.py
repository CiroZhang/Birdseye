"""BFMD, generalized to any match: run VIMO on rally-interval frames using
BFMD's own player bounding boxes, camera pre-calibrated locally. Same logic
as run_vimo_bfmd_pilot.py, parametrized by match name (passed as argv[1],
matching a key in all_cameras.json and a directory under prepped/).

Expects, under /scratch/ciro/bfmd_all/<match_name>/:
  video.mp4, camera.json, rallies.json, bbox_by_frame.json
Resume-safe: reloads joints.json if present, skips rallies already done.
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
OUT_PATH = f"{ROOT}/joints.json"

cam = json.load(open(f"{ROOT}/camera.json"))
K = np.array(cam["K"]); R = np.array(cam["R"]); t = np.array(cam["t"])

rallies = json.load(open(f"{ROOT}/rallies.json"))
bbox_by_frame = json.load(open(f"{ROOT}/bbox_by_frame.json"))

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
BONES = [(i, p) for i, p in enumerate(SMPL_PARENTS) if p >= 0]

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

FRAMES_DIR = f"{ROOT}/frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

for ridx, r in enumerate(rallies):
    tag = f"g{r['game']}_r{r['rally']:03d}"
    if tag in out_data["rallies"]:
        continue
    start, end = r["start"], r["end"]
    n_frames = end - start + 1
    print(f"\n[{ridx+1}/{len(rallies)}] {MATCH} rally {tag}: frames {start}-{end} ({n_frames} frames)", flush=True)

    pattern_dir = f"{FRAMES_DIR}/{tag}"
    os.makedirs(pattern_dir, exist_ok=True)
    existing = sorted(glob.glob(f"{pattern_dir}/*.jpg"))
    if len(existing) != n_frames:
        start_t = start / FPS
        cmd = ["ffmpeg", "-y", "-ss", f"{start_t:.6f}", "-i", f"{ROOT}/video.mp4",
               "-frames:v", str(n_frames), "-qscale:v", "2",
               f"{pattern_dir}/%06d.jpg", "-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd, check=True)
    imgfiles = sorted(glob.glob(f"{pattern_dir}/*.jpg"))
    if len(imgfiles) != n_frames:
        print(f"  WARNING: expected {n_frames} frames, got {len(imgfiles)}, skipping rally")
        continue

    boxes = {"top": np.zeros((n_frames, 4), dtype=np.float32),
             "bottom": np.zeros((n_frames, 4), dtype=np.float32)}
    valid = {"top": np.zeros(n_frames, dtype=bool), "bottom": np.zeros(n_frames, dtype=bool)}
    for local_i in range(n_frames):
        global_f = start + local_i
        fb = bbox_by_frame.get(str(global_f), {})
        for side in ["top", "bottom"]:
            if side in fb:
                boxes[side][local_i] = fb[side]
                valid[side][local_i] = True

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
        rally_out[side] = j3d_world_full.tolist()
        print(f"  {side}: {len(recon_frame_idx)}/{n_frames} reconstructed", flush=True)

    if rally_out:
        out_data["rallies"][tag] = {"start": start, "end": end, "joints": rally_out}
        json.dump(out_data, open(OUT_PATH, "w"))

    for f in imgfiles:
        os.remove(f)

print(f"\n{MATCH}: ALL DONE")
