"""Our own player detection on a BFMD match, replacing the annotation
shortcut (`bbox_by_frame.json` read directly from BFMD's player_bbox) with a
real detector -- same hybrid scheme as the validated original pipeline
(run_vimo_batch_v2.py): plain YOLOv8x picks the two player boxes (robust,
near-0% miss rate), YOLOv8x-pose is matched on top purely for ankle
keypoints. Candidates are filtered to inside the real court (+ margin) using
OUR OWN self-detected camera (all_cameras_self_detected.json), not BFMD's
annotation-based one, then split into near/far by floor-projected depth.

Usage: python3 detect_players_bfmd.py <match_name>
Expects (reusing the existing bfmd_all layout from the original VIMO run):
  /scratch/ciro/bfmd_all/<match>/video.mp4, /scratch/ciro/bfmd_all/<match>/rallies.json
  /scratch/ciro/bfmd_all/all_cameras_self_detected.json (the NEW self-detected camera, not the
  annotation-based all_cameras.json already sitting there from the original run)
Resume-safe: writes one output file per match, skips rallies already done.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
from ultralytics import YOLO

ROOT = "/scratch/ciro/bfmd_all"
FPS = 30
COURT_WIDTH, COURT_LENGTH = 6.1, 13.4
COURT_MARGIN = 1.0
W, H = 1280, 720

MATCH = sys.argv[1]
MATCH_DIR = f"{ROOT}/{MATCH}"
OUT_PATH = f"{MATCH_DIR}/boxes_selfdetected.json"

cam_all = json.load(open(f"{ROOT}/all_cameras_self_detected.json"))
cam = cam_all[MATCH]
K, R, t = np.array(cam["K"]), np.array(cam["R"]), np.array(cam["t"])


def floor_xy(u, v):
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
    return point[0], point[1]


def in_court(xy):
    if xy is None:
        return False
    x, y = xy
    return (-COURT_MARGIN <= x <= COURT_WIDTH + COURT_MARGIN and
            -COURT_MARGIN <= y <= COURT_LENGTH + COURT_MARGIN)


VIDEO = f"{MATCH_DIR}/video.mp4"
rallies = json.load(open(f"{MATCH_DIR}/rallies.json"))

if os.path.exists(OUT_PATH):
    out_data = json.load(open(OUT_PATH))
    print(f"resuming: {len(out_data)} rallies already done")
else:
    out_data = {}

print("loading YOLO models...")
det_model = YOLO("yolov8x.pt")
pose_model = YOLO("yolov8x-pose.pt")


def match_box_to_pose(box, pose_boxes):
    if len(pose_boxes) == 0:
        return None
    bc = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    pc = np.stack([(pose_boxes[:, 0] + pose_boxes[:, 2]) / 2,
                   (pose_boxes[:, 1] + pose_boxes[:, 3]) / 2], axis=1)
    d = np.linalg.norm(pc - bc, axis=1)
    j = int(np.argmin(d))
    box_diag = np.linalg.norm([box[2] - box[0], box[3] - box[1]])
    if d[j] < 0.5 * box_diag:
        return j
    return None


FRAMES_DIR = f"/scratch/ciro/bfmd_selfdetect/{MATCH}/frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

for ridx, r in enumerate(rallies):
    tag = f"g{r['game']}_r{r['rally']:03d}"
    if tag in out_data:
        continue
    start, end = r["start"], r["end"]
    n_frames = end - start + 1
    print(f"[{ridx+1}/{len(rallies)}] {MATCH} {tag}: frames {start}-{end} ({n_frames})", flush=True)

    pattern_dir = f"{FRAMES_DIR}/{tag}"
    os.makedirs(pattern_dir, exist_ok=True)
    existing = sorted(glob.glob(f"{pattern_dir}/*.jpg"))
    if len(existing) != n_frames:
        start_t = start / FPS
        # some source videos (VICTOR China Open + one YONEX match) are encoded
        # with a non-full-range YUV pixel format that ffmpeg's mjpeg encoder
        # refuses under strict standard compliance; -strict unofficial is
        # ffmpeg's own suggested fix for exactly this warning
        cmd = ["/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/ffmpeg", "-y", "-ss", f"{start_t:.6f}", "-i", VIDEO,
               "-frames:v", str(n_frames), "-qscale:v", "2", "-strict", "unofficial", "-threads", "1",
               f"{pattern_dir}/%06d.jpg", "-hide_banner", "-loglevel", "error"]
        subprocess.run(cmd, check=True)
    imgfiles = sorted(glob.glob(f"{pattern_dir}/*.jpg"))
    if len(imgfiles) != n_frames:
        print(f"  WARNING: expected {n_frames}, got {len(imgfiles)}, skipping")
        continue

    top_boxes = np.zeros((n_frames, 4), dtype=np.float32)
    bottom_boxes = np.zeros((n_frames, 4), dtype=np.float32)
    top_valid = np.zeros(n_frames, dtype=bool)
    bottom_valid = np.zeros(n_frames, dtype=bool)
    top_ankle = np.full((n_frames, 2, 2), np.nan)     # [frame, L/R, xy]
    bottom_ankle = np.full((n_frames, 2, 2), np.nan)

    for i, imgfile in enumerate(imgfiles):
        result = det_model.predict(imgfile, classes=[0], verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        if len(boxes) < 2:
            continue
        candidates = []
        for box, conf in zip(boxes, confs):
            u = (box[0] + box[2]) / 2
            v = box[3]
            xy = floor_xy(u, v)
            if in_court(xy):
                candidates.append((conf, xy[1], box))
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda c: -c[0])
        candidates = candidates[:2]
        candidates.sort(key=lambda c: c[1])  # near (small y) first -> "bottom", far -> "top"

        pose_result = pose_model.predict(imgfile, classes=[0], verbose=False)[0]
        pose_boxes = (pose_result.boxes.xyxy.cpu().numpy()
                      if pose_result.keypoints is not None and len(pose_result.boxes) > 0
                      else np.zeros((0, 4)))
        pose_kpts = (pose_result.keypoints.xy.cpu().numpy()
                     if pose_result.keypoints is not None and len(pose_result.boxes) > 0
                     else np.zeros((0, 17, 2)))

        for side, cand in zip(["bottom", "top"], candidates):
            box = cand[2]
            if side == "bottom":
                bottom_boxes[i] = box
                bottom_valid[i] = True
            else:
                top_boxes[i] = box
                top_valid[i] = True
            j = match_box_to_pose(box, pose_boxes)
            if j is not None:
                ank = pose_kpts[j, 15:17]  # left_ankle, right_ankle
                if side == "bottom":
                    bottom_ankle[i] = ank
                else:
                    top_ankle[i] = ank

    out_data[tag] = {
        "top_boxes": top_boxes.tolist(), "top_valid": top_valid.tolist(), "top_ankle": top_ankle.tolist(),
        "bottom_boxes": bottom_boxes.tolist(), "bottom_valid": bottom_valid.tolist(), "bottom_ankle": bottom_ankle.tolist(),
        "n_frames": n_frames,
    }
    print(f"  top: {int(top_valid.sum())}/{n_frames} detected, bottom: {int(bottom_valid.sum())}/{n_frames} detected", flush=True)
    json.dump(out_data, open(OUT_PATH, "w"))

    for f in imgfiles:
        os.remove(f)

print(f"\n{MATCH}: ALL DONE, saved to {OUT_PATH}")
