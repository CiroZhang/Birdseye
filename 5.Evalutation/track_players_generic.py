"""Generic version of track_backview_players_v2.py: nearest-centroid
(Hungarian) player tracking on ANY back-view rally video, using the same
conf=0.12/iou=0.3 detection settings and MAX_MATCH_DIST that fixed the 4th
player's tracking on Rally-1. Parameterized by video path + output path so it
can run unchanged across every rally that shares the same fixed camera."""
import functools
import json
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)

VIDEO = sys.argv[1]
OUT_JSON = sys.argv[2]
L_ANKLE, R_ANKLE = 15, 16
MAX_MATCH_DIST = 250.0

model = YOLO("yolov8x-pose.pt")
results = model.predict(VIDEO, classes=[0], conf=0.12, iou=0.3, stream=False, verbose=False)
n_frames = len(results)
print(f"{n_frames} frames processed")


def centroid(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


per_frame = []
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
    kconfs = r.keypoints.conf.cpu().numpy() if (r.keypoints is not None and r.keypoints.conf is not None) else None
    ankles = []
    for i in range(len(boxes)):
        ank = None
        if kpts is not None:
            pconf = kconfs[i] if kconfs is not None else np.ones(17)
            a = [kpts[i][j] for j in (L_ANKLE, R_ANKLE) if pconf[j] > 0.3]
            if a:
                ank = np.mean(a, axis=0)
        ankles.append(ank)
    per_frame.append({"boxes": boxes, "ankles": ankles})

n_dets = [len(f["boxes"]) for f in per_frame]
print(f"detections per frame: min={min(n_dets)} max={max(n_dets)} frames_with_4={sum(1 for n in n_dets if n == 4)}/{n_frames}")

init_f = next((f for f, n in enumerate(n_dets) if n == 4), None)
if init_f is None:
    init_f = int(np.argmax(n_dets))
    print(f"WARNING: no frame with exactly 4 detections, using frame {init_f} with {n_dets[init_f]}")
n_slots = 4
print(f"initializing slots from frame {init_f}")
slots = [centroid(b) for b in per_frame[init_f]["boxes"][:n_slots]]
while len(slots) < n_slots:
    slots.append(np.array([1920.0, 1080.0]))

tracks = {s: {} for s in range(n_slots)}
for f in range(init_f, n_frames):
    fr = per_frame[f]
    boxes, ankles = fr["boxes"], fr["ankles"]
    if len(boxes) == 0:
        continue
    cents = np.array([centroid(b) for b in boxes])
    slot_pos = np.array(slots)
    cost = np.linalg.norm(slot_pos[:, None, :] - cents[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    for r_i, c_i in zip(row_ind, col_ind):
        if cost[r_i, c_i] > MAX_MATCH_DIST:
            continue
        tracks[r_i][f] = {"box": boxes[c_i].tolist(), "ankle": (ankles[c_i].tolist() if ankles[c_i] is not None else None)}
        slots[r_i] = cents[c_i]

if init_f > 0:
    slots_back = [centroid(b) for b in per_frame[init_f]["boxes"][:n_slots]]
    for f in range(init_f - 1, -1, -1):
        fr = per_frame[f]
        boxes, ankles = fr["boxes"], fr["ankles"]
        if len(boxes) == 0:
            continue
        cents = np.array([centroid(b) for b in boxes])
        slot_pos = np.array(slots_back)
        cost = np.linalg.norm(slot_pos[:, None, :] - cents[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r_i, c_i in zip(row_ind, col_ind):
            if cost[r_i, c_i] > MAX_MATCH_DIST:
                continue
            tracks[r_i][f] = {"box": boxes[c_i].tolist(), "ankle": (ankles[c_i].tolist() if ankles[c_i] is not None else None)}
            slots_back[r_i] = cents[c_i]

for tid in range(n_slots):
    print(f"  slot {tid}: {len(tracks[tid])}/{n_frames} frames")

out = {}
for tid in range(n_slots):
    tdata = tracks[tid]
    boxes_arr = np.full((n_frames, 4), np.nan)
    valid_arr = np.zeros(n_frames, dtype=bool)
    ankle_arr = np.full((n_frames, 2, 2), np.nan)
    for f, d in tdata.items():
        boxes_arr[f] = d["box"]
        valid_arr[f] = True
        if d["ankle"] is not None:
            ankle_arr[f, 0] = d["ankle"]
            ankle_arr[f, 1] = d["ankle"]
    out[f"player_{tid + 1}"] = {
        "boxes": boxes_arr.tolist(),
        "valid": valid_arr.tolist(),
        "ankle": ankle_arr.tolist(),
    }

json.dump(out, open(OUT_JSON, "w"))
print(f"\nsaved {OUT_JSON}")
