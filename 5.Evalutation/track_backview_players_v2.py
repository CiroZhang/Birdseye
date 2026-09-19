"""Real player tracking on the back-view drone doubles video, take 2.

ByteTrack (both stock and a low-new_track_thresh variant) kept losing and
re-acquiring the 4th (far-side, small-in-frame, ~0.13-0.20 conf) player --
its box is detected in nearly every frame, but ByteTrack's IoU-based
association breaks on the jitter, fragmenting it into 6 short track IDs
instead of one continuous one. Per-frame detection is NOT the problem
(confirmed by hand: 4 boxes present in every sampled frame at conf=0.12,
iou=0.3). So: run plain per-frame pose detection (no built-in tracker), then
do our own greedy nearest-centroid assignment across exactly 4 known player
slots via the Hungarian algorithm -- centroid motion frame-to-frame is much
smaller than the separation between distinct players except during the
brief net-crossing around frame ~113-125, so this is robust where IoU-based
matching on a noisy small box was not.
"""
import functools
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)

VIDEO = "/tmp/drone_doubles/Study_Videos/BD12/Rally-1/video_a.mp4"
L_ANKLE, R_ANKLE = 15, 16
MAX_MATCH_DIST = 250.0  # px, gate beyond which a detection can't match a slot

model = YOLO("yolov8x-pose.pt")
results = model.predict(VIDEO, classes=[0], conf=0.12, iou=0.3, stream=False, verbose=False)
n_frames = len(results)
print(f"{n_frames} frames processed")


def centroid(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


per_frame = []
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
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
    per_frame.append({"boxes": boxes, "confs": confs, "ankles": ankles})

n_dets = [len(f["boxes"]) for f in per_frame]
print(f"detections per frame: min={min(n_dets)} max={max(n_dets)} "
      f"frames_with_4={sum(1 for n in n_dets if n == 4)}/{n_frames}")

# initialize the 4 slots from the earliest frame with exactly 4 detections
init_f = next(f for f, n in enumerate(n_dets) if n == 4)
print(f"initializing slots from frame {init_f}")
slots = [centroid(b) for b in per_frame[init_f]["boxes"]]
n_slots = 4

tracks = {s: {} for s in range(n_slots)}
for f in range(init_f, n_frames):
    fr = per_frame[f]
    boxes, confs, ankles = fr["boxes"], fr["confs"], fr["ankles"]
    n_det = len(boxes)
    if n_det == 0:
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

# also walk backward from init_f to frame 0 the same way (in case some
# players' slot identity is established later than frame 0)
if init_f > 0:
    slots_back = [centroid(b) for b in per_frame[init_f]["boxes"]]
    for f in range(init_f - 1, -1, -1):
        fr = per_frame[f]
        boxes, confs, ankles = fr["boxes"], fr["confs"], fr["ankles"]
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

json.dump(out, open("/tmp/backview_tracked_players.json", "w"))
print("\nsaved /tmp/backview_tracked_players.json")
