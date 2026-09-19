"""Rebuild the back-view skeleton overlay video with the fixed conf=0.12
detection threshold, so the video panel shows all 4 players -- same
detection settings that fixed the 4th player's tracking/reconstruction
(conf=0.3 was hiding them at ~0.13-0.20 confidence)."""
import functools

import cv2
import numpy as np
import torch
from ultralytics import YOLO

_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)

BACK_VIEW = "/tmp/drone_doubles/Study_Videos/BD12/Rally-1/video_a.mp4"
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

model = YOLO("yolov8x-pose.pt")

cap = cv2.VideoCapture(BACK_VIEW)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_w, out_h = 960, 540
scale_x, scale_y = out_w / w, out_h / h
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("/tmp/back_view_skeleton.mp4", fourcc, fps, (out_w, out_h))

print(f"video: {w}x{h} @ {fps}fps")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, classes=[0], conf=0.12, iou=0.3, verbose=False)
    r = results[0]
    small_frame = cv2.resize(frame, (out_w, out_h))
    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        kpts = r.keypoints.xy.cpu().numpy()
        confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
        for person_idx in range(kpts.shape[0]):
            pts_orig = kpts[person_idx]
            pts = pts_orig * [scale_x, scale_y]
            pconf = confs[person_idx] if confs is not None else np.ones(17)
            # YOLOv8-pose sometimes snaps an occluded keypoint to literal
            # pixel (0,0) with a confidence just above our threshold instead
            # of a properly low one -- reject those explicitly, since no real
            # keypoint in this footage legitimately sits at the frame origin.
            valid = pconf > 0.3
            valid &= (pts_orig[:, 0] > 5) | (pts_orig[:, 1] > 5)
            for a, b in SKELETON:
                if valid[a] and valid[b]:
                    pa, pb = tuple(pts[a].astype(int)), tuple(pts[b].astype(int))
                    cv2.line(small_frame, pa, pb, (61, 220, 132), 2)
            for j in range(17):
                if valid[j]:
                    cv2.circle(small_frame, tuple(pts[j].astype(int)), 3, (94, 94, 255), -1)
    out.write(small_frame)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"  frame {frame_idx}")

cap.release()
out.release()
print(f"done, {frame_idx} frames, wrote /tmp/back_view_skeleton.mp4")
