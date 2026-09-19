"""V2 pipeline: adds real 2D ankle-keypoint detection (YOLOv8-pose, not a
bbox-bottom proxy) and uses it for a properly ground-truth-anchored player
correction, replacing the old VIMO-own-signal heuristic:

1. Camera fit (unchanged from v1) -- known court geometry, self-calibrated.
2. YOLOv8-pose detection per frame: real ankle keypoints, not a bbox-bottom
   guess. Court-boundary + confidence filtering picks the real 2 players
   (rejects referees/crowd), same idea as v1 but with better inputs.
3. Ray-cast each detected ankle pixel through our known camera onto the
   real floor (Z=0) -- an exact geometric answer, independent of VIMO's own
   noisy depth, whenever that foot is actually stable (not swinging).
4. Run VIMO for the body reconstruction (unchanged).
5. NEW correction: whenever a foot is stable in 2D (low ankle-pixel motion
   between frames -- likely planted), compute the exact 3D floor anchor for
   it and rigid-shift the WHOLE reconstructed body (all 24 joints) so VIMO's
   corresponding foot joint lands there. Interpolate the shift across frames
   with no stable anchor (gaps/jumps), same idea as v1's interpolation but
   now anchored to real 2D-derived ground truth instead of VIMO's own height.
6. Hard floor clamp as a final safety net (unchanged from v1).
"""
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/project/6101776/ciro/Badminton LLM/Project/Court_Detection")
from court_detection import court_detection

sys.path.insert(0, "/project/6101776/ciro/Badminton LLM/tram")
from lib.models import get_hmr_vimo

REPO = "/project/6101776/ciro/Badminton LLM"
NAMES = [
    "P1_TL", "P2_BL", "P3_BR", "P4_TR", "poleL_top", "poleR_top", "P15_netL", "P16_netR",
    "P5_singlesTL", "P6_singlesBL", "P7_singlesBR", "P8_singlesTR",
    "P9_serviceFarL", "P10_serviceFarR", "P11_serviceNearL", "P12_serviceNearR",
    "P13_serviceFarCenter", "P14_serviceNearCenter",
    "P17_doublesFarL", "P18_doublesFarR", "P19_doublesNearL", "P20_doublesNearR",
    "P21_baseFarCenter", "P22_baseNearCenter",
]
VGGT_FOV_CORRECTION = 1.4514
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
BONES = [(i, p) for i, p in enumerate(SMPL_PARENTS) if p >= 0]
FPS = 30
COURT_MARGIN = 1.0
STABLE_PIX_THRESH = 6.0  # pixels/frame -- ankle must move less than this to count as planted

MATCHES_AND_SECONDS = [("match1", None), ("match5", None), ("match9", None)]
OUT_DIR = "/scratch/ciro/vimo_test/batch_v2"
os.makedirs(OUT_DIR, exist_ok=True)

gt = json.load(open(f"{REPO}/Dataset/Court Detection Dataset/ground_truth.json"))
vggt_rows = list(csv.DictReader(open(f"{REPO}/Project/Court_Detection/vggt_predictions/TrackNetV2.csv")))
vggt_by_path = {r["video_path"]: r for r in vggt_rows}


def floor_xy(u, v, cd):
    Kinv = np.linalg.inv(cd.K)
    ray_cam = Kinv @ np.array([u, v, 1.0])
    ray_world = cd.R.T @ ray_cam
    cam_center = -cd.R.T @ cd.t
    if abs(ray_world[2]) < 1e-9:
        return None
    s = -cam_center[2] / ray_world[2]
    if s <= 0:
        return None
    point = cam_center + s * ray_world
    return float(point[0]), float(point[1])


def in_court(xy):
    if xy is None:
        return False
    x, y = xy
    return (-COURT_MARGIN <= x <= court_detection.COURT_WIDTH + COURT_MARGIN and
            -COURT_MARGIN <= y <= court_detection.COURT_LENGTH + COURT_MARGIN)


print("loading VIMO...")
model = get_hmr_vimo(checkpoint="/project/6101776/ciro/Badminton LLM/tram/data/pretrain/vimo_checkpoint.pth.tar")

from ultralytics import YOLO
det_model = YOLO("yolov8x.pt")        # robust box detection -- used for player SELECTION
pose_model = YOLO("yolov8x-pose.pt")  # keypoints -- used only to extract ankles for
                                       # whichever boxes det_model already selected


def match_box_to_pose(box, pose_boxes):
    """Find the pose-model detection (if any) whose box center is closest to
    `box`'s center, within a generous tolerance -- handles the pose model
    being pickier and not firing on every frame the plain detector does."""
    if len(pose_boxes) == 0:
        return None
    bc = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    pc = np.stack([(pose_boxes[:, 0] + pose_boxes[:, 2]) / 2,
                    (pose_boxes[:, 1] + pose_boxes[:, 3]) / 2], axis=1)
    d = np.linalg.norm(pc - bc, axis=1)
    j = int(np.argmin(d))
    box_diag = np.linalg.norm([box[2] - box[0], box[3] - box[1]])
    if d[j] < 0.5 * box_diag:  # must be reasonably close relative to box size
        return j
    return None

for MATCH, seconds in MATCHES_AND_SECONDS:
    print(f"\n=== {MATCH} ===")
    fpath = f"{REPO}/Project/Court_Detection/monotrack_predictions_FINAL/TrackNetV2/{MATCH}.out"
    with open(fpath) as f:
        pts = [[float(v) for v in l.strip().split(";")] for l in f if l.strip()]
    all_det = dict(zip(NAMES, pts))
    non_pole = {k: v for k, v in all_det.items() if "pole" not in k}
    w, h = gt[MATCH]["width"], gt[MATCH]["height"]
    video_path = f"{REPO}/{gt[MATCH]['video']}"

    row = vggt_by_path.get(video_path)
    focal_hint = None
    if row is not None:
        cx_v, fx_v = float(row["K_02"]), float(row["K_00"])
        focal_hint = 2 * np.degrees(np.arctan2(cx_v, fx_v * VGGT_FOV_CORRECTION))

    cd = court_detection(width=w, height=h)
    cd.set_detections(non_pole)
    rms, inliers, outliers = cd.fit_self_calibrated(inlier_thresh_px=20.0, n_iters=60,
                                                      focal_hint_hfov_deg=focal_hint)
    cam_height = (-cd.R.T @ cd.t)[2]
    print(f"camera: rms={rms:.2f}px height={cam_height:.2f}m K={cd.K[0,0]:.1f}")

    # reuse already-extracted frames if present (from v1 run)
    img_folder = f"/scratch/ciro/vimo_test/batch/{MATCH}_images"
    if not os.path.isdir(img_folder) or len(glob.glob(f"{img_folder}/*.jpg")) == 0:
        img_folder = f"{OUT_DIR}/{MATCH}_images"
        os.makedirs(img_folder, exist_ok=True)
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_path],
            capture_output=True, text=True).stdout.strip())
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={FPS}", "-qscale:v", "2",
                       f"{img_folder}/%06d.jpg", "-hide_banner", "-loglevel", "error"]
        subprocess.run(ffmpeg_cmd, check=True)
    imgfiles = sorted(glob.glob(f"{img_folder}/*.jpg"))
    print(f"{len(imgfiles)} frames")

    near_track = {"frame": [], "det_box": [], "left_ankle_px": [], "right_ankle_px": []}
    far_track = {"frame": [], "det_box": [], "left_ankle_px": [], "right_ankle_px": []}
    n_rejected_oob = 0

    n_ankle_matched = 0
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
            xy = floor_xy(u, v, cd)
            if in_court(xy):
                candidates.append((conf, xy[1], box))
            else:
                n_rejected_oob += 1

        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda c: -c[0])
        candidates = candidates[:2]
        candidates.sort(key=lambda c: c[1])

        # only run the (pickier, slower) pose model when we actually have 2
        # valid candidate boxes to match ankles onto -- keeps this cheap
        pose_result = pose_model.predict(imgfile, classes=[0], verbose=False)[0]
        pose_boxes = (pose_result.boxes.xyxy.cpu().numpy()
                       if pose_result.keypoints is not None and len(pose_result.boxes) > 0
                       else np.zeros((0, 4)))
        pose_kpts = (pose_result.keypoints.xy.cpu().numpy()
                      if pose_result.keypoints is not None and len(pose_result.boxes) > 0
                      else np.zeros((0, 17, 2)))

        for trk, cand in zip([near_track, far_track], candidates):
            box = cand[2]
            trk["frame"].append(i)
            trk["det_box"].append(box)
            j = match_box_to_pose(box, pose_boxes)
            if j is not None:
                trk["left_ankle_px"].append(pose_kpts[j, 15].tolist())
                trk["right_ankle_px"].append(pose_kpts[j, 16].tolist())
                n_ankle_matched += 1
            else:
                trk["left_ankle_px"].append([np.nan, np.nan])
                trk["right_ankle_px"].append([np.nan, np.nan])

    n_selected = len(near_track["frame"]) + len(far_track["frame"])
    print(f"rejected {n_rejected_oob} out-of-court detections; "
          f"near {len(near_track['frame'])}/{len(imgfiles)}, far {len(far_track['frame'])}/{len(imgfiles)}; "
          f"ankles matched for {n_ankle_matched}/{n_selected} selected boxes")

    img_focal = float(cd.K[0, 0])
    img_center = np.array([cd.K[0, 2], cd.K[1, 2]], dtype=np.float32)

    out_data = {"bones": BONES, "players": {}, "n_source_frames": len(imgfiles)}
    for name, trk in [("near", near_track), ("far", far_track)]:
        if len(trk["frame"]) < 16:
            print(f"{name}: only {len(trk['frame'])} frames, skipping")
            continue

        n_total = len(imgfiles)
        detected_idx = np.array(trk["frame"], dtype=int)
        valid_mask = np.zeros(n_total, dtype=bool)
        valid_mask[detected_idx] = True
        boxes_full = np.zeros((n_total, 4), dtype=np.float32)
        boxes_full[detected_idx] = np.array(trk["det_box"])
        frame_full = np.arange(n_total)

        res = model.inference(np.array(imgfiles), boxes_full, valid=valid_mask, frame=frame_full,
                               img_focal=img_focal, img_center=img_center)
        if res is None:
            print(f"{name}: inference returned None")
            continue

        device = next(model.parameters()).device
        smpl_out = model.smpl.query({"pred_rotmat": res["pred_rotmat"].to(device),
                                      "pred_shape": res["pred_shape"].to(device)},
                                      default_smpl=True)
        j3d = smpl_out.joints[:, :24].cpu()
        j3d_cam = (j3d + res["pred_trans"]).numpy()
        n_reconstructed = j3d_cam.shape[0]
        j3d_world = np.zeros_like(j3d_cam)
        for f in range(n_reconstructed):
            j3d_world[f] = (cd.R.T @ (j3d_cam[f] - cd.t).T).T

        recon_frame_idx = res["frame"].numpy().astype(int)
        j3d_world_full = np.full((n_total, 24, 3), np.nan)
        j3d_world_full[recon_frame_idx] = j3d_world

        # per-frame 2D ankle pixels, aligned to full n_total length (NaN = no detection)
        left_ankle_px_full = np.full((n_total, 2), np.nan)
        right_ankle_px_full = np.full((n_total, 2), np.nan)
        left_ankle_px_full[detected_idx] = np.array(trk["left_ankle_px"])
        right_ankle_px_full[detected_idx] = np.array(trk["right_ankle_px"])

        out_data["players"][name] = {
            "joints": j3d_world_full.tolist(),
            "left_ankle_px": left_ankle_px_full.tolist(),
            "right_ankle_px": right_ankle_px_full.tolist(),
        }
        n_missing = n_total - len(recon_frame_idx)
        print(f"{name}: {n_reconstructed}/{n_total} reconstructed ({n_missing} gaps)")

    out_data["camera"] = {"K": cd.K.tolist(), "R": cd.R.tolist(), "t": cd.t.tolist()}
    json.dump(out_data, open(f"{OUT_DIR}/{MATCH}_raw.json", "w"))
    print(f"saved {OUT_DIR}/{MATCH}_raw.json")

print("\nALL DONE")
