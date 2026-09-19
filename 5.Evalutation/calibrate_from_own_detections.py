"""Calibrate BFMD cameras from OUR OWN detected court points (the real MonoTrack
detections already sitting in monotrack_bfmd_detections/, never used until now)
instead of BFMD's own annotated corners -- this is the actual validated method
(fit_self_calibrated, RANSAC, net poles excluded from calibration and instead
reprojected/constructed afterward), applied honestly to a new dataset.

Also measures reprojection accuracy against BFMD's own annotated court corners
as ground truth, for an honest "does our method generalize" number.
"""
import csv
import json
import os

import numpy as np

from court_detection import court_detection

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get(
    "BFMD_DATA_ROOT",
    os.path.normpath(os.path.join(_HERE, "..", "..", "Dataset", "BFMD", "BFMD_data")),
)
# NOTE: this is a duplicate of 1:Court/calibrate_from_own_detections.py that has drifted from the
# original -- 1:Court/court_detection.py and this folder's court_detection.py are no longer
# identical either. This folder has no monotrack_bfmd_detections/ of its own; it reads 1:Court's.
DET_DIR = os.path.normpath(os.path.join(_HERE, "..", "1:Court", "monotrack_bfmd_detections"))

# canonical 24-point order the MonoTrack detect binary always outputs in
# (verified against run_vimo_batch_v2.py's NAMES list, same binary/format)
NAMES = ["P1_TL", "P2_BL", "P3_BR", "P4_TR", "poleL_top", "poleR_top", "P15_netL", "P16_netR",
         "P5_singlesTL", "P6_singlesBL", "P7_singlesBR", "P8_singlesTR",
         "P9_serviceFarL", "P10_serviceFarR", "P11_serviceNearL", "P12_serviceNearR",
         "P13_serviceFarCenter", "P14_serviceNearCenter",
         "P17_doublesFarL", "P18_doublesFarR", "P19_doublesNearL", "P20_doublesNearR",
         "P21_baseFarCenter", "P22_baseNearCenter"]

W, H = 1280, 720

rows = [r for r in csv.DictReader(open(f"{BASE}/match_index.csv")) if r["match_name"] and r["match_name"] != "TOTAL"]
matches = [r["match_name"] for r in rows]

results = {}
for m in matches:
    with open(f"{DET_DIR}/{m}.out") as f:
        lines = [l.strip() for l in f if l.strip()]
    pts = [[float(v) for v in l.split(";")] for l in lines]
    if len(pts) != len(NAMES):
        print(f"{m}: SKIP, expected {len(NAMES)} points, got {len(pts)}")
        continue
    all_det = dict(zip(NAMES, pts))
    non_pole = {k: v for k, v in all_det.items() if "pole" not in k}

    cd = court_detection(width=W, height=H)
    cd.set_detections(non_pole)
    try:
        rms_px, inliers, outliers = cd.fit_self_calibrated(inlier_thresh_px=20.0, n_iters=100)
    except Exception as e:
        print(f"{m}: FAILED ({e})")
        continue

    cam_center = -cd.R.T @ cd.t
    det_r = np.linalg.det(cd.R)

    # honest accuracy check: reproject BFMD's OWN annotated court corners
    # through OUR self-detected-and-calibrated camera, compare to their
    # annotated pixel location -- this is the real "does our method work on
    # this benchmark" number, not just our own detector's self-consistency
    court_ann = json.load(open(f"{BASE}/annotations/court/{m}_court.json"))
    gt_points = {
        "P1_TL": court_ann["court"]["top_left"], "P4_TR": court_ann["court"]["top_right"],
        "P2_BL": court_ann["court"]["bottom_left"], "P3_BR": court_ann["court"]["bottom_right"],
        "P15_netL": court_ann["net"]["left_base"], "P16_netR": court_ann["net"]["right_base"],
        "poleL_top": court_ann["net"]["left_top"], "poleR_top": court_ann["net"]["right_top"],
    }
    gt_errs = []
    for name, (u_gt, v_gt) in gt_points.items():
        X = np.array(cd.COURT_POINTS_3D[name])
        Xc = cd.R @ X + cd.t
        proj = cd.K @ Xc
        proj = proj[:2] / proj[2]
        gt_errs.append(float(np.hypot(proj[0] - u_gt, proj[1] - v_gt)))

    results[m] = {
        "K": cd.K.tolist(), "R": cd.R.tolist(), "t": cd.t.tolist(),
        "n_inliers": len(inliers), "n_outliers": len(outliers),
        "self_consistency_rms_px": rms_px,
        "cam_height_m": float(cam_center[2]), "det_R": float(det_r),
        "vs_annotation_mean_err_px": float(np.mean(gt_errs)),
        "vs_annotation_max_err_px": float(np.max(gt_errs)),
    }
    print(f"{m}: self-fit rms={rms_px:.2f}px ({len(inliers)}/{len(non_pole)} inliers), "
          f"height={cam_center[2]:.2f}m, det(R)={det_r:.4f}, "
          f"vs BFMD annotation: mean={np.mean(gt_errs):.1f}px max={np.max(gt_errs):.1f}px")

json.dump(results, open(f"{BASE}/all_cameras_self_detected.json", "w"))
print(f"\nsaved {len(results)}/{len(matches)} -> {BASE}/all_cameras_self_detected.json")
