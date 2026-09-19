# NOTE: local-machine-only script -- shells out to the compiled arm64 MonoTrack `detect` binary,
# which only exists on the Mac this was developed on (see readme.md section 1: "the real detector
# binary itself lives outside this project folder ... runs locally, no cluster needed"). Not
# runnable on Killarney or any other machine without that binary and paths below adjusted.
import json
import subprocess
import numpy as np

from court_detection import court_detection

gt = json.load(open("/tmp/tnv2_demo/ground_truth.json"))
DETECT_BIN = "/Users/cirozhang/Desktop/Badminton LLM/monotrack/badminton-court-detection/build/detect"

results = {}
for name, info in gt.items():
    video_mp4 = f"/tmp/tnv2_demo/videos/{name}.mp4"
    avi = f"/tmp/tnv2_demo/videos/{name}.avi"
    out_pts = f"/tmp/tnv2_demo/videos/{name}_pts.out"
    subprocess.run(["ffmpeg", "-y", "-i", video_mp4, "-c:v", "mjpeg", "-q:v", "3", "-an", avi, "-loglevel", "error"], check=True)
    subprocess.run([DETECT_BIN, avi, out_pts], capture_output=True, text=True, timeout=60)
    try:
        with open(out_pts) as f:
            pts = [[float(v) for v in l.strip().split(";")] for l in f if l.strip()]
    except FileNotFoundError:
        print(f"{name}: DETECT FAILED", flush=True)
        results[name] = None
        json.dump(results, open("/tmp/tnv2_demo/all_results.json", "w"))
        continue
    if len(pts) != 6:
        print(f"{name}: unexpected point count {len(pts)}", flush=True)
        results[name] = None
        json.dump(results, open("/tmp/tnv2_demo/all_results.json", "w"))
        continue
    names6 = ["P1_TL", "P2_BL", "P3_BR", "P4_TR", "poleL_top", "poleR_top"]
    det = dict(zip(names6, pts))

    corners = info["corners"]
    gt_corners = {"P2_BL": tuple(corners[0]), "P3_BR": tuple(corners[1]),
                  "P1_TL": tuple(corners[2]), "P4_TR": tuple(corners[3])}

    raw_errs = {n: float(np.hypot(det[n][0] - gt_corners[n][0], det[n][1] - gt_corners[n][1])) for n in gt_corners}

    w, h = info["width"], info["height"]
    cd = court_detection(width=w, height=h)
    cd.set_detections(det)
    calib_errs = None
    try:
        K, R, t = cd._solve_pose_and_focal(list(det.keys()), n_restarts=15)
        cd.K, cd.R, cd.t = K, R, t
        calib_errs = {}
        for n, (u, v) in gt_corners.items():
            p = cd.reproject(n)
            calib_errs[n] = float(np.hypot(p[0] - u, p[1] - v))
    except Exception as e:
        print(f"{name}: calib fit failed: {e}", flush=True)

    results[name] = {"det": det, "gt_corners": gt_corners, "raw_errs": raw_errs, "calib_errs": calib_errs}
    raw_mean = float(np.mean(list(raw_errs.values())))
    calib_mean = float(np.mean(list(calib_errs.values()))) if calib_errs else None
    print(f"{name}: raw_mean={raw_mean:.1f}px  calib_mean={calib_mean if calib_mean is None else round(calib_mean, 1)}px", flush=True)
    json.dump(results, open("/tmp/tnv2_demo/all_results.json", "w"))

print("ALL DONE", flush=True)
