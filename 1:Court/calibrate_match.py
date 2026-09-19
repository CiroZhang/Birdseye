"""Camera self-calibration for any BFMD match, using its own court+net
annotation (4 outer corners + 4 net points) as correspondence points --
generalized from calibrate_pilot.py to take a match name."""
import json
import os

import numpy as np

from court_detection import court_detection

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get(
    "BFMD_DATA_ROOT",
    os.path.normpath(os.path.join(_HERE, "..", "..", "Dataset", "BFMD", "BFMD_data")),
)


def calibrate(match_name, verbose=True):
    court = json.load(open(f"{BASE}/annotations/court/{match_name}_court.json"))
    W, H = court["resolution"]["width"], court["resolution"]["height"]

    points_2d = {
        "P1_TL": tuple(court["court"]["top_left"]),
        "P4_TR": tuple(court["court"]["top_right"]),
        "P2_BL": tuple(court["court"]["bottom_left"]),
        "P3_BR": tuple(court["court"]["bottom_right"]),
        "P15_netL": tuple(court["net"]["left_base"]),
        "P16_netR": tuple(court["net"]["right_base"]),
        "poleL_top": tuple(court["net"]["left_top"]),
        "poleR_top": tuple(court["net"]["right_top"]),
    }

    cd = court_detection(width=W, height=H)
    cd.set_detections(points_2d)
    K, R, t = cd._solve_pose_and_focal(names=list(points_2d.keys()), n_restarts=6)
    cd.K, cd.R, cd.t = K, R, t

    errs = [cd._per_point_error(name, K, R, t) for name in points_2d]
    cam_center = -R.T @ t
    det_r = np.linalg.det(R)
    if verbose:
        print(f"{match_name}: mean_err={np.mean(errs):.2f}px max_err={np.max(errs):.2f}px "
              f"height={cam_center[2]:.2f}m det(R)={det_r:.4f}")

    ok = np.mean(errs) < 10.0 and abs(det_r - 1.0) < 1e-3 and cam_center[2] > 0.5
    return {"K": K.tolist(), "R": R.tolist(), "t": t.tolist(), "width": W, "height": H,
            "mean_err_px": float(np.mean(errs)), "max_err_px": float(np.max(errs)),
            "cam_height_m": float(cam_center[2])}, ok


if __name__ == "__main__":
    import csv
    rows = [r for r in csv.DictReader(open(f"{BASE}/match_index.csv")) if r["match_name"] and r["match_name"] != "TOTAL"]
    results = {}
    for r in rows:
        m = r["match_name"]
        try:
            cam, ok = calibrate(m)
            results[m] = cam
            if not ok:
                print(f"  ^ WARNING: {m} calibration looks physically implausible")
        except Exception as e:
            print(f"{m}: FAILED ({e})")
    json.dump(results, open(f"{BASE}/all_cameras.json", "w"))
    print(f"\nsaved {len(results)}/{len(rows)} camera calibrations to {BASE}/all_cameras.json")
