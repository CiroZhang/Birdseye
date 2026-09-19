"""Computes the final combined position error for the doubles drone
validation (BD12 Rally-1/5/20 + BD13 Rally-2), comparing our back-view
VIMO/SMPL reconstruction (ankle, ground-projected) against the real
top-view ground-truth CSV positions. Uses the corrected (length, width)
axis convention for COURT_CORNERS_WORLD in validate_against_drone.py --
the original (width, length) order was a real bug (confirmed both
visually and via the corner pixel-distance ratio) that inflated the
error from ~0.32m to ~3.0m median before the fix.

Also documents the 0.8m tolerance rationale: average adult male arm span
~1.8m; a badminton player's reconstructed foot/ankle position vs. the
ground-truth annotation point can reasonably differ by less than half an
arm span (~0.9m) given players often have an arm/racket extended -- 0.8m
is a slightly conservative fraction of that.
"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from validate_against_drone import fit_topview_homography, pixel_to_world

HALF_W, HALF_L = 6.1 / 2, 13.4 / 2
CORNERS_PX = [(190, 286), (3687, 296), (3723, 1913), (172, 1901)]
HTOP = fit_topview_homography(CORNERS_PX)

RALLIES = {
    "BD12_Rally1": ("BD12_Rally1_joints.json", "BD12/Rally-1/1.csv"),
    "BD12_Rally5": ("BD12_Rally5_joints.json", "BD12/Rally-5/1.csv"),
    "BD12_Rally20": ("BD12_Rally20_joints.json", "BD12/Rally-20/1.csv"),
    "BD13_Rally2": ("BD13_Rally2_joints.json", "BD13/Rally-2/1.csv"),
}


def topview_player_positions_safe(csv_path, H):
    """Same as validate_against_drone.topview_player_positions but tolerant
    of a handful of empty CSV cells (BD13/Rally-2 has 4 out of 1376)."""
    out = {}
    for r in csv.DictReader(open(csv_path)):
        frame = int(r["frame"])
        positions = {}
        for n in (1, 2, 3, 4):
            xs, ys = r.get(f"player{n}x", ""), r.get(f"player{n}y", "")
            if xs == "" or ys == "":
                continue
            px, py = float(xs), float(ys)
            if px < 0 or py < 0:
                continue
            wx, wy = pixel_to_world(H, px, py)
            positions[n] = (wx, wy)
        out[frame] = positions
    return out


def main(joints_dir, csv_dir):
    all_errors = []
    per_rally_summary = {}
    for name, (joints_file, csv_rel) in RALLIES.items():
        gt = topview_player_positions_safe(f"{csv_dir}/{csv_rel}", HTOP)
        frame_offset = sorted(gt.keys())[0]

        d = json.load(open(f"{joints_dir}/{joints_file}"))
        players = d["players"]
        n_frames = len(next(iter(players.values())))

        errors = []
        for f in range(n_frames):
            csv_frame = f + frame_offset
            if csv_frame not in gt:
                continue
            gt_positions = gt[csv_frame]
            recon_pts = []
            for pname, arr in players.items():
                j = np.array(arr[f])
                if np.isnan(j).any():
                    continue
                ankle = j[7] if j[7, 2] < j[8, 2] else j[8]
                wx, wy = ankle[0] - HALF_W, ankle[1] - HALF_L
                # (length, width) order -- matches the corrected COURT_CORNERS_WORLD
                recon_pts.append((wy, wx))
            if not recon_pts or not gt_positions:
                continue
            gtlist = list(gt_positions.values())
            used_gt = set()
            for rx, ry in recon_pts:
                dists = [(np.hypot(rx - gx, ry - gy), i) for i, (gx, gy) in enumerate(gtlist) if i not in used_gt]
                if not dists:
                    continue
                dmin, imin = min(dists)
                used_gt.add(imin)
                errors.append(dmin)
        errors = np.array(errors)
        all_errors.extend(errors.tolist())
        per_rally_summary[name] = {
            "n": int(len(errors)), "median_m": float(np.median(errors)),
            "mean_m": float(errors.mean()), "p90_m": float(np.percentile(errors, 90)),
            "max_m": float(errors.max()), "pct_within_0.8m": float(100 * (errors < 0.8).mean()),
        }

    all_errors = np.array(all_errors)
    tolerances = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    summary = {
        "method": "back-view single-camera VIMO/SMPL 3D reconstruction, ankle ground-projected, "
                  "compared against real top-view ground-truth CSV positions (nearest-neighbor "
                  "matched per frame). Camera: one hand-calibrated pose (8 real annotated points, "
                  "RMS 15.5px, solved via court_detection.py _solve_pose_and_focal) reused across "
                  "all rallies/matches sharing the same fixed physical camera (confirmed identical "
                  "framing across BD12 and BD13).",
        "tolerance_rationale": "0.8m chosen as a body-scale tolerance: average adult male arm span "
                  "~1.8m, and a badminton player's reconstructed 'position' (foot/ankle) vs. the "
                  "ground-truth annotation point can reasonably differ by less than half an arm "
                  "span (~0.9m) given players often have an arm/racket extended -- 0.8m is a "
                  "slightly conservative fraction of that.",
        "per_rally": per_rally_summary,
        "combined": {
            "n": int(len(all_errors)), "median_m": float(np.median(all_errors)),
            "mean_m": float(all_errors.mean()), "p90_m": float(np.percentile(all_errors, 90)),
            "max_m": float(all_errors.max()),
            "pct_within_tolerance": {f"{t}m": float(100 * (all_errors < t).mean()) for t in tolerances},
        },
    }
    return summary


if __name__ == "__main__":
    joints_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    csv_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    print(json.dumps(main(joints_dir, csv_dir), indent=2))
