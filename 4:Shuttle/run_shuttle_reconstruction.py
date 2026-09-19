"""Driver: reconstruct 3D shuttle trajectories per shot for one BFMD match,
using our own self-detected camera + self-detected player reconstruction
(real 3D hand anchors) + BFMD's own 2D shuttle annotations.

A "shot" = the interval from one real hit (BFMD's hit_inferred) to the next
hit in the same rally (or to the rally's end, for the last shot -- dropped,
no second anchor / uncertain landing).
"""
import json
import os
import sys

import numpy as np

from shuttle_physics import fit_shot, project

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get(
    "BFMD_DATA_ROOT",
    os.path.normpath(os.path.join(_HERE, "..", "..", "Dataset", "BFMD", "BFMD_data")),
)
# joints_selfdetected.json lives nested as SD_DIR/<match_name>/joints_selfdetected.json (real
# Killarney /scratch/ciro/bfmd_all layout) -- ephemeral scratch, no durable copy in /project yet.
SD_DIR = os.environ.get("BFMD_SELFDETECTED_ROOT", "/scratch/ciro/bfmd_all")
FPS = 30.0

# SMPL 24-joint hand indices
HAND = {"L": 22, "R": 23}


def load_match(match_name):
    cam = json.load(open(f"{BASE}/all_cameras_self_detected.json"))[match_name]
    K, R, t = np.array(cam["K"]), np.array(cam["R"]), np.array(cam["t"])

    joints_data = json.load(open(f"{SD_DIR}/{match_name}/joints_selfdetected.json"))
    hits = json.load(open(f"{BASE}/annotations/hit_inferred/{match_name}.json"))["hits"]

    shuttle_ann = json.load(open(f"{BASE}/annotations/shuttle/{match_name}.json"))
    seq = shuttle_ann["predictions"][0]["result"][0]["value"]["sequence"]
    W, H = 1280, 720
    shuttle_by_frame = {}
    for pt in seq:
        if not pt.get("enabled", True):
            continue
        cx_pct = pt["x"] + pt["width"] / 2.0
        cy_pct = pt["y"] + pt["height"] / 2.0
        shuttle_by_frame[pt["frame"]] = (cx_pct / 100.0 * W, cy_pct / 100.0 * H)

    return K, R, t, joints_data, hits, shuttle_by_frame


def hand_position(joints_data, tag, side, local_frame):
    """Pick whichever hand (L=22, R=23) has the higher speed at this frame
    -- the racket hand during a real swing moves much faster than the idle
    hand."""
    r = joints_data["rallies"][tag]
    if side not in r["joints"]:
        return None
    arr = np.array(r["joints"][side])  # (n_frames, 24, 3)
    n = arr.shape[0]
    if not (0 <= local_frame < n):
        return None
    speeds = {}
    for name, idx in HAND.items():
        lo, hi = max(0, local_frame - 2), min(n - 1, local_frame + 2)
        if hi <= lo or np.isnan(arr[lo, idx]).any() or np.isnan(arr[hi, idx]).any():
            continue
        speeds[name] = np.linalg.norm(arr[hi, idx] - arr[lo, idx]) / ((hi - lo) / FPS)
    if not speeds:
        return None
    best = max(speeds, key=speeds.get)
    pos = arr[local_frame, HAND[best]]
    return None if np.isnan(pos).any() else pos


def find_rally_tag(rallies_meta, game, rally):
    for tag, r in rallies_meta.items():
        g = int(tag.split("_")[0][1:])
        ridx = int(tag.split("_")[1][1:])
        if g == game and ridx == rally:
            return tag, r["start"]
    return None, None


def main(match_name, max_shots=None):
    K, R, t, joints_data, hits, shuttle_by_frame = load_match(match_name)
    rallies_meta = joints_data["rallies"]

    hits_by_rally = {}
    for h in hits:
        hits_by_rally.setdefault((h["game"], h["rally"]), []).append(h)
    for key in hits_by_rally:
        hits_by_rally[key].sort(key=lambda h: h["frame"])

    results = []
    n_done = 0
    for (game, rally), rally_hits in hits_by_rally.items():
        tag, start = find_rally_tag(rallies_meta, game, rally)
        if tag is None or "top" not in rallies_meta[tag]["joints"] or "bottom" not in rallies_meta[tag]["joints"]:
            continue
        for i in range(len(rally_hits) - 1):
            h0, h1 = rally_hits[i], rally_hits[i + 1]
            f0, f1 = h0["frame"], h1["frame"]
            if f1 <= f0 or f1 - f0 > 90:  # skip absurdly long/short/bad gaps
                continue
            local0 = f0 - start
            local1 = f1 - start
            hand0 = hand_position(joints_data, tag, h0["side"], local0)
            if hand0 is None:
                continue
            # the receiver is the OTHER side from whoever just hit
            receiver_side = "bottom" if h0["side"] == "top" else "top"
            hand1 = hand_position(joints_data, tag, receiver_side, local1)

            obs_frames, obs_px = [], []
            for f in range(f0, f1 + 1):
                if f in shuttle_by_frame:
                    obs_frames.append(f - f0)
                    obs_px.append(shuttle_by_frame[f])
            if len(obs_frames) < 5:
                continue
            obs_frames = np.array(obs_frames, dtype=float)
            obs_px = np.array(obs_px, dtype=float)

            try:
                p0_fit, v0_fit, result, valid = fit_shot(
                    obs_frames, obs_px, FPS, K, R, t, hand0, 0,
                    end_anchor_3d=hand1, end_anchor_frame=(f1 - f0) if hand1 is not None else None,
                )
            except Exception:
                continue
            if not valid:
                continue
            reproj_rmse = np.sqrt(np.mean(result.fun[: len(obs_px) * 2] ** 2)) * np.sqrt(2)
            results.append({
                "game": game, "rally": rally, "frame_start": f0, "frame_end": f1,
                "side": h0["side"], "shot_type": h0["shot_type"], "n_obs": len(obs_frames),
                "p0": p0_fit.tolist(), "v0": v0_fit.tolist(),
                "speed_kmh": float(np.linalg.norm(v0_fit) * 3.6),
                "reproj_rmse_px": float(reproj_rmse),
                "has_end_anchor": hand1 is not None,
            })
            n_done += 1
            if max_shots and n_done >= max_shots:
                return results
    return results


if __name__ == "__main__":
    match_name = sys.argv[1]
    max_shots = int(sys.argv[2]) if len(sys.argv) > 2 else None
    results = main(match_name, max_shots)
    print(f"{match_name}: fit {len(results)} shots")
    for r in results[:20]:
        anchor_tag = "2-anchor" if r["has_end_anchor"] else "1-anchor"
        print(f"  g{r['game']}r{r['rally']} f{r['frame_start']}-{r['frame_end']} {r['side']} {r['shot_type']} [{anchor_tag}]: "
              f"speed={r['speed_kmh']:.1f}km/h reproj_rmse={r['reproj_rmse_px']:.1f}px n_obs={r['n_obs']}")
    if results:
        rmses = [r["reproj_rmse_px"] for r in results]
        speeds = [r["speed_kmh"] for r in results]
        print(f"\nmean reproj RMSE: {np.mean(rmses):.1f}px, median: {np.median(rmses):.1f}px")
        print(f"speed range: {np.min(speeds):.0f}-{np.max(speeds):.0f} km/h, median: {np.median(speeds):.0f} km/h")
