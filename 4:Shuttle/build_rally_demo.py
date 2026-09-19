"""General builder: produce the shuttle-demo JSON payload (player skeletons,
shuttle 3D trajectory, video clip) for any (match, game, rally) that has a
full chain of consecutive real hits (BFMD hit_inferred) each followed by
another real hit -- i.e. every shot in the rally gets a genuine two-anchor
physics fit, no extrapolated tail.

Reuses the same fitting method as run_shuttle_reconstruction.py / shuttle_physics.py
(no shortcuts here -- same court calibration, same self-detected player joints,
same two-anchor drag+gravity fit).
"""
import base64
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# smooth_joints.py lives in the sibling 3:Action folder, not here -- add it to the path.
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "3:Action")))

from run_shuttle_reconstruction import load_match, hand_position, find_rally_tag, FPS
from shuttle_physics import fit_shot, simulate
from smooth_joints import smooth_player

BASE = os.environ.get(
    "BFMD_DATA_ROOT",
    os.path.normpath(os.path.join(_HERE, "..", "..", "Dataset", "BFMD", "BFMD_data")),
)
VIDEO_DIR = f"{BASE}/videos"

BONES = [[1,0],[2,0],[3,0],[4,1],[5,2],[6,3],[7,4],[8,5],[9,6],[10,7],[11,8],
         [12,9],[13,9],[14,9],[15,12],[16,13],[17,14],[18,16],[19,17],
         [20,18],[21,19],[22,20],[23,21]]


def build(match_name, game, rally, out_tag, w_px=1280, h_px=720):
    K, R, t, joints_data, hits, shuttle_by_frame = load_match(match_name)
    rallies_meta = joints_data["rallies"]
    tag, start = find_rally_tag(rallies_meta, game, rally)
    assert tag is not None, f"rally g{game}r{rally} not found in {match_name}"

    rally_hits = sorted([h for h in hits if h["game"] == game and h["rally"] == rally],
                         key=lambda h: h["frame"])
    assert len(rally_hits) >= 3, f"only {len(rally_hits)} hits, need a real chain"

    f_first, f_last = rally_hits[0]["frame"], rally_hits[-1]["frame"]
    n_frames = f_last - f_first + 1
    print(f"{tag}: {len(rally_hits)} hits, frames {f_first}-{f_last} ({n_frames} frames)")

    shuttle_traj = np.full((n_frames, 3), np.nan)
    shots_meta = []
    reprojs = []
    for i in range(len(rally_hits) - 1):
        h0, h1 = rally_hits[i], rally_hits[i + 1]
        f0, f1 = h0["frame"], h1["frame"]
        if f1 <= f0 or f1 - f0 > 90:
            print(f"  SKIP shot {i}: bad gap {f0}-{f1}")
            continue
        local0, local1 = f0 - start, f1 - start
        hand0 = hand_position(joints_data, tag, h0["side"], local0)
        if hand0 is None:
            print(f"  SKIP shot {i}: no hand anchor")
            continue
        receiver_side = "bottom" if h0["side"] == "top" else "top"
        hand1 = hand_position(joints_data, tag, receiver_side, local1)

        obs_frames, obs_px = [], []
        for f in range(f0, f1 + 1):
            if f in shuttle_by_frame:
                obs_frames.append(f - f0)
                obs_px.append(shuttle_by_frame[f])
        if len(obs_frames) < 5:
            print(f"  SKIP shot {i}: only {len(obs_frames)} 2D obs")
            continue
        obs_frames = np.array(obs_frames, dtype=float)
        obs_px = np.array(obs_px, dtype=float)

        p0_fit, v0_fit, result, valid = fit_shot(
            obs_frames, obs_px, FPS, K, R, t, hand0, 0,
            end_anchor_3d=hand1, end_anchor_frame=(f1 - f0) if hand1 is not None else None,
        )
        if not valid:
            print(f"  SKIP shot {i}: invalid fit")
            continue
        reproj_rmse = float(np.sqrt(np.mean(result.fun[: len(obs_px) * 2] ** 2)) * np.sqrt(2))
        reprojs.append(reproj_rmse)

        n_sub = f1 - f0 + 1
        t_eval = np.arange(n_sub) / FPS
        traj = simulate(p0_fit, v0_fit, t_eval)
        gstart_local = f0 - f_first
        shuttle_traj[gstart_local:gstart_local + n_sub] = traj

        shots_meta.append({
            "frame": local0 if i == 0 else f0 - f_first,
            "side": h0["side"], "shot_type": h0["shot_type"],
            "speed_kmh": round(float(np.linalg.norm(v0_fit) * 3.6), 1),
            "reproj_rmse_px": round(reproj_rmse, 1),
        })
        print(f"  shot {i} {h0['side']} {h0['shot_type']}: speed={np.linalg.norm(v0_fit)*3.6:.1f}km/h reproj={reproj_rmse:.1f}px")

    # fix shots_meta frame indices properly (local to the clip, not the shot)
    for k, h in enumerate(rally_hits[:len(shots_meta)]):
        shots_meta[k]["frame"] = h["frame"] - f_first

    # trim any leading/trailing frames where no shot's fit covers the shuttle
    # (a skipped/invalid shot at the very start or end of the chain) -- keep
    # only the span with a real reconstructed trajectory throughout
    valid_mask = ~np.isnan(shuttle_traj).any(axis=1)
    valid_idx = np.where(valid_mask)[0]
    trim_lo, trim_hi = int(valid_idx[0]), int(valid_idx[-1]) + 1
    if trim_lo > 0 or trim_hi < n_frames:
        print(f"  trimming {trim_lo} leading / {n_frames - trim_hi} trailing frame(s) with no shuttle fit")
        shuttle_traj = shuttle_traj[trim_lo:trim_hi]
        shots_meta = [s for s in shots_meta if trim_lo <= s["frame"] < trim_hi]
        for s in shots_meta:
            s["frame"] -= trim_lo
        f_first_clip = f_first + trim_lo
        n_frames = trim_hi - trim_lo
    else:
        f_first_clip = f_first

    # player joints for this frame window, smoothed for display
    r = rallies_meta[tag]
    lo, hi = f_first_clip - start, f_first_clip - start + n_frames
    top = np.array(r["joints"]["top"])[lo:hi]
    bottom = np.array(r["joints"]["bottom"])[lo:hi]
    top_s = smooth_player(top)
    bottom_s = smooth_player(bottom)

    # video clip
    video_path = f"{VIDEO_DIR}/{match_name}.mp4"
    clip_path = f"/tmp/{out_tag}_clip.mp4"
    start_sec = f_first_clip / FPS
    dur_sec = n_frames / FPS
    subprocess.run([
        "ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", video_path,
        "-t", f"{dur_sec:.3f}", "-an", "-c:v", "libx264", "-crf", "28",
        "-preset", "fast", "-vf", "scale=640:360", clip_path,
    ], check=True, capture_output=True)
    video_b64 = base64.b64encode(open(clip_path, "rb").read()).decode("ascii")

    median_reproj = round(float(np.median(reprojs)), 1) if reprojs else None
    data = {
        "bones": BONES,
        "top": top_s.tolist(),
        "bottom": bottom_s.tolist(),
        "shuttle": shuttle_traj.tolist(),
        "n_frames": n_frames,
        "video_b64": video_b64,
        "shots": shots_meta,
        "median_reproj_px": median_reproj,
    }
    out_path = f"viz/{out_tag}_data.json"
    json.dump(data, open(out_path, "w"))
    print(f"saved {out_path} ({len(shots_meta)} shots, median reproj {median_reproj}px, n_frames={n_frames})")
    return data


if __name__ == "__main__":
    match_name, game, rally, out_tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    build(match_name, game, rally, out_tag)
