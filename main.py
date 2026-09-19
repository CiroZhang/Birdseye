"""End-to-end pipeline entry point: links all four sections together for one
BFMD match and writes the result to result/.

    1:Court  -- self-detected camera (all_cameras_self_detected.json)
    2:Player -- self-detected 3D player joints (BFMD_data_selfdetected/*_joints_selfdetected.json)
    3:Action -- OUR TRAINED MODEL predicts hits from those joints (checkpoints/hit_model_final_transformer.pt)
                -- NOTE: every other evaluation in this project (the reported 92.7% F1, the shuttle
                   reconstruction report) uses BFMD's own hit_inferred labels as the shot-boundary
                   ground truth. This script is the one place the model's OWN predicted hits (not
                   the ground truth) are fed forward into the next stage -- a genuine full-pipeline
                   test, not a component-by-component re-validation against ground truth.
    4:Shuttle -- reconstructs a 3D trajectory for each shot bounded by two CONSECUTIVE PREDICTED
                hits (not ground-truth hits), using the self-detected hand positions as anchors.

Known limitation, stated plainly: the shuttle stage still needs a 2D shuttle pixel track as its
observation signal, and this project never built its own shuttle detector -- it uses BFMD's own
shuttle annotation for that one input. Everything else in this script (camera, player pose, hit
timing/type) is entirely our own detection, with zero BFMD annotations upstream.

Usage: python3 main.py [match_name]  (default: the demo match used throughout this project)
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_training_data_selfdetected import canonicalize, velocity
from hit_model_transformer import HitSpotterTransformer
from shuttle_physics import fit_shot

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{BASE}/BFMD_data"
SD_DIR = f"{BASE}/BFMD_data_selfdetected"
CKPT = f"{BASE}/checkpoints/hit_model_final_transformer.pt"
RESULT_DIR = f"{BASE}/result"
FPS = 30.0
TOLERANCE_FRAMES = 10
HIT_THRESHOLD = 0.5  # the threshold that gave the reported 92.7% F1

SHOT_VOCAB = ['block', 'clear', 'drive', 'drop', 'flick_serve', 'lift', 'net_kill',
              'net_shot', 'press', 'push', 'serve', 'smash']
SHOT_IDX = {s: i for i, s in enumerate(SHOT_VOCAB)}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LR_SWAP = {1: 2, 2: 1, 4: 5, 5: 4, 7: 8, 8: 7, 10: 11, 11: 10, 13: 14, 14: 13,
           16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20, 22: 23, 23: 22}
LR_PERM = [LR_SWAP.get(i, i) for i in range(24)]
HAND = {"L": 22, "R": 23}


def mirror_lr(rel):
    m = rel[:, LR_PERM, :].copy()
    m[..., 0] *= -1
    return m


def make_features(self_rel, self_vel, self_acc, other_rel, other_vel):
    T = self_rel.shape[0]
    return np.concatenate([
        self_rel.reshape(T, -1), self_vel.reshape(T, -1), self_acc.reshape(T, -1),
        other_rel.reshape(T, -1), other_vel.reshape(T, -1),
    ], axis=1)


def accel(vel):
    a = np.zeros_like(vel)
    a[1:] = vel[1:] - vel[:-1]
    return a


def load_model():
    model = HitSpotterTransformer(in_dim=360, d_model=192, n_heads=6, n_layers=4,
                                    n_shot_types=len(SHOT_VOCAB)).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    return model


def predict_hits_for_rally(model, top, bottom):
    """top/bottom: (n_frames, 24, 3) self-detected joints (may contain NaN).
    Returns a list of {local_frame, side, shot_type, prob} sorted by frame,
    using the model's OWN predictions (peak-detection on hit probability),
    mirror test-time ensemble, same recipe as the reported evaluation."""
    top_rel = canonicalize(np.nan_to_num(top), "top")
    bot_rel = canonicalize(np.nan_to_num(bottom), "bottom")
    top_vel, bot_vel = velocity(top_rel), velocity(bot_rel)
    top_acc, bot_acc = accel(top_vel), accel(bot_vel)

    hits = []
    for self_rel, self_vel, self_acc, other_rel, other_vel, side in [
        (top_rel, top_vel, top_acc, bot_rel, bot_vel, "top"),
        (bot_rel, bot_vel, bot_acc, top_rel, top_vel, "bottom"),
    ]:
        feats = make_features(self_rel, self_vel, self_acc, other_rel, other_vel)
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            hl, sl = model(x)
        probs = torch.sigmoid(hl)[0].cpu().numpy()
        shot_pred = sl[0].argmax(dim=-1).cpu().numpy()

        m_feats = make_features(mirror_lr(self_rel), mirror_lr(self_vel), mirror_lr(self_acc),
                                 mirror_lr(other_rel), mirror_lr(other_vel))
        xm = torch.tensor(m_feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            hlm, _ = model(xm)
        probs = (probs + torch.sigmoid(hlm)[0].cpu().numpy()) / 2

        peaks, _ = find_peaks(probs, height=HIT_THRESHOLD, distance=TOLERANCE_FRAMES)
        for p in peaks:
            hits.append({"local_frame": int(p), "side": side,
                         "shot_type": SHOT_VOCAB[shot_pred[p]], "prob": float(probs[p])})
    hits.sort(key=lambda h: h["local_frame"])
    return hits


def hand_position(rallies_joints, tag, side, local_frame):
    if side not in rallies_joints[tag]["joints"]:
        return None
    arr = np.array(rallies_joints[tag]["joints"][side])
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
    pos = arr[local_frame, HAND[max(speeds, key=speeds.get)]]
    return None if np.isnan(pos).any() else pos


def load_shuttle_2d(match_name):
    """BFMD's own shuttle 2D annotation -- see module docstring, this is the
    one input this project never replaced with its own detector."""
    ann = json.load(open(f"{DATA}/annotations/shuttle/{match_name}.json"))
    seq = ann["predictions"][0]["result"][0]["value"]["sequence"]
    W, H = 1280, 720
    by_frame = {}
    for pt in seq:
        if not pt.get("enabled", True):
            continue
        cx = (pt["x"] + pt["width"] / 2.0) / 100.0 * W
        cy = (pt["y"] + pt["height"] / 2.0) / 100.0 * H
        by_frame[pt["frame"]] = (cx, cy)
    return by_frame


def run_pipeline(match_name):
    print(f"=== running full pipeline on {match_name} ===")

    print("[1:Court] loading self-detected camera...")
    cam = json.load(open(f"{DATA}/all_cameras_self_detected.json"))[match_name]
    K, R, t = np.array(cam["K"]), np.array(cam["R"]), np.array(cam["t"])

    print("[2:Player] loading self-detected player reconstruction...")
    joints_data = json.load(open(f"{SD_DIR}/{match_name}_joints_selfdetected.json"))
    rallies = joints_data["rallies"]

    print("[3:Action] loading trained hit-detection model + predicting hits per rally...")
    model = load_model()
    true_hits = json.load(open(f"{DATA}/annotations/hit_inferred/{match_name}.json"))["hits"]
    true_by_rally = {}
    for h in true_hits:
        true_by_rally.setdefault((h["game"], h["rally"]), []).append(h)

    rally_results = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    for tag, r in rallies.items():
        if "top" not in r["joints"] or "bottom" not in r["joints"]:
            continue
        top = np.array(r["joints"]["top"])
        bottom = np.array(r["joints"]["bottom"])
        if len(top) < 5:
            continue
        pred_hits = predict_hits_for_rally(model, top, bottom)

        g = int(tag.split("_")[0][1:])
        ridx = int(tag.split("_")[1][1:])
        true_this_rally = sorted(true_by_rally.get((g, ridx), []), key=lambda h: h["frame"])
        true_local = [{"local_frame": h["frame"] - r["start"], "side": h["side"], "shot_type": h["shot_type"]}
                       for h in true_this_rally]
        matched = set()
        tp = 0
        for ph in pred_hits:
            cands = [i for i, th in enumerate(true_local)
                     if i not in matched and th["side"] == ph["side"]
                     and abs(th["local_frame"] - ph["local_frame"]) <= TOLERANCE_FRAMES]
            if cands:
                best = min(cands, key=lambda i: abs(true_local[i]["local_frame"] - ph["local_frame"]))
                matched.add(best)
                tp += 1
        fp = len(pred_hits) - tp
        fn = len(true_local) - len(matched)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        rally_results[tag] = {
            "game": g, "rally": ridx, "start_frame": r["start"], "n_frames": len(top),
            "predicted_hits": pred_hits, "n_true_hits": len(true_local),
            "tp": tp, "fp": fp, "fn": fn,
        }

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"  hit detection on this match: P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
          f"(tp={total_tp} fp={total_fp} fn={total_fn})")

    print("[4:Shuttle] reconstructing 3D trajectory per shot, using the model's OWN predicted hits...")
    shuttle_2d = load_shuttle_2d(match_name)
    shot_results = []
    n_attempted, n_valid = 0, 0
    for tag, rr in rally_results.items():
        pred_hits = rr["predicted_hits"]
        start = rr["start_frame"]
        for i in range(len(pred_hits) - 1):
            h0, h1 = pred_hits[i], pred_hits[i + 1]
            # NOTE: named fr0/fr1 (not f0/f1) deliberately -- this function
            # also has an outer `f1` holding the hit-detection F1 score;
            # Python for-loops aren't block-scoped, so reusing f0/f1 here
            # silently clobbered that score (a real bug caught by testing:
            # the written result briefly reported F1=133755 instead of 0.94).
            fr0, fr1 = h0["local_frame"] + start, h1["local_frame"] + start
            if fr1 <= fr0 or fr1 - fr0 > 90:
                continue
            n_attempted += 1
            hand0 = hand_position(rallies, tag, h0["side"], h0["local_frame"])
            if hand0 is None:
                continue
            receiver_side = "bottom" if h0["side"] == "top" else "top"
            hand1 = hand_position(rallies, tag, receiver_side, h1["local_frame"])

            obs_frames, obs_px = [], []
            for fr in range(fr0, fr1 + 1):
                if fr in shuttle_2d:
                    obs_frames.append(fr - fr0)
                    obs_px.append(shuttle_2d[fr])
            if len(obs_frames) < 5:
                continue
            try:
                p0_fit, v0_fit, result, valid = fit_shot(
                    np.array(obs_frames, dtype=float), np.array(obs_px, dtype=float), FPS, K, R, t,
                    hand0, 0, end_anchor_3d=hand1,
                    end_anchor_frame=(fr1 - fr0) if hand1 is not None else None,
                )
            except Exception:
                continue
            if not valid:
                continue
            n_valid += 1
            reproj_rmse = float(np.sqrt(np.mean(result.fun[: len(obs_px) * 2] ** 2)) * np.sqrt(2))
            shot_results.append({
                "tag": tag, "frame_start": fr0, "frame_end": fr1, "side": h0["side"],
                "shot_type": h0["shot_type"], "speed_kmh": float(np.linalg.norm(v0_fit) * 3.6),
                "reproj_rmse_px": reproj_rmse, "has_end_anchor": hand1 is not None,
            })

    print(f"  shuttle: {n_valid}/{n_attempted} shots fit successfully (both stages chained from "
          f"the model's own predicted hits, not ground truth)")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out = {
        "match_name": match_name,
        "hit_detection": {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "n_rallies": len(rally_results),
        },
        "shuttle_reconstruction": {
            "n_shots_attempted": n_attempted, "n_shots_valid": n_valid,
            "median_speed_kmh": float(np.median([s["speed_kmh"] for s in shot_results])) if shot_results else None,
            "median_reproj_px": float(np.median([s["reproj_rmse_px"] for s in shot_results])) if shot_results else None,
        },
        "rallies": rally_results,
        "shots": shot_results,
    }
    out_path = f"{RESULT_DIR}/{match_name}_pipeline_result.json"
    json.dump(out, open(out_path, "w"))
    print(f"\nsaved {out_path}")

    summary_path = f"{RESULT_DIR}/{match_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"FULL PIPELINE RESULT -- {match_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write("1:Court  -- self-detected camera (loaded from all_cameras_self_detected.json)\n")
        f.write("2:Player -- self-detected 3D player reconstruction (loaded)\n")
        f.write(f"3:Action -- trained transformer predicts hits on {len(rally_results)} rallies\n")
        f.write(f"           P={precision:.3f} R={recall:.3f} F1={f1:.3f} vs BFMD's real hit labels "
                f"(tp={total_tp} fp={total_fp} fn={total_fn})\n")
        f.write(f"4:Shuttle -- 3D trajectory fit per shot, using the model's OWN predicted hit pairs "
                f"as shot boundaries (not ground truth)\n")
        f.write(f"           {n_valid}/{n_attempted} shots fit successfully\n")
        if shot_results:
            f.write(f"           median speed: {out['shuttle_reconstruction']['median_speed_kmh']:.1f} km/h, "
                    f"median reprojection: {out['shuttle_reconstruction']['median_reproj_px']:.1f}px\n")
        f.write("\nKnown limitation: the shuttle stage's 2D observation still comes from BFMD's own\n")
        f.write("shuttle annotation -- this project never built its own shuttle 2D detector. Every\n")
        f.write("other stage (camera, player pose, hit timing/type) is entirely our own detection.\n")
    print(f"saved {summary_path}")
    return out


if __name__ == "__main__":
    match = sys.argv[1] if len(sys.argv) > 1 else \
        "KAPAL-API-Indonesia-Open-2025-Shi-Yu-Qi-CHN-1-vs.-Anders-Antonsen-DEN-3-SF"
    run_pipeline(match)
