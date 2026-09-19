"""Same alignment logic as build_training_data_all.py, but reading the new
self-detected reconstruction (our own court detection + our own player
detection + VIMO + real ground correction, joints_selfdetected.json) instead
of the annotation-based one (*_joints.json). This produces an honestly
end-to-end training set: no BFMD annotations anywhere upstream of the hit
labels themselves (which remain real supervision, not a shortcut).
"""
import glob
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get(
    "BFMD_DATA_ROOT",
    os.path.normpath(os.path.join(_HERE, "..", "..", "Dataset", "BFMD", "BFMD_data")),
)
# NOTE: the self-detected joints (<match>/joints_selfdetected.json, nested per-match) currently
# live only on Killarney's /scratch (ephemeral, subject to purge) -- there is no durable copy in
# /project yet. Override via env var, or copy the data into /project and point this at that copy.
SD_DIR = os.environ.get("BFMD_SELFDETECTED_ROOT", "/scratch/ciro/bfmd_all")
FPS = 30.0
WINDOW_SEC = 1.0 / 3.0
PELVIS = 0


def canonicalize(joints, side):
    root = joints[:, PELVIS:PELVIS + 1, :]
    rel = joints - root
    if side == "top":
        rel = rel.copy()
        rel[..., 0] *= -1
        rel[..., 1] *= -1
    return rel


def velocity(rel_joints):
    v = np.zeros_like(rel_joints)
    v[1:] = rel_joints[1:] - rel_joints[:-1]
    return v


def find_completed_matches():
    """joints_selfdetected.json lives nested as SD_DIR/<match_name>/joints_selfdetected.json
    (real Killarney /scratch/ciro/bfmd_all layout) -- not a flat SD_DIR/<match>_joints_....json
    naming, which never actually existed as a persistent directory anywhere."""
    out = {}
    for path in glob.glob(f"{SD_DIR}/*/joints_selfdetected.json"):
        match_name = os.path.basename(os.path.dirname(path))
        out[match_name] = path
    return out


def build():
    matches = find_completed_matches()
    print(f"found {len(matches)} completed self-detected matches: {list(matches.keys())}")

    dataset = {"rallies": {}, "shot_type_vocab": set(), "match_of_rally": {}}
    all_shot_types = set()

    for match_name, joints_path in matches.items():
        joints_data = json.load(open(joints_path))
        hits = json.load(open(f"{BASE}/annotations/hit_inferred/{match_name}.json"))["hits"]

        rally_meta = joints_data["rallies"]
        tag_by_game_rally = {}
        for tag, r in rally_meta.items():
            g = int(tag.split("_")[0][1:])
            ridx = int(tag.split("_")[1][1:])
            tag_by_game_rally[(g, ridx)] = (tag, r["start"])

        samples = []
        for h in hits:
            key = (h["game"], h["rally"])
            if key not in tag_by_game_rally:
                continue
            tag, start = tag_by_game_rally[key]
            r = rally_meta[tag]
            if "top" not in r["joints"] or "bottom" not in r["joints"]:
                continue
            local_frame = h["frame"] - start
            n_local_frames = len(r["joints"]["top"])
            if not (0 <= local_frame < n_local_frames):
                continue
            samples.append({"tag": tag, "local_frame": local_frame, "side": h["side"], "shot_type": h["shot_type"]})
            all_shot_types.add(h["shot_type"])

        hits_by_tag = {}
        for s in samples:
            hits_by_tag.setdefault(s["tag"], []).append(s)

        win_frames = WINDOW_SEC * FPS
        n_rallies_added = 0
        for tag, r in rally_meta.items():
            if "top" not in r["joints"] or "bottom" not in r["joints"]:
                continue
            top = np.array(r["joints"]["top"], dtype=float)
            bot = np.array(r["joints"]["bottom"], dtype=float)
            n = len(top)
            if n < 5:
                continue

            top_rel = canonicalize(top, "top")
            bot_rel = canonicalize(bot, "bottom")
            top_vel = velocity(top_rel)
            bot_vel = velocity(bot_rel)

            target_top = np.zeros(n)
            target_bottom = np.zeros(n)
            shot_type_top = ["none"] * n
            shot_type_bottom = ["none"] * n
            for s in hits_by_tag.get(tag, []):
                f0 = s["local_frame"]
                target = target_top if s["side"] == "top" else target_bottom
                shot_arr = shot_type_top if s["side"] == "top" else shot_type_bottom
                lo, hi = max(0, int(f0 - win_frames)), min(n, int(f0 + win_frames) + 1)
                for f in range(lo, hi):
                    w = max(0.0, 1.0 - abs(f - f0) / win_frames)
                    target[f] = max(target[f], w)
                if 0 <= f0 < n:
                    shot_arr[f0] = s["shot_type"]

            unique_tag = f"{match_name}::{tag}"
            dataset["rallies"][unique_tag] = {
                "n_frames": n,
                "top_rel": top_rel.tolist(), "bottom_rel": bot_rel.tolist(),
                "top_vel": top_vel.tolist(), "bottom_vel": bot_vel.tolist(),
                "target_top": target_top.tolist(), "target_bottom": target_bottom.tolist(),
                "shot_type_top": shot_type_top, "shot_type_bottom": shot_type_bottom,
                "nan_frac_top": float(np.isnan(top[:, 0, 0]).mean()),
                "nan_frac_bottom": float(np.isnan(bot[:, 0, 0]).mean()),
            }
            dataset["match_of_rally"][unique_tag] = match_name
            n_rallies_added += 1

        n_hit_matched = len(samples)
        print(f"  {match_name}: {n_rallies_added} rallies, {n_hit_matched}/{len(hits)} hits matched")

    dataset["shot_type_vocab"] = sorted(all_shot_types)
    out_path = f"{BASE}/training_dataset_all_selfdetected.json"
    json.dump(dataset, open(out_path, "w"))
    n_frames_total = sum(r["n_frames"] for r in dataset["rallies"].values())
    print(f"\nsaved {out_path}: {len(dataset['rallies'])} rallies across {len(matches)} matches, "
          f"{n_frames_total} total frames, {len(dataset['shot_type_vocab'])} shot types")


if __name__ == "__main__":
    build()
