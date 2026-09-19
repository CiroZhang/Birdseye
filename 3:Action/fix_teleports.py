"""Detect and fix single-frame joint 'teleports' in the demo rally's already
ground-corrected joints: VIMO's per-frame reconstruction is least reliable
for fast-moving extremities (wrist/hand during a swing, exactly at hit
frames), and one bad frame stretches a bone cylinder into a long visible
streak across the court in the viewer. Same failure class documented in
Project/METHODS.md ("Boundary-frame teleport-detection gap"), just never
guarded for this rally's joints specifically (only the feet got an outlier
guard, as part of the ground-correction pass).

For each joint independently: a real fast swing still traces a smooth path
frame to frame, so compare each frame's position against the straight-line
interpolation of its neighbors -- a single bad frame deviates sharply from
that local path (teleports out AND back), while genuine fast motion stays
consistent with its neighbors. Replace flagged frames with that interpolation.
"""
import json

import numpy as np

DEVIATION_THRESH = 0.5  # meters; a real per-frame human joint motion at 30fps
                          # basically never deviates this far from its neighbors'
                          # straight-line path, even during a hard smash swing


def fix_player(joints):
    joints = np.array(joints)  # (n_frames, 24, 3)
    n_frames = joints.shape[0]
    n_fixed = 0
    for j in range(24):
        pos = joints[:, j, :]
        valid = ~np.isnan(pos).any(axis=1)
        for f in range(1, n_frames - 1):
            if not (valid[f - 1] and valid[f] and valid[f + 1]):
                continue
            interp = (pos[f - 1] + pos[f + 1]) / 2
            deviation = np.linalg.norm(pos[f] - interp)
            if deviation > DEVIATION_THRESH:
                joints[f, j, :] = interp
                n_fixed += 1
    return joints.tolist(), n_fixed


demo = json.load(open("viz/demo_data.json"))
for side in ("top", "bottom"):
    fixed, n_fixed = fix_player(demo[side])
    demo[side] = fixed
    print(f"{side}: fixed {n_fixed} single-frame joint teleports")

json.dump(demo, open("viz/demo_data.json", "w"))
print("saved viz/demo_data.json")
