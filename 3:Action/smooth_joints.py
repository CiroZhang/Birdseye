"""Temporal smoothing pass for the demo rally's joint data -- purely a
display-quality fix, not a change to the model or its predictions (those
were already computed on the raw reconstruction). VIMO reconstructs each
frame close to independently, so even a correctly-tracked player has real
per-frame position noise ("shake") that a human skeleton never actually
does frame to frame. A Savitzky-Golay filter smooths that high-frequency
noise while preserving the real underlying motion trend (a swing's actual
speed/shape survives; the frame-to-frame vibration on top of it doesn't).
"""
import json

import numpy as np
from scipy.signal import savgol_filter

WINDOW = 9   # ~0.3s at 30fps
POLYORDER = 2


def smooth_player(joints):
    joints = np.array(joints)  # (n_frames, 24, 3)
    n_frames = joints.shape[0]
    out = joints.copy()
    for j in range(24):
        for c in range(3):
            series = joints[:, j, c]
            valid = ~np.isnan(series)
            if valid.sum() < WINDOW:
                continue
            idx = np.arange(n_frames)
            # interpolate any small gaps so savgol has a continuous series
            filled = np.interp(idx, idx[valid], series[valid])
            smoothed = savgol_filter(filled, window_length=WINDOW, polyorder=POLYORDER)
            out[:, j, c] = np.where(valid, smoothed, np.nan)
    return out


def jitter_metric(joints):
    """mean frame-to-frame acceleration magnitude (2nd derivative) across all
    joints -- real motion has smooth-ish 1st derivative; jitter shows up as
    noisy 2nd derivative. Lower = smoother."""
    joints = np.array(joints)
    vel = np.diff(joints, axis=0)
    acc = np.diff(vel, axis=0)
    mag = np.linalg.norm(acc, axis=2)
    return np.nanmean(mag)


demo = json.load(open("viz/demo_data.json"))
for side in ("top", "bottom"):
    before = jitter_metric(demo[side])
    smoothed = smooth_player(demo[side])
    after = jitter_metric(smoothed)
    demo[side] = smoothed.tolist()
    print(f"{side}: jitter (mean accel mag) {before:.4f} -> {after:.4f}")

json.dump(demo, open("viz/demo_data.json", "w"))
print("saved viz/demo_data.json")
