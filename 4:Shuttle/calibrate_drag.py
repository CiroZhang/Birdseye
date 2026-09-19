"""Calibrate the shuttlecock drag constant against real data instead of
trusting a literature terminal-velocity estimate: for each candidate k,
refit a sample of well-observed shots and measure reprojection error.
Pick the k that minimizes error, but sanity-check the resulting speed
distribution stays physically plausible (a value that fits noise better
by drifting to an unphysical regime would be a red flag, not a win)."""
import json

import numpy as np
from scipy.optimize import least_squares

import shuttle_physics as sp

FPS = 30.0
sample = json.load(open("shuttle_calib_sample.json"))
rng = np.random.default_rng(0)
idx = rng.choice(len(sample), size=min(150, len(sample)), replace=False)
sample = [sample[i] for i in idx]
print(f"calibrating on {len(sample)} shots")


def fit_one(shot, k):
    K, R, t = np.array(shot["K"]), np.array(shot["R"]), np.array(shot["t"])
    hand0, hand1 = np.array(shot["hand0"]), np.array(shot["hand1"])
    obs_frames = np.array(shot["obs_frames"])
    obs_px = np.array(shot["obs_px"])
    t_obs = obs_frames / FPS
    end_t = shot["f1_f0"] / FPS

    dt = max(end_t, 1e-3)
    v0_guess = (hand1 - hand0) / dt

    def residuals(params):
        p0, v0 = params[:3], params[3:6]
        anchor_res = (p0 - hand0) / 0.15
        sim_times = np.sort(np.unique(np.concatenate([[0.0, end_t], t_obs])))
        try:
            traj = sp.simulate(p0, v0, sim_times, k=k)
        except Exception:
            return np.full(len(obs_px) * 2 + 6, 1e3)
        traj_at_obs = np.array([traj[np.searchsorted(sim_times, tt)] for tt in t_obs])
        proj_px = sp.project(K, R, t, traj_at_obs)
        reproj_res = (proj_px - obs_px).ravel()
        end_pos = traj[np.searchsorted(sim_times, end_t)]
        end_res = (end_pos - hand1) / 0.15
        return np.concatenate([reproj_res, anchor_res, end_res])

    x0 = np.concatenate([hand0, v0_guess])
    result = least_squares(residuals, x0, method="lm", max_nfev=3000)
    v0_fit = result.x[3:6]
    speed = np.linalg.norm(v0_fit)
    reproj_rmse = np.sqrt(np.mean(result.fun[: len(obs_px) * 2] ** 2)) * np.sqrt(2)
    valid = np.all(np.isfinite(result.x)) and speed < sp.MAX_PLAUSIBLE_SPEED_MS
    return reproj_rmse, speed, valid


# k = g / v_terminal^2 -> sweep v_terminal from 5 to 12 m/s (a wide, physically
# reasonable band -- real shuttles cluster around 6-8 m/s depending on type)
v_terminal_grid = np.arange(4.0, 13.0, 0.5)
k_grid = 9.81 / v_terminal_grid**2

results_by_k = []
for k, vt in zip(k_grid, v_terminal_grid):
    rmses, speeds, n_valid = [], [], 0
    for shot in sample:
        rmse, speed, valid = fit_one(shot, k)
        if valid:
            rmses.append(rmse)
            speeds.append(speed * 3.6)
            n_valid += 1
    median_rmse = np.median(rmses) if rmses else np.inf
    median_speed = np.median(speeds) if speeds else np.nan
    results_by_k.append((k, vt, median_rmse, median_speed, n_valid))
    print(f"v_terminal={vt:.1f}m/s (k={k:.4f}): median reproj={median_rmse:.2f}px, "
          f"median speed={median_speed:.0f}km/h, n_valid={n_valid}/{len(sample)}", flush=True)

best = min(results_by_k, key=lambda r: r[2])
print(f"\nBEST: v_terminal={best[1]:.1f}m/s (k={best[0]:.4f}), median reproj={best[2]:.2f}px, median speed={best[3]:.0f}km/h")
print(f"(previous literature-guess k=0.20 / v_terminal=7.0m/s, for comparison)")
