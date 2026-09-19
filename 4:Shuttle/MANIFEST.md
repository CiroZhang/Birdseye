# 4:Shuttle -- file guide

Read SHUTTLE_RECONSTRUCTION_REPORT.txt first for the full writeup (goal, why this is a fresh build
-- the original pre-session code couldn't be found anywhere -- method, results, honest limitations).
See readme.md section 4 for the up-to-date summary + demo link.

## Core method
- **shuttle_physics.py** -- the physics model: `shuttle_ode`/`simulate` (point mass under gravity +
  quadratic drag, `DRAG_K` empirically calibrated to v_terminal=6.5m/s, integrated via
  `scipy.integrate.solve_ivp`), `project` (camera reprojection), `fit_shot` (the core per-shot
  nonlinear-least-squares fit -- anchored by TWO real 3D points, the hitting player's hand at the
  hit frame AND the receiving player's hand at the next hit, both from `2:Player`'s self-detected
  VIMO reconstruction; single-anchor fits produced physically impossible speeds before this fix).
  `MAX_PLAUSIBLE_SPEED_MS=170.0` flags divergent fits as invalid rather than reporting nonsense.
- **calibrate_drag.py** -- swept v_terminal 4.0-12.5 m/s on 150 real well-observed shots
  (from `shuttle_calib_sample.json`, 764 shots gathered across 4 matches) to find the drag constant
  empirically instead of only trusting the literature guess. Found v_terminal=6.5m/s (median reproj
  9.69px vs 10.00px for the 7.0m/s literature guess). Log: calibrate_drag.log.
- **run_shuttle_reconstruction.py** -- per-match driver (`load_match`, `hand_position`,
  `find_rally_tag`, `main`): segments each rally into shots via consecutive real hits
  (`hit_inferred`), fits each with `fit_shot`.
- **run_shuttle_all.py** -- batch driver over all 12 matches. Log: shuttle_all.log. Output:
  **shuttle_reconstruction_all.json** (9,594 raw shot fits, before the post-hoc plausibility filter
  discussed in the report -- 4.8% excluded above a realistic 300km/h ceiling for organic rally play).
- **build_rally_demo.py** -- builds the per-rally JSON payload (player skeletons + smoothed display,
  reconstructed 3D shuttle trajectory, embedded video clip) used by the published demo. Reusable for
  any (match, game, rally) with a full chain of consecutive real hits -- validated by rebuilding the
  original demo rally byte-for-byte identical before using it for 3 new ones.

## Results (see readme.md for the full table + honest limitations)
9,135 shots after filtering: median reprojection 11.9px, median speed 69km/h. Median speed by shot
type came out in the exact real physical order (smash 183 > ... > net_shot 42 km/h) without the
fitting method ever being told what the labels mean -- the strongest available validation signal,
since BFMD has no 3D shuttle ground truth to check against directly.

**Demo**: https://claude.ai/code/artifact/d35999d8-aee6-41d1-9e23-ebfa907c3604 -- rally selector
across 4 rallies from 3 different matches.
