import json
import numpy as np
from run_shuttle_reconstruction import main

matches = json.load(open("BFMD_data/all_cameras_self_detected.json")).keys()
all_results = {}
for m in matches:
    try:
        res = main(m)
    except Exception as e:
        print(f"{m}: FAILED ({e})")
        continue
    all_results[m] = res
    rmses = [r["reproj_rmse_px"] for r in res]
    speeds = [r["speed_kmh"] for r in res]
    n_2anchor = sum(1 for r in res if r["has_end_anchor"])
    print(f"{m}: {len(res)} shots fit ({n_2anchor} with 2 anchors), "
          f"median reproj={np.median(rmses):.1f}px, median speed={np.median(speeds):.0f}km/h, "
          f"speed range={np.min(speeds):.0f}-{np.max(speeds):.0f}", flush=True)

json.dump(all_results, open("BFMD_data/shuttle_reconstruction_all.json", "w"))

all_rmses = [r["reproj_rmse_px"] for res in all_results.values() for r in res]
all_speeds = [r["speed_kmh"] for res in all_results.values() for r in res]
n_total = len(all_rmses)
print(f"\n=== TOTAL: {n_total} shots across {len(all_results)} matches ===")
print(f"median reproj RMSE: {np.median(all_rmses):.1f}px, mean: {np.mean(all_rmses):.1f}px")
print(f"speed: median={np.median(all_speeds):.0f}km/h, p5={np.percentile(all_speeds,5):.0f}, p95={np.percentile(all_speeds,95):.0f}")
print(f"fraction with speed > 400km/h (implausible, likely fit failure): {np.mean(np.array(all_speeds)>400)*100:.1f}%")
print(f"fraction with reproj RMSE < 20px (good fit): {np.mean(np.array(all_rmses)<20)*100:.1f}%")

# by shot type
by_type = {}
for res in all_results.values():
    for r in res:
        by_type.setdefault(r["shot_type"], []).append(r["speed_kmh"])
print("\nmedian speed by shot type:")
for st, speeds in sorted(by_type.items(), key=lambda x: -np.median(x[1])):
    print(f"  {st}: {np.median(speeds):.0f}km/h (n={len(speeds)})")
