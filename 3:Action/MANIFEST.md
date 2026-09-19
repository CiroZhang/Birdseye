# 3:Action -- file guide

Read HIT_DETECTION_REPORT.txt for the original full writeup (goal, method, published-work
comparison) -- it predates the self-detected redo, whose result supersedes its "Known limitation"
section. See readme.md section 3 for the full narrative + baseline comparison table.

**Housekeeping note (2026-09-09)**: this folder was fully consolidated down to only what reproduces
the one number actually reported. Removed the entire v2/v3/v4/v5/v6/multiseed progression
(`train_hit_model.py` v1, `train_hit_model_v2/v3/v4/v6.py`, `train_hit_model_v4_selfdetected.py`,
`train_hit_model_v6_selfdetected.py`, `train_multiseed.py` + `train_multiseed_selfdetected.py`,
`eval_hit_model.py`, `eval_ensemble.py`, `eval_final_ensemble_selfdetected.py`, `kfold_eval.py`,
the v5 shuttle-feature experiment + its `add_shuttle_features.py` helper, the smoothed-training
experiment), the annotation-based `hit_model.py` (TCN) architecture and `build_training_data_all.py`
(nothing uses either anymore), and every checkpoint/log belonging to a removed script. All of those
numbers (94.7% annotation single, 95.2% annotation ensemble, 91.3% self-detected TCN, 89.8%+/-3.3%
k-fold, the 93.2% self-detected ensemble found-but-not-reported, the 90.7% shuttle-feature negative
result, the 91.17%/91.31% smoothing-as-training-lever negative result) are preserved in prose in
readme.md and HIT_DETECTION_REPORT.txt -- only the code to regenerate them is gone. The one thing we
actually care about going forward -- the self-detected transformer, 92.7% F1 -- is fully
self-contained and independently verified to still reproduce exactly (P=0.924 R=0.930 F1=0.927 at
thresh=0.5) after the consolidation, from the checkpoint alone, with zero cross-file imports.

## What's here now
- **hit_model_transformer.py** -- `HitSpotterTransformer`, the only architecture kept (global
  self-attention; see file docstring for why this beat the TCN family it replaced).
- **build_training_data_selfdetected.py** -- aligns our self-detected VIMO reconstruction
  (`2:Player/run_vimo_bfmd_selfdetect.py`'s output) with BFMD's `hit_inferred` labels for shot
  timing/type (real supervision -- there's no other way to know when a shot starts/ends). Produces
  `training_dataset_all_selfdetected.json` (1055 rallies, 329,495 frames, 12/12 matches).
- **train_final.py** -- self-contained training script (no imports from any other train_*.py --
  everything it needs, `mirror_lr`, `collate`, `RallyDataset`, `make_features`, the training loop,
  `evaluate_peaks`, is inlined). Same recipe/split/seed as the original run. Saves the best
  checkpoint to `checkpoints/hit_model_final_transformer.pt`.
- **eval_final.py** -- loads that checkpoint and reproduces the reported number without retraining
  (imports the dataset/eval helpers from train_final.py). This is also the harness to run on a new
  match: call `predict_probs` on any self-detected joint sequence for per-frame hit probability +
  shot-type prediction.
- **checkpoints/hit_model_final_transformer.pt** -- the one weight file that matters. **92.7% F1
  (P=92.4%, R=93.0%)** -- self-detected end-to-end (our own court + player detection, no BFMD
  annotations anywhere upstream), transformer, single model, mirror test-time ensemble, threshold
  swept on the val split (match-level holdout, seed 0).

## Demo/visualization-quality-only scripts (do NOT affect the trained model's F1)
- fix_teleports.py -- removes single-frame joint position glitches, applied only to the one demo
  rally for a clean visualization.
- smooth_joints.py -- Savitzky-Golay temporal smoothing, display only. Has an `if __name__ ==
  "__main__"` guard (added 2026-09-09 after an import of its functions elsewhere accidentally
  re-triggered its own top-level script) -- import `smooth_player`/`jitter_metric` freely, it no
  longer re-runs on import.

## Known issue -- hardcoded paths
train_final.py and eval_final.py have the working session's local scratchpad path hardcoded
(`BASE`, `CKPT_DIR`) rather than a path relative to this folder -- edit those two constants before
running directly from Killarney. Same caveat as 1:Court's scripts.
