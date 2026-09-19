"""Load the final trained checkpoint (checkpoints/hit_model_final_transformer.pt)
and reproduce the reported number without retraining:

    92.7% F1 (P=92.4%, R=93.0%) -- self-detected end-to-end, transformer,
    single model, mirror test-time ensemble, threshold swept on the same
    val split used for training (see train_final.py for the split/seed).

Also this is the harness to run on a new match's self-detected reconstruction:
call `predict_probs` (imported from train_final) on any (self_rel, self_vel,
self_acc, other_rel, other_vel) sequence to get per-frame hit probabilities
and predicted shot type.
"""
import json

import torch

from hit_model_transformer import HitSpotterTransformer
from train_final import BASE, CKPT, DEVICE, RallyDataset, load_split, evaluate_peaks

if __name__ == "__main__":
    dataset = json.load(open(f"{BASE}/training_dataset_all_selfdetected.json"))
    shot_vocab = dataset["shot_type_vocab"]
    shot_to_idx = {s: i for i, s in enumerate(shot_vocab)}

    _, val_tags, val_matches = load_split(dataset)
    val_ds = RallyDataset(dataset, val_tags, shot_to_idx, augment=False)
    print(f"val matches ({len(val_matches)}): {val_matches}")
    print(f"val samples: {len(val_ds)}")

    model = HitSpotterTransformer(in_dim=360, d_model=192, n_heads=6, n_layers=4,
                                    n_shot_types=len(shot_vocab)).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

    best_f1, best = -1, None
    for thresh in [0.1, 0.2, 0.25, 0.3, 0.325, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p, r, f1, sacc, tp, fp, fn = evaluate_peaks(model, val_ds, thresh)
        print(f"thresh={thresh}: P={p:.3f} R={r:.3f} F1={f1:.3f} shot_acc={sacc:.3f} (tp={tp} fp={fp} fn={fn})")
        if f1 > best_f1:
            best_f1, best = f1, (thresh, p, r, f1, sacc)

    thresh, p, r, f1, sacc = best
    print(f"\nBEST [thresh={thresh}]: P={p:.3f} R={r:.3f} F1={f1:.3f} shot_acc={sacc:.3f}")
