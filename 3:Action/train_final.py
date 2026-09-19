"""Self-contained training + evaluation script for the final, reported hit-
detection model: HitSpotterTransformer (hit_model_transformer.py) trained on
our own self-detected reconstruction (court + player detection, no BFMD
annotations anywhere upstream -- see 1:Court and 2:Player).

This is the consolidated result of a longer progression (see readme.md
section 3 and HIT_DETECTION_REPORT.txt for the full history: a TCN baseline,
added opponent-context features, added acceleration, this transformer variant,
and a 3-model ensemble that scored higher but was deliberately not reported
due to multiple-comparisons risk on a single validation split). All of that
is documented in prose already; this file only keeps what reproduces the one
number we actually report:

    **92.7% F1 (P=92.4%, R=93.0%)** -- self-detected end-to-end, transformer,
    single model, mirror test-time ensemble.

Run with no arguments to retrain from scratch on
BFMD_data/training_dataset_all_selfdetected.json (produced by
build_training_data_selfdetected.py); saves the best checkpoint to
checkpoints/hit_model_final_transformer.pt and prints P/R/F1 at the
best-found peak-detection threshold, same recipe/split/seed as the original
run.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks

from hit_model_transformer import HitSpotterTransformer

_HERE = os.path.dirname(os.path.abspath(__file__))
# training_dataset_all_selfdetected.json is produced by build_training_data_selfdetected.py,
# written into the same BFMD_data dir it reads annotations from -- see that script for BASE.
BASE = os.environ.get(
    "BFMD_DATA_ROOT",
    os.path.normpath(os.path.join(_HERE, "..", "..", "Dataset", "BFMD", "BFMD_data")),
)
CKPT_DIR = os.path.join(_HERE, "checkpoints")
CKPT = f"{CKPT_DIR}/hit_model_final_transformer.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TOLERANCE_FRAMES = 10

# SMPL 24-joint left/right swap pairs; 0,3,6,9,12,15 are midline and stay put.
LR_SWAP = {1: 2, 2: 1, 4: 5, 5: 4, 7: 8, 8: 7, 10: 11, 11: 10, 13: 14, 14: 13,
           16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20, 22: 23, 23: 22}
LR_PERM = [LR_SWAP.get(i, i) for i in range(24)]


def mirror_lr(rel_joints):
    """rel_joints: (T, 24, 3). Reflect across the lateral (X) axis and swap
    L/R joint indices -- a real, anatomically valid mirrored pose. Used both
    for training-time augmentation and free test-time ensembling."""
    m = rel_joints[:, LR_PERM, :].copy()
    m[..., 0] *= -1
    return m


def accel(vel):
    a = np.zeros_like(vel)
    a[1:] = vel[1:] - vel[:-1]
    return a


def make_features(self_rel, self_vel, self_acc, other_rel, other_vel):
    T = self_rel.shape[0]
    return np.concatenate([
        self_rel.reshape(T, -1), self_vel.reshape(T, -1), self_acc.reshape(T, -1),
        other_rel.reshape(T, -1), other_vel.reshape(T, -1),
    ], axis=1)


class RallyDataset(Dataset):
    def __init__(self, dataset, tags, shot_to_idx, augment=False):
        self.samples = []
        self.augment = augment
        for tag in tags:
            r = dataset["rallies"][tag]
            if r["nan_frac_top"] > 0.5 or r["nan_frac_bottom"] > 0.5:
                continue
            top_rel = np.nan_to_num(np.array(r["top_rel"], dtype=np.float32))
            top_vel = np.nan_to_num(np.array(r["top_vel"], dtype=np.float32))
            bot_rel = np.nan_to_num(np.array(r["bottom_rel"], dtype=np.float32))
            bot_vel = np.nan_to_num(np.array(r["bottom_vel"], dtype=np.float32))
            top_acc, bot_acc = accel(top_vel), accel(bot_vel)

            for self_rel, self_vel, self_acc, other_rel, other_vel, tgt_key, shot_key in [
                (top_rel, top_vel, top_acc, bot_rel, bot_vel, "target_top", "shot_type_top"),
                (bot_rel, bot_vel, bot_acc, top_rel, top_vel, "target_bottom", "shot_type_bottom"),
            ]:
                hit_tgt = np.array(r[tgt_key], dtype=np.float32)
                shot_tgt = np.array([shot_to_idx.get(s, -1) for s in r[shot_key]], dtype=np.int64)
                self.samples.append((self_rel, self_vel, self_acc, other_rel, other_vel, hit_tgt, shot_tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        self_rel, self_vel, self_acc, other_rel, other_vel, hit_tgt, shot_tgt = self.samples[i]
        if self.augment and np.random.rand() < 0.5:
            self_rel, self_vel, self_acc = mirror_lr(self_rel), mirror_lr(self_vel), mirror_lr(self_acc)
            other_rel, other_vel = mirror_lr(other_rel), mirror_lr(other_vel)
        feats = make_features(self_rel, self_vel, self_acc, other_rel, other_vel)
        return feats, hit_tgt, shot_tgt


def collate(batch):
    max_len = max(b[0].shape[0] for b in batch)
    B = len(batch)
    in_dim = batch[0][0].shape[1]
    feats = np.zeros((B, max_len, in_dim), dtype=np.float32)
    hit_tgt = np.zeros((B, max_len), dtype=np.float32)
    shot_tgt = np.full((B, max_len), -1, dtype=np.int64)
    mask = np.zeros((B, max_len), dtype=bool)
    for i, (f, h, s) in enumerate(batch):
        T = f.shape[0]
        feats[i, :T] = f
        hit_tgt[i, :T] = h
        shot_tgt[i, :T] = s
        mask[i, :T] = True
    return (torch.tensor(feats), torch.tensor(hit_tgt), torch.tensor(shot_tgt), torch.tensor(mask))


def run_epoch(model, loader, opt=None, pos_weight=1.0):
    train = opt is not None
    model.train(train)
    total_hit_loss, total_shot_loss, n_batches = 0.0, 0.0, 0
    for feats, hit_tgt, shot_tgt, mask in loader:
        feats, hit_tgt, shot_tgt, mask = (t.to(DEVICE) for t in (feats, hit_tgt, shot_tgt, mask))
        pad_mask = ~mask  # collate's mask is True=valid; transformer wants True=padded
        hit_logits, shot_logits = model(feats, pad_mask=pad_mask)

        weight = 1.0 + (pos_weight - 1.0) * hit_tgt
        hit_loss_raw = F.binary_cross_entropy_with_logits(hit_logits, hit_tgt, reduction="none") * weight
        hit_loss = (hit_loss_raw * mask).sum() / mask.sum()

        shot_mask = (shot_tgt >= 0) & mask
        shot_loss = (F.cross_entropy(shot_logits[shot_mask], shot_tgt[shot_mask])
                     if shot_mask.any() else torch.tensor(0.0, device=DEVICE))

        loss = hit_loss + 0.3 * shot_loss
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_hit_loss += hit_loss.item()
        total_shot_loss += shot_loss.item()
        n_batches += 1
    return total_hit_loss / n_batches, total_shot_loss / n_batches


def predict_probs(model, self_rel, self_vel, self_acc, other_rel, other_vel, mirror_ensemble=True):
    feats = make_features(self_rel, self_vel, self_acc, other_rel, other_vel)
    x = torch.tensor(feats).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        hl, sl = model(x)
    probs = torch.sigmoid(hl)[0].cpu().numpy()
    shot_pred = sl[0].argmax(dim=-1).cpu().numpy()
    if mirror_ensemble:
        m_feats = make_features(mirror_lr(self_rel), mirror_lr(self_vel), mirror_lr(self_acc),
                                 mirror_lr(other_rel), mirror_lr(other_vel))
        xm = torch.tensor(m_feats).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            hlm, _ = model(xm)
        probs = (probs + torch.sigmoid(hlm)[0].cpu().numpy()) / 2
    return probs, shot_pred


def evaluate_peaks(model, ds, height_thresh, mirror_ensemble=True):
    model.eval()
    tp, fp, fn = 0, 0, 0
    shot_correct, shot_total = 0, 0
    for i in range(len(ds)):
        self_rel, self_vel, self_acc, other_rel, other_vel, hit_tgt, shot_tgt = ds.samples[i]
        probs, shot_pred = predict_probs(model, self_rel, self_vel, self_acc, other_rel, other_vel, mirror_ensemble)
        true_hits = [f for f in range(len(hit_tgt)) if shot_tgt[f] >= 0]
        peaks, _ = find_peaks(probs, height=height_thresh, distance=TOLERANCE_FRAMES)
        matched = set()
        for p in peaks:
            candidates = [f for f in true_hits if abs(f - p) <= TOLERANCE_FRAMES and f not in matched]
            if candidates:
                best = min(candidates, key=lambda f: abs(f - p))
                matched.add(best)
                tp += 1
                shot_total += 1
                if shot_pred[p] == shot_tgt[best]:
                    shot_correct += 1
            else:
                fp += 1
        fn += len(true_hits) - len(matched)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    shot_acc = shot_correct / shot_total if shot_total else 0.0
    return precision, recall, f1, shot_acc, tp, fp, fn


def load_split(dataset):
    """Same seed/split as the original run: match-level holdout, 1/5 of
    matches held out for validation."""
    match_of_rally = dataset["match_of_rally"]
    matches = sorted(set(match_of_rally.values()))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(matches))
    n_val_matches = max(2, len(matches) // 5)
    val_matches = {matches[i] for i in perm[:n_val_matches]}
    train_matches = {matches[i] for i in perm[n_val_matches:]}
    all_tags = sorted(dataset["rallies"].keys())
    train_tags = [t for t in all_tags if match_of_rally[t] in train_matches]
    val_tags = [t for t in all_tags if match_of_rally[t] in val_matches]
    return train_tags, val_tags, val_matches


if __name__ == "__main__":
    dataset = json.load(open(f"{BASE}/training_dataset_all_selfdetected.json"))
    shot_vocab = dataset["shot_type_vocab"]
    shot_to_idx = {s: i for i, s in enumerate(shot_vocab)}
    print(f"shot vocab ({len(shot_vocab)}): {shot_vocab}")

    train_tags, val_tags, val_matches = load_split(dataset)
    print(f"val matches ({len(val_matches)}): {val_matches}")

    train_ds = RallyDataset(dataset, train_tags, shot_to_idx, augment=True)
    val_ds = RallyDataset(dataset, val_tags, shot_to_idx, augment=False)
    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    all_targets = np.concatenate([s[5] for s in train_ds.samples])
    pos_frac = (all_targets > 0.5).mean()
    pos_weight = min(10.0, (1 - pos_frac) / max(pos_frac, 1e-4))
    print(f"positive frame fraction: {pos_frac:.3f}, pos_weight: {pos_weight:.2f}")

    # small batch: full self-attention is O(T^2) in sequence length and some
    # rallies run 900+ frames; ran out of MPS memory at batch_size=24.
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate)

    model = HitSpotterTransformer(in_dim=360, d_model=192, n_heads=6, n_layers=4,
                                    n_shot_types=len(shot_vocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    N_EPOCHS = 60
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    best_f1 = -1
    for epoch in range(N_EPOCHS):
        tr_hit, tr_shot = run_epoch(model, train_loader, opt, pos_weight=pos_weight)
        va_hit, va_shot = run_epoch(model, val_loader, None, pos_weight=pos_weight)
        sched.step()
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            best_thresh_f1, best_thresh = -1, 0.3
            for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                p, r, f1, sacc, tp, fp, fn = evaluate_peaks(model, val_ds, thresh)
                if f1 > best_thresh_f1:
                    best_thresh_f1, best_thresh = f1, thresh
            p, r, f1, sacc, tp, fp, fn = evaluate_peaks(model, val_ds, best_thresh)
            print(f"epoch {epoch:2d}: train hit={tr_hit:.4f} shot={tr_shot:.3f} | val hit={va_hit:.4f} shot={va_shot:.3f} | "
                  f"[thresh={best_thresh}] P={p:.3f} R={r:.3f} F1={f1:.3f} shot_acc={sacc:.3f} (tp={tp} fp={fp} fn={fn})",
                  flush=True)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), CKPT)
        else:
            print(f"epoch {epoch:2d}: train hit={tr_hit:.4f} shot={tr_shot:.3f} | val hit={va_hit:.4f} shot={va_shot:.3f}", flush=True)

    print("done, best val F1:", best_f1)
