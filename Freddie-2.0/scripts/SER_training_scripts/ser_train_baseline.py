#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
scripts/ser_train_baseline.py
=============================

Train a simple 7-class Speech Emotion Recognition (SER) baseline from a sliced
audio manifest.

Manifest
--------
CSV with header `file,label`, one row per slice (paths can be relative to the
manifest directory or absolute).

Labels
------
JSON array of class names, e.g.:
["angry","disgust","fear","happy","neutral","sad","shocked"]

Examples (PowerShell one-liners)
--------------------------------
# STRATIFIED split (row-wise; typical baseline)
PS> python scripts\ser_train_baseline.py --manifest data\ser_slices_v0p25_7c_en_sp\manifest_balanced.csv --labels data\ser_slices_v0p25_7c_en_sp\labels.json --out models\ser_svc_bal_c5_g015_melspitch_STRAT.joblib --split stratified --feat mels_pitch --class-weight none --C 5 --gamma 0.15 --kernel rbf --test-size 0.15 --seed 42

# GROUPED split (no leakage across slices from the same source file)
PS> python scripts\ser_train_baseline.py --manifest data\ser_slices_v0p25_7c_en_sp\manifest_balanced.csv --labels data\ser_slices_v0p25_7c_en_sp\labels.json --out models\ser_svc_bal_c5_g015_melspitch_GROUP.joblib --split grouped --min-per-class 50 --feat mels_pitch --class-weight none --C 5 --gamma 0.15 --kernel rbf --test-size 0.15 --seed 42

Outputs
-------
- Trained model: <out>.joblib
- Metadata:      <out>_stats.json, <out>_labels.json
- Optional PNG:  <out>_cm.png (confusion matrix; if matplotlib available)
"""
__author__  = "Rambod Taherian"
__email__   = "rambodt@uw.com"
__version__ = "1.0.0"
__license__ = "MIT"
__url__     = "https://github.com/rambodt/Freddie-2.0"

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil  # kept for compatibility if users script around it
from pathlib import Path
from typing import List, Tuple

import joblib
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ===== Defaults (match slicer/runtime) =======================================
SR     = 16000
N_FFT  = 400
HOP    = 160
N_MELS = 40
WIN_S  = 3.0
# ============================================================================


def load_manifest_features(manifest_csv: str, labels_json: str, feat_mode: str):
    """
    Convenience loader: returns (X, y, labels).

    Parameters
    ----------
    manifest_csv : str
        Path to manifest CSV.
    labels_json : str
        Path to labels JSON array.
    feat_mode : str
        Feature mode; this helper assumes 'mels_pitch' in build_dataset.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        Feature matrix X, integer labels y, and label list.
    """
    with open(labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)
    X, y = build_dataset(manifest_csv, labels, feat_mode)
    return X, y, labels


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder for feat_mode="mels_pitch" → 165-D vector:
#   160 = [logmel mean(40) + logmel std(40) + Δlogmel mean(40) + Δlogmel std(40)]
#     5 = [f0_mean, f0_std, f0_min, f0_max, voiced_ratio]
# Assumes each manifest row points to a ~3s slice at sr=16000.
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(manifest_csv, labels, feat_mode):
    import soundfile as sf  # local import to keep module load light

    assert feat_mode == "mels_pitch", f"Unsupported feat_mode={feat_mode}"

    # Config
    sr          = 16000
    n_fft       = 400
    hop_length  = 160
    n_mels      = 40
    win_s       = 3.0
    win_len     = int(sr * win_s)

    rows = read_manifest(Path(manifest_csv))

    X_list, y_list = [], []
    label2id = {c: i for i, c in enumerate(labels)}

    for wav_path, lab in rows:
        # Load audio (wav/flac), mono, ~3 s
        x, sr_ = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if sr_ != sr:
            x = librosa.resample(x, orig_sr=sr_, target_sr=sr)
        if x.ndim > 1:
            x = np.mean(x, axis=-1)

        # Center crop/pad to exact 3 s for stable stats
        if len(x) < win_len:
            pad = win_len - len(x)
            pre = pad // 2
            post = pad - pre
            x = np.pad(x, (pre, post))
        elif len(x) > win_len:
            start = (len(x) - win_len) // 2
            x = x[start:start + win_len]

        # Log-mel + delta
        S = librosa.feature.melspectrogram(
            y=x, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=2.0
        )
        S_db = librosa.power_to_db(S, ref=np.max)           # (40, T)
        d1   = librosa.feature.delta(S_db, order=1)         # (40, T)

        m0 = S_db.mean(axis=1); s0 = S_db.std(axis=1)
        m1 = d1.mean(axis=1);   s1 = d1.std(axis=1)
        mel_stats = np.concatenate([m0, s0, m1, s1], axis=0)  # (160,)

        # Pitch (YIN) with simple voiced mask
        f0 = librosa.yin(x, fmin=50, fmax=500, sr=sr, frame_length=n_fft, hop_length=hop_length)
        voiced = np.isfinite(f0) & (f0 > 0)
        if voiced.any():
            f0v = f0[voiced]
            fstats = np.array([f0v.mean(), f0v.std(), f0v.min(), f0v.max(), voiced.mean()], dtype=np.float32)
        else:
            fstats = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        feat = np.concatenate([mel_stats.astype(np.float32), fstats], axis=0)  # (165,)
        X_list.append(feat)
        y_list.append(label2id[lab])

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# ---------- I/O helpers ----------

def load_labels(path: Path) -> Tuple[List[str], dict]:
    """Load labels.json (array of strings) and return (list, map)."""
    with open(path, "rb") as f:
        labs = json.loads(f.read().decode("utf-8-sig"))
    if not isinstance(labs, list) or not all(isinstance(x, str) for x in labs):
        raise ValueError("labels.json must be a JSON array of strings")
    lab2i = {lab: i for i, lab in enumerate(labs)}
    return labs, lab2i


def read_manifest(path: Path) -> List[Tuple[Path, str]]:
    """Read (file,label) rows from manifest; resolve relative paths against manifest dir."""
    rows: List[Tuple[Path, str]] = []
    root = path.parent
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "file" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("manifest.csv must have columns: file,label")
        for r in reader:
            p = Path(r["file"])
            if not p.is_absolute():
                p = (root / p).resolve()
            rows.append((p, r["label"]))
    return rows


# ---------- Feature shortcuts (for --feat modes) ----------

def _mels(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, power=2.0)
    logS = librosa.power_to_db(S, ref=np.max)
    mel_mu = np.mean(logS, axis=1)
    mel_sd = np.std(logS, axis=1)
    return np.concatenate([mel_mu, mel_sd]).astype(np.float32)  # 80


def _extras(y: np.ndarray, sr: int) -> np.ndarray:
    # f0 + RMS + centroid → 7 dims
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=1024)
        f0 = f0[np.isfinite(f0)]
        if f0.size:
            f0_mu, f0_sd = float(np.mean(f0)), float(np.std(f0))
            f0_vr = float(f0.size / max(1, int(len(y) / HOP)))
        else:
            f0_mu = f0_sd = 0.0
            f0_vr = 0.0
    except Exception:
        f0_mu = f0_sd = 0.0
        f0_vr = 0.0
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP).squeeze()
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP).squeeze()
    rms_mu, rms_sd = float(np.mean(rms)), float(np.std(rms))
    cen_mu, cen_sd = float(np.mean(cen)), float(np.std(cen))
    return np.array([f0_mu, f0_sd, f0_vr, rms_mu, rms_sd, cen_mu, cen_sd], dtype=np.float32)


def _mfcc_stats_from_mels(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, power=2.0)
    D = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=D, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(d1,   axis=1), np.std(d1,   axis=1),
        np.mean(d2,   axis=1), np.std(d2,   axis=1),
    ]).astype(np.float32)  # 78


def wav_to_feat(path: Path, mode: str) -> np.ndarray:
    """Compute a feature vector for one slice according to `mode`."""
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    if len(y) < SR * WIN_S:
        y = np.pad(y, (0, int(SR * WIN_S) - len(y)), mode="constant")
    if mode == "mels":
        x = _mels(y, sr)                                                # 80
    elif mode == "mels_plus7":
        x = np.concatenate([_mels(y, sr), _extras(y, sr)], 0)           # 87
    elif mode == "mels_pitch":
        x = np.concatenate([_mels(y, sr), _mfcc_stats_from_mels(y, sr), _extras(y, sr)], 0)  # 165
    else:
        raise ValueError("feat mode must be one of: mels, mels_plus7, mels_pitch")
    return x


# ---------- Grouping (avoid leakage across slices from same source) ----------

_slice_suffix = re.compile(r"_t\d{8}$")


def group_from_slice_path(p: Path) -> str:
    """
    Use original (unsliced) file stem as the group id.

    Examples
    --------
    'dia125_utt3_t00000000.wav'      -> 'dia125_utt3'
    '03-01-03-01-01-02-23_t00000000' -> '03-01-03-01-01-02-23'
    """
    stem = p.stem
    orig = _slice_suffix.sub("", stem)
    if not orig:
        orig = stem
    return orig


# ---------- Split helpers ----------

def try_group_split(y: np.ndarray, groups: np.ndarray, test_size: float, seed: int,
                    min_per_class: int, n_classes: int, max_tries: int = 200):
    """
    Repeatedly sample a GroupShuffleSplit until each class has at least
    `min_per_class` in TRAIN and at least 1 sample in TEST.
    """
    gss = GroupShuffleSplit(n_splits=max_tries, test_size=test_size, random_state=seed)
    best = None
    for t, (tr, te) in enumerate(gss.split(np.zeros_like(y), y, groups)):
        train_counts = np.bincount(y[tr], minlength=n_classes)
        test_counts  = np.bincount(y[te], minlength=n_classes)
        ok_train = not np.any(train_counts < min_per_class)
        ok_test  = not np.any(test_counts == 0)
        best = (tr, te, train_counts, test_counts)
        if ok_train and ok_test:
            return best, True, t + 1
    return best, False, max_tries


# ───────────────────────────────────── Main ──────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--labels',   required=True)
    ap.add_argument('--out',      required=True)

    ap.add_argument('--feat', choices=['mels', 'mels_plus7', 'mels_pitch'], default='mels_pitch')
    ap.add_argument('--class-weight', default='balanced', choices=['balanced', 'none'])
    ap.add_argument('--C', type=float, default=3.0)
    ap.add_argument('--gamma', default='scale', help="float or 'scale'/'auto'")
    ap.add_argument('--kernel', default='rbf', choices=['rbf', 'linear', 'poly', 'sigmoid'])

    ap.add_argument('--split', choices=['stratified', 'grouped'], default='stratified',
                    help="Train/test split strategy. 'grouped' avoids leakage across slices from the same source file.")
    ap.add_argument('--min-per-class', type=int, default=50,
                    help="Min samples per class in TRAIN for grouped split; ignored for stratified.")
    ap.add_argument('--test-size', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    MAN = Path(args.manifest).resolve()
    LAB = Path(args.labels).resolve()
    OUT = Path(args.out)

    classes, lab2i = load_labels(LAB)
    print(f"Classes ({len(classes)}): {classes}")

    feat_dims = dict(mels=80, mels_plus7=87, mels_pitch=165)[args.feat]
    print(f"[info] feat_mode={args.feat}, expected_dim={feat_dims}")

    rows = read_manifest(MAN)
    print(f"Found {len(rows)} slices in manifest")

    # Build dataset arrays
    X_list, y_list, groups_list = [], [], []
    skipped = 0
    for p, lab in rows:
        try:
            x = wav_to_feat(p, args.feat)
            X_list.append(x)
            y_list.append(lab2i[lab])
            groups_list.append(group_from_slice_path(p))
        except Exception as e:
            print(f"!! skip {p}: {e}")
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} files due to errors.")
    if not X_list:
        raise RuntimeError("No features loaded; check manifest paths.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(groups_list)

    # Split
    if args.split == 'stratified':
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        print("[split] STRATIFIED row-wise split")
    else:
        (tr_idx, te_idx, tr_counts, te_counts), ok, tries = try_group_split(
            y, groups, args.test_size, args.seed, args.min_per_class, n_classes=len(classes), max_tries=200
        )
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        print(f"[split] GROUPED by source (tries={tries}, ok={ok})")
        print("        train per-class:", {classes[i]: int(c) for i, c in enumerate(tr_counts)})
        print("        test  per-class:", {classes[i]: int(c) for i, c in enumerate(te_counts)})
        if not ok:
            print("        (warning) Could not meet min-per-class in train and >=1 per class in test.")

    # Model (scikit-learn pipeline)
    cw = 'balanced' if args.class_weight == 'balanced' else None
    gamma_val = args.gamma
    if isinstance(gamma_val, str):
        if gamma_val.lower() not in ('scale', 'auto'):
            try:
                gamma_val = float(gamma_val)
            except Exception:
                gamma_val = 'scale'

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(kernel=args.kernel, gamma=gamma_val, C=args.C, probability=True, class_weight=cw)
    )

    # Train & report
    clf.fit(Xtr, ytr)
    ypr = clf.predict(Xte)
    acc = accuracy_score(yte, ypr)
    print(f"Test accuracy: {acc:.3f}")
    try:
        print(classification_report(yte, ypr, target_names=classes, digits=3))
    except Exception:
        print(classification_report(yte, ypr, digits=3))

    # Save model and metadata
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT)
    print(f"Saved model to {OUT}")

    stats = dict(sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                 win_s=WIN_S, classes=classes, feat=args.feat, split=args.split)
    with open(OUT.with_suffix('').as_posix() + '_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    with open(OUT.with_suffix('').as_posix() + '_labels.json', 'w', encoding='utf-8') as f:
        json.dump(classes, f, indent=2)
    print("Wrote *_stats.json and *_labels.json next to the model.")

    # Optional: confusion matrix PNG
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator  # noqa: F401 (kept for axes control if edited)
        cm = confusion_matrix(yte, ypr, labels=list(range(len(classes))))
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
               xticklabels=classes, yticklabels=classes,
               ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
        fig.tight_layout()
        out_png = OUT.with_suffix('').as_posix() + "_cm.png"
        plt.savefig(out_png)
        print(f"Saved confusion matrix to {out_png}")
    except Exception as e:
        print("(confusion matrix not saved:", e, ")")


if __name__ == "__main__":
    main()
