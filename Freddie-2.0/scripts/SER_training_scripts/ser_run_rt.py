#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
scripts/ser_run_rt.py
=====================

Run a classical Speech Emotion Recognition (SER) model on a WAV file or
microphone stream with post-hoc bias calibration, EMA smoothing, RMS gating,
and true sliding-window streaming for mic input.

Examples (PowerShell, single line)
----------------------------------
# List audio devices (requires model/labels so feature shape can be inferred)
PS> python scripts\ser_run_rt.py --model models\ser_svc_bal_c5_g015_mp.joblib --labels models\ser_svc_bal_c5_g015_mp_labels.json --list-devices

# Microphone (device by index or substring)
PS> python scripts\ser_run_rt.py --model models\ser_svc_bal_c5_g015_mp.joblib --labels models\ser_svc_bal_c5_g015_mp_labels.json --mic --device 1 --feat auto --bias "neutral=-0.15,happy=+0.10,angry=-0.10" --min-conf 0.20 --margin 0.10 --ema 0.6 --rms-gate 0.008 --step 0.25 --debug

# WAV file (sliding across long files)
PS> python scripts\ser_run_rt.py --model models\ser_svc_bal_c5_g015_mp.joblib --labels models\ser_svc_bal_c5_g015_mp_labels.json --wav "path\to\file.wav" --feat auto

Inputs & Outputs
----------------
- Input audio is resampled to 16 kHz mono.
- For WAV mode: windows of 3.0 s with 0.75 s hop.
- For mic mode: windows of 3.0 s with user-defined hop (--step), EMA smoothing,
  RMS gating, and post-softmax bias calibration.
- Printed output includes the top class with confidence and optional top-k list.
"""

__author__  = "Rambod Taherian"
__email__   = "rambodt@uw.com"
__version__ = "1.0.0"
__license__ = "MIT"
__url__     = "https://github.com/rambodt/Freddie-2.0"

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import librosa
import numpy as np\


# ======= Defaults (should match training) ====================================
SR: int     = 16000
N_FFT: int  = 400
HOP: int    = 160
N_MELS: int = 40
WIN_S: float = 3.0     # analysis window (seconds)
STEP_S: float = 0.75   # hop between windows for WAV mode; mic uses --step
# ============================================================================


# ───────────────────────────── I/O helpers ───────────────────────────────────

def load_labels(path: Path) -> List[str]:
    """Load UTF-8/UTF-8-SIG JSON list of label names."""
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8-sig"))


def parse_bias(bias_arg: str, labels: List[str]) -> Dict[str, float]:
    """
    Parse bias string like: "neutral=-0.15,happy=+0.10,angry=-0.10" into a dict.

    Parameters
    ----------
    bias_arg : str
        Comma-separated k=v pairs.
    labels : list[str]
        Known label set to initialize all biases to 0.0.

    Returns
    -------
    dict[str, float]
        Mapping label -> bias value (logit offset).
    """
    bias = {lab: 0.0 for lab in labels}
    if not bias_arg:
        return bias
    for kv in bias_arg.split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        try:
            bias[k] = float(v)
        except Exception:
            pass
    return bias


def apply_bias_and_softmax(logits: np.ndarray, labels: List[str], bias_map: Dict[str, float]) -> np.ndarray:
    """
    Add per-class bias to logits and re-normalize with softmax.

    Parameters
    ----------
    logits : np.ndarray
        Array of log-probabilities or logits, shape (C,).
    labels : list[str]
        Class labels corresponding to `logits` indices.
    bias_map : dict[str, float]
        Label → bias to add before softmax.

    Returns
    -------
    np.ndarray
        Softmax probabilities after biasing, shape (C,), dtype float32.
    """
    out = logits.astype(np.float64).copy()
    for i, lab in enumerate(labels):
        out[i] += float(bias_map.get(lab, 0.0))
    e = np.exp(out - out.max())
    return (e / e.sum()).astype(np.float32)


# ───────────────────────────── Feature builders ──────────────────────────────

def _mels(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Log-mel statistics: mean & std over frames for N_MELS bands → 80 dims.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, power=2.0)
    logS = librosa.power_to_db(S, ref=np.max)
    mel_mu = np.mean(logS, axis=1)
    mel_sd = np.std(logS, axis=1)
    return np.concatenate([mel_mu, mel_sd]).astype(np.float32)  # 80 dims


def _extras(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extra summary features: f0 (yin) stats + RMS + spectral centroid → 7 dims.
    """
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
    """
    MFCC + deltas/delta-deltas from log-mels, mean/std over frames → 78 dims.
    """
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
    ]).astype(np.float32)  # 78 dims


def feats(y: np.ndarray, sr: int, mode: str) -> np.ndarray:
    """
    Build feature vector per segment.

    Parameters
    ----------
    y : np.ndarray
        Audio segment (float32/float64).
    sr : int
        Sample rate (Hz).
    mode : {"mels","mels_plus7","mels_pitch"}
        - mels        → 80 dims
        - mels_plus7  → 87 dims (mels + extras)
        - mels_pitch  → 165 dims (mels + mfcc stats + extras)

    Returns
    -------
    np.ndarray
        Feature vector for classifier input.
    """
    if mode == "mels":
        return _mels(y, sr)
    if mode == "mels_plus7":
        return np.concatenate([_mels(y, sr), _extras(y, sr)], axis=0)
    if mode == "mels_pitch":
        return np.concatenate([_mels(y, sr), _mfcc_stats_from_mels(y, sr), _extras(y, sr)], axis=0)
    raise ValueError(f"Unknown feat mode '{mode}'")


# ─────────────────────────── Model shape helpers ─────────────────────────────

def expected_n_features_from_model(clf) -> int:
    """
    Return expected input feature count from a scikit-learn Pipeline.
    Assumes pipeline like: StandardScaler -> SVC (probability=True).
    """
    try:
        scaler = getattr(clf, "named_steps", {}).get("standardscaler", None)
        n_in = getattr(scaler, "n_features_in_", None)
        if n_in is None and hasattr(clf, "__getitem__"):
            n_in = getattr(clf[0], "n_features_in_", None)
        return int(n_in) if n_in is not None else -1
    except Exception:
        return -1


def infer_feat_mode_from_model(clf) -> str:
    """
    Guess the correct feature mode from the model's expected input size.
    """
    n_in = expected_n_features_from_model(clf)
    if n_in >= 160:
        return "mels_pitch"     # 165
    if n_in == 87:
        return "mels_plus7"     # 80 + 7
    if n_in == 80:
        return "mels"           # 80
    return "mels_pitch"


def topk_from_probs(P: np.ndarray, labels: List[str], k: int = 4) -> List[Tuple[str, float]]:
    """Return top-k (label, probability) pairs from a probability vector."""
    idx = np.argsort(P)[::-1]
    return [(labels[i], float(P[i])) for i in idx[:k]]


# ───────────────────────────── Prediction path ───────────────────────────────

def predict_segment(clf, labels: List[str], y: np.ndarray, feat_mode: str, bias_map: Dict[str, float]) -> np.ndarray:
    """
    Compute class probabilities for a single audio segment with bias calibration.
    """
    x = feats(y, SR, feat_mode)
    pr = clf.predict_proba([x])[0]                     # probabilities
    logit = np.log(np.clip(pr, 1e-9, 1.0))             # log-prob as "logit-like"
    pb = apply_bias_and_softmax(logit, labels, bias_map)
    return pb


def do_wav(args, clf, labels, feat_mode) -> None:
    """
    Evaluate a WAV file by sliding a fixed window over the audio and
    average the probabilities across windows.
    """
    y, _ = librosa.load(str(args.wav), sr=SR, mono=True)
    n_win = int(WIN_S * SR)
    step  = int(STEP_S * SR)
    probs = []
    for t in range(0, max(1, len(y) - n_win + 1), step):
        seg = y[t:t + n_win]
        if len(seg) < n_win:
            break
        pb = predict_segment(clf, labels, seg, feat_mode, args.bias_map)
        probs.append(pb)
    if not probs:
        print("No slices produced (audio too short?).")
        return
    P = np.mean(np.stack(probs, 0), 0)
    idx = np.argsort(P)[::-1]
    top1, top2 = idx[0], idx[1]
    p1, p2 = P[top1], P[top2]
    lab = labels[top1]
    tags = []
    if (p1 - p2) < args.margin:
        tags.append("lowmargin")
    if p1 < args.min_conf:
        tags.append("lowconf")
    stem = Path(args.wav).name
    print(f"{stem}: {lab}  p={p1:.2f} top={[ (labels[i], round(P[i],3)) for i in idx[:4] ]} {' '.join(tags)}")


def list_devices(device_hint: bool | None = None) -> None:
    """
    Print available PortAudio devices with input/output channel counts.
    """
    import sounddevice as sd
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        print(f"[{i:>2}] {d['name']}  in={d['max_input_channels']} out={d['max_output_channels']}")
    if device_hint is not None:
        print("\n(choose with --device <index> or part of the name string)")


def pick_device(device_arg) -> int | None:
    """
    Resolve a device index from an integer or name substring.
    Returns None if no match found.
    """
    import sounddevice as sd
    if device_arg is None:
        return None
    try:
        return int(device_arg)
    except Exception:
        pass
    devs = sd.query_devices()
    name = str(device_arg).lower()
    candidates = [i for i, d in enumerate(devs) if (d["max_input_channels"] > 0 and name in d["name"].lower())]
    return candidates[0] if candidates else None


def do_mic(args, clf, labels, feat_mode) -> None:
    """
    Real-time microphone streaming with true sliding-window processing.

    - A ring buffer accumulates samples; processing occurs when at least
      `--step` seconds of *new* audio arrive.
    - RMS gating suppresses low-energy segments.
    - EMA smoothing stabilizes probabilities over time.
    """
    import sounddevice as sd

    n_win  = int(SR * WIN_S)
    n_step = int(SR * args.step)
    ring   = deque(maxlen=n_win)
    emaP   = None

    # Sample counter to ensure we only process when NEW audio >= n_step arrived
    sample_counter = [0]
    last_proc_samples = 0

    # Select device
    dev_index = pick_device(args.device)
    if dev_index is None:
        print("No matching input device; showing devices:\n")
        list_devices()
        return
    if args.debug:
        print(f"Using input device index: {dev_index}")

    def cb(indata, frames, time_info, status):
        if status:
            print("[stream status]", status, file=sys.stderr)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        ring.extend(mono.tolist())
        sample_counter[0] += len(mono)

    blocksize = max(128, n_step)
    with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                        callback=cb, blocksize=blocksize, device=dev_index):
        print("Mic mode: press Ctrl+C to stop.")
        while True:
            time.sleep(0.02)
            if len(ring) < n_win:
                continue
            # Gate: only process when enough NEW samples came in
            if (sample_counter[0] - last_proc_samples) < n_step:
                continue
            last_proc_samples = sample_counter[0]

            buf = np.asarray(ring, dtype=np.float32)
            seg = buf[-n_win:]

            rms = float(np.sqrt(np.mean(seg**2)))
            if args.debug:
                print(f"[debug] rms={rms:.6f}")
            if rms < args.rms_gate:
                if args.debug:
                    print(f"[debug] below gate ({args.rms_gate}) -> skip")
                continue

            P = predict_segment(clf, labels, seg, feat_mode, args.bias_map)

            if emaP is None or args.ema <= 0.0:
                emaP = P
            else:
                emaP = args.ema * emaP + (1.0 - args.ema) * P
            Puse = emaP if args.ema > 0 else P

            idx = np.argsort(Puse)[::-1]
            top1, top2 = idx[0], idx[1]
            p1, p2 = float(Puse[top1]), float(Puse[top2])
            lab = labels[top1]
            tags = []
            if (p1 - p2) < args.margin:
                tags.append("lowmargin")
            if p1 < args.min_conf:
                tags.append("lowconf")

            if args.show_topk > 0:
                show = [(labels[i], round(float(Puse[i]), 3)) for i in idx[:args.show_topk]]
                print(f"{lab:<8} p={p1:.2f} top={show} {' '.join(tags)}")
            else:
                print(f"{lab:<8} p={p1:.2f} {' '.join(tags)}")


# ──────────────────────────────── CLI ────────────────────────────────────────

def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True, help="Path to joblib model (sklearn Pipeline with predict_proba)")
    ap.add_argument("--labels", required=True, help="Path to JSON list of label names")
    ap.add_argument("--wav", help="Path to WAV file for offline evaluation")
    ap.add_argument("--mic", action="store_true", help="Use microphone for real-time inference")
    ap.add_argument("--mic-simple", action="store_true", help="(compat) same as --mic")
    ap.add_argument("--device", help="Input device index or substring")
    ap.add_argument("--list-devices", action="store_true", help="Print audio devices and exit")

    ap.add_argument("--feat", choices=["auto", "mels", "mels_plus7", "mels_pitch"], default="auto")
    ap.add_argument("--bias", default="neutral=-0.15,happy=+0.10,angry=-0.10", help="Per-class bias map k=v,... (applied in logit space)")
    ap.add_argument("--min-conf", type=float, default=0.20, help="Min confidence to avoid 'lowconf' tag")
    ap.add_argument("--margin",   type=float, default=0.10, help="Min margin between top-2 to avoid 'lowmargin' tag")
    ap.add_argument("--ema",      type=float, default=0.6,  help="EMA smoothing factor (0 disables smoothing)")
    ap.add_argument("--rms-gate", type=float, default=0.008, help="RMS energy gate for mic segments")
    ap.add_argument("--step",     type=float, default=0.25, help="Mic analysis hop in seconds")
    ap.add_argument("--show-topk", type=int, default=4, help="How many classes to print")
    ap.add_argument("--debug", action="store_true", help="Print extra diagnostics")
    args = ap.parse_args()

    # Load model & labels
    labels = load_labels(Path(args.labels))
    clf = joblib.load(args.model)

    # Infer expected features / feature mode
    n_in = expected_n_features_from_model(clf)
    feat_mode = infer_feat_mode_from_model(clf) if args.feat == "auto" else args.feat
    print(f"[info] expected_n_features={n_in}, feat_mode={feat_mode}")

    # Bias map parsed once and attached to args
    args.bias_map = parse_bias(args.bias, labels)

    if args.list_devices:
        list_devices(device_hint=True)
        return

    if args.wav:
        do_wav(args, clf, labels, feat_mode)
        return

    if args.mic or args.mic_simple:
        do_mic(args, clf, labels, feat_mode)
        return

    print("Provide either --wav or --mic/--mic-simple (use --list-devices to see input devices).")


if __name__ == "__main__":
    main()
