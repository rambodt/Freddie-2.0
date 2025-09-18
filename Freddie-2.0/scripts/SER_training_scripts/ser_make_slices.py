#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
slice_ser_windows.py
====================

Slice raw audio into fixed windows with overlap and write a manifest for a
7-class Speech Emotion Recognition (SER) dataset.

Supported sources (recognized by path/file patterns)
----------------------------------------------------
- CREMA-D        .../CREMA-D/AudioWAV/*.wav            (e.g., 1001_IOM_SAD_XX.wav)
- MELD (sorted)  .../meld_sorted/<label>/**            (label taken from folder)
- ESD            .../Emotional Speech Dataset/<Label>/** (folder name = label)
- ASVP-ESD       .../ASVP-ESD-Update/Audio/actor_*/03-01-...-lang-...wav (filters)
- JL Corpus      .../JL Corpus/**                      (best-effort via folder tokens)

Labels (fixed order)
--------------------
angry, disgust, fear, happy, neutral, sad, shocked
- "surprise" → "shocked"
- Files with unmappable labels are skipped.

Command-line (PowerShell)
-------------------------
PS> python .\slice_ser_windows.py --src data\audio_raw --dst data\slices --win 3.0 --overlap 0.25

Outputs
-------
- <dst>/labels.json      : class list (fixed order)
- <dst>/manifest.csv     : rows: file,label (paths relative to <dst>)
- <dst>/<label>/*.wav    : sliced audio windows
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
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import librosa
import soundfile as sf

# ─────────────────────────── Configuration (defaults) ─────────────────────────

SR = 16000            # Target sample rate (Hz) for all slices
WIN_S = 3.0           # Window length in seconds
OVERLAP = 0.25        # Fractional overlap between adjacent windows
MIN_KEEP = 0.40       # Require ≥ this fraction of non-zero samples to keep a slice
EXTS = {".wav", ".mp4", ".m4a", ".flac", ".ogg"}  # Input extensions to scan

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "shocked"]
CLASS_SET = set(CLASSES)


# ─────────────────────────────── Utilities ────────────────────────────────────

def safe_lower(s: str) -> str:
    """Lowercase a string safely (returns original on failure)."""
    try:
        return s.lower()
    except Exception:
        return s


def map_emotion_token(tok: str) -> Optional[str]:
    """
    Map a free-form token to one of the 7 fixed labels, if possible.

    Parameters
    ----------
    tok : str
        Token string (typically a folder or filename component).

    Returns
    -------
    str | None
        Mapped label or None if no confident mapping exists.
    """
    t = safe_lower(tok)
    # Direct match
    if t in CLASS_SET:
        return t
    # Common aliases
    if t in {"surprised", "surprise", "astonished", "amazed"}:
        return "shocked"
    if t in {"calm"}:
        return "neutral"
    if t in {"happiness", "excited", "joy", "joyful", "pleased"}:
        return "happy"
    if t in {"fearful", "panic"}:
        return "fear"
    if t in {"anger", "mad"}:
        return "angry"
    if t in {"disgusted", "contempt"}:
        return "disgust"
    return None


def infer_label(p: Path, *, asvp_lang: str, asvp_speech_only: bool) -> Tuple[Optional[str], bool]:
    """
    Infer a 7-class label for a file path and decide whether to keep it.

    Parameters
    ----------
    p : Path
        Audio file path.
    asvp_lang : str
        Language filter for ASVP-ESD ("all" or "en").
    asvp_speech_only : bool
        If True, keep ASVP-ESD speech-only channel (channel == "01") files.

    Returns
    -------
    (label, keep) : (str | None, bool)
        label : Mapped label or None if unmappable.
        keep  : Whether the file passes dataset-specific filters.
    """
    pp = safe_lower(str(p))
    name = safe_lower(p.name)
    parts = [safe_lower(x) for x in p.parts]

    # MELD (sorted by folder): .../meld_sorted/<label>/**
    for k in CLASS_SET.union({"surprise"}):
        if Path("meld_sorted") in p.parents and f"{k}" in parts:
            lab = "shocked" if k == "surprise" else k
            return lab, True

    # ESD: folder names are the label
    if any("emotional speech dataset" in x for x in parts):
        for ancestor in p.parents:
            m = map_emotion_token(ancestor.name)
            if m:
                return m, True

    # CREMA-D: filename like 1001_IOM_SAD_XX.wav
    if "crema-d" in pp or "crema\\crema-d" in pp or "crema" in pp:
        try:
            bits = name.replace(".wav", "").split("_")
            emo = bits[2]
            m = map_emotion_token(emo)
            if m:
                return m, True
        except Exception:
            pass

    # Other corpora (SAVEE/TESS/Emo-DB) via folder tokens
    for token in p.parts:
        m = map_emotion_token(token)
        if m:
            return m, True

    # JL Corpus: folder tokens e.g., Female1_Angry_3A
    if "jl corpus" in pp:
        for token in parts:
            m = map_emotion_token(token.replace("_", "-").split("-")[0])
            if m:
                return m, True

    # ASVP-ESD update: hyphen-separated fields in filename
    if "asvp-esd" in pp:
        stem = p.stem
        segs = stem.split("-")
        # Expected (≥9 fields): [0]=03 modality, [1]=channel, [2]=emotion, ..., [8]=language
        if len(segs) >= 9:
            channel = segs[1]
            emo_code = segs[2]
            lang = segs[8]

            # Optional filters
            if asvp_speech_only and channel != "01":
                return None, False
            if asvp_lang != "all":
                # For English, require lang code "02"
                if asvp_lang == "en" and lang != "02":
                    return None, False

            code2lab = {
                "01": "neutral",
                "02": "neutral",
                "03": "happy",
                "04": "sad",
                "05": "angry",
                "06": "fear",
                "07": "disgust",
                "08": "shocked",  # surprised
                # 09 excited (skip), 10 pleasure (skip), 11 pain (skip),
                # 12 disappointment (skip), 13 breath (skip)
            }
            lab = code2lab.get(emo_code)
            if lab:
                return lab, True
            return None, False

    # Default: unknown corpus
    return None, True


def slice_one(src: Path, dst_root: Path, label: str, sr: int, win_s: float, overlap: float) -> List[str]:
    """
    Slice one audio file into fixed-length windows.

    Parameters
    ----------
    src : Path
        Source audio path.
    dst_root : Path
        Destination root directory for slices.
    label : str
        Target label subfolder under `dst_root`.
    sr : int
        Target sample rate (Hz).
    win_s : float
        Window length in seconds.
    overlap : float
        Overlap fraction in [0, 1).

    Returns
    -------
    list[str]
        Paths relative to `dst_root` for each slice written.
    """
    rels: List[str] = []
    try:
        y, _ = librosa.load(str(src), sr=sr, mono=True)
    except Exception as e:
        print(f"!! load fail {src}: {e}")
        return rels

    # Simple VAD: keep voiced intervals, drop long silences
    intervals = librosa.effects.split(y, top_db=25)
    if intervals.size:
        y = librosa.util.fix_length(
            librosa.util.flatten([y[a:b] for a, b in intervals]),
            size=len(y)  # length itself not used later
        )

    # Sliding windows
    n_win = int(sr * win_s)
    hop = int(n_win * (1 - overlap))
    if hop < 1:
        hop = 1

    out_dir = dst_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    t = 0
    while t + n_win <= len(y):
        seg = y[t:t + n_win]
        # Keep windows with sufficient non-zero content
        nz = (seg != 0).sum() / max(1, len(seg))
        if nz >= MIN_KEEP:
            out = out_dir / f"{src.stem}_t{int(t/sr*1000):08d}.wav"
            sf.write(str(out), seg, sr)
            rels.append(str(out.relative_to(dst_root)).replace("\\", "/"))
        t += hop

    return rels


# ──────────────────────────────── CLI ────────────────────────────────────────

def main() -> None:
    """
    Command-line entry point.

    Options
    -------
    --src : root directory to scan for audio files
    --dst : destination directory for slices (required)
    --win : window length in seconds (default: WIN_S)
    --sr  : sample rate (default: SR)
    --min-keep : minimum non-zero fraction per slice (default: MIN_KEEP)
    --overlap  : overlap fraction between slices (default: OVERLAP)
    --exts     : comma-separated list of file extensions to include
    --asvp-lang: ASVP-ESD language filter ("all" or "en", default "en")
    --asvp-speech-only : keep ASVP speech-only channel (channel == "01")
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/audio_raw", help="root to scan for audio")
    ap.add_argument("--dst", required=True, help="destination folder for slices")
    ap.add_argument("--win", type=float, default=WIN_S)
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--min-keep", type=float, default=MIN_KEEP)
    ap.add_argument("--overlap", type=float, default=OVERLAP)
    ap.add_argument("--exts", default=",".join(sorted(EXTS)))
    # ASVP filters
    ap.add_argument("--asvp-lang", choices=["all", "en"], default="en",
                    help="ASVP-ESD language filter (default en)")
    ap.add_argument("--asvp-speech-only", action="store_true",
                    help="ASVP-ESD: keep speech-only (channel=01)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    # Bind run-time overrides
    global SR, WIN_S, MIN_KEEP
    SR = args.sr
    WIN_S = args.win
    MIN_KEEP = args.min_keep
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}

    manifest_rows: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {c: 0 for c in CLASSES}
    skipped: List[Tuple[str, str]] = []

    asvp_drop_lang = 0
    asvp_drop_chan = 0

    files = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    for i, f in enumerate(files, 1):
        lab, keep = infer_label(f, asvp_lang=args.asvp_lang, asvp_speech_only=args.asvp_speech_only)
        if not keep:
            # Count ASVP-specific drops (best effort)
            if "asvp-esd" in safe_lower(str(f)):
                stem = f.stem.split("-")
                if len(stem) >= 9:
                    if args.asvp_speech_only and stem[1] != "01":
                        asvp_drop_chan += 1
                    lang = stem[8]
                    if args.asvp_lang == "en" and lang != "02":
                        asvp_drop_lang += 1
            continue

        if lab is None or lab not in CLASS_SET:
            skipped.append((str(f), "no_label_match"))
            continue

        rels = slice_one(f, dst, lab, args.sr, args.win, args.overlap)
        counts[lab] += len(rels)
        for r in rels:
            manifest_rows.append((r, lab))

    # Write labels.json (fixed class order)
    labels_json = dst / "labels.json"
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)

    # Write manifest.csv
    manifest_csv = dst / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label"])
        for r, lab in manifest_rows:
            w.writerow([r, lab])

    total = sum(counts.values())
    print(f"Saved {total} slices to {dst}")
    print(f"Label counts: {counts}")
    if asvp_drop_lang or asvp_drop_chan:
        print(f"ASVP filters -> dropped (lang): {asvp_drop_lang}, (channel): {asvp_drop_chan}")
    if skipped:
        print("Skipped files:")
        for s, why in skipped[:200]:
            print(f"  - {s} -> {why}")
        if len(skipped) > 200:
            print(f"  ... and {len(skipped) - 200} more")


if __name__ == "__main__":
    main()
