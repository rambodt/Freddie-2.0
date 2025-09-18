#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
phoneme_viseme_detector_simple.py
=================================

Simple, real-time viseme detection using classic audio analysis — **no ML** required.
Works with NumPy only; SciPy is optional and not required for core detection.

Features
--------
- Streaming viseme detection from short audio chunks (e.g., 50 ms).
- Lightweight heuristics using RMS energy, FFT magnitude bands, spectral centroid,
  zero-crossing rate, and rolloff.
- Small, practical viseme set: {"AI","E","O","U","MBP","FV","SZ","L","REST"}.
- Optional WAV file processing utility for offline testing and statistics.

Dependencies
------------
- numpy (required)
- scipy (optional; imported but not required for core detection)

Usage (PowerShell)
------------------
PS> python .\phoneme_viseme_detector_simple.py

This runs a synthetic self-test that prints the detected viseme for a few test signals.

API Quick Reference
-------------------
- class PhonemeVisemeDetector:
    detect(audio_chunk: np.ndarray) -> tuple[str, float]

- class RealtimeVisemeStream:
    __init__(mask_controller, emotion: str = "neutral")
    process_wav_file(wav_path: str) -> dict[str, int]

The `mask_controller` passed to `RealtimeVisemeStream` is expected to provide:
    set_viseme_pose(viseme: str, weight: float, emotion: str) -> None
"""

from __future__ import annotations
import time
from typing import Tuple, List, Dict
import numpy as np

__author__  = "Rambod Taherian"
__email__   = "rambodt@uw.com"
__version__ = "1.0.0"
__license__ = "MIT"
__url__     = "https://github.com/rambodt/Freddie-2.0"

try:
    from scipy import signal  # Imported for potential future use; not required here.
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class PhonemeVisemeDetector:
    """
    Heuristic, non-ML viseme detector for short audio frames.

    Methodology
    -----------
    - Computes RMS energy for silence gating.
    - Uses a Hann-windowed FFT (2048 points) to obtain normalized magnitude.
    - Aggregates energy in speech-relevant bands (20–8000 Hz).
    - Derives spectral centroid, zero-crossing rate (ZCR), and 85% rolloff frequency.
    - Applies rule-based decisions to map features to a compact viseme set.
    - Applies short history smoothing to reduce rapid label switching.

    Attributes
    ----------
    sample_rate : int
        Expected sampling rate for the input chunks (Hz). Default 16000.
    prev_viseme : str
        Last emitted viseme label.
    viseme_history : list[str]
        Short list (length ≤ 3) of recent visemes for smoothing.

    Notes
    -----
    - Input chunks may be any length ≥ 256 samples. For spectral analysis, up to the
      first 2048 samples are used (zero-padding is not required here).
    - Returned confidence is in [0.0, 1.0] and reflects heuristic certainty.
    """

    def __init__(self, device: str = 'cpu'):
        """Initialize detector. `device` is accepted for API compatibility."""
        self.sample_rate: int = 16000
        self.prev_viseme: str = 'REST'
        self.viseme_history: List[str] = []
        print("[PhonemeDetector] Using audio analysis-based viseme detection (no ML required)")

    def detect(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        """
        Infer a viseme label from a short audio chunk.

        Parameters
        ----------
        audio_chunk : np.ndarray
            1-D float32/float64 waveform, typically 50–100 ms at 16 kHz.

        Returns
        -------
        (viseme, confidence) : tuple[str, float]
            viseme     → One of: {"AI","E","O","U","MBP","FV","SZ","L","REST"}.
            confidence → Heuristic confidence in [0.0, 1.0].

        Behavior
        --------
        - Frames with low RMS energy fall back to 'REST'.
        - Fricatives (S/Z/SH) correlate with higher ZCR and high-band energy.
        - Plosives (M/B/P) exhibit low RMS with relatively stronger low-mid energy.
        - Vowels are roughly distinguished by spectral shape/centroid.
        """
        # Quick length guard for extremely short buffers
        if len(audio_chunk) < 256:
            return 'REST', 0.5

        # RMS energy for silence gating
        rms = float(np.sqrt(np.mean(audio_chunk**2)))

        # Silence / near-silence
        if rms < 0.01:
            return 'REST', 0.9

        # Limit analysis to up to 2048 samples for consistent FFT resolution
        chunk = audio_chunk[:2048] if len(audio_chunk) > 2048 else audio_chunk

        # Apply Hann window to reduce spectral leakage
        window = np.hanning(len(chunk))
        windowed = chunk * window

        # Real FFT (zero-pad to 2048 points if chunk shorter)
        fft = np.fft.rfft(windowed, n=2048)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(2048, 1 / self.sample_rate)

        # Normalize magnitude to [0, 1] to make thresholds sample-scale invariant
        max_mag = np.max(magnitude)
        if max_mag > 0:
            magnitude = magnitude / max_mag
        else:
            return 'REST', 0.5

        # Helper to compute mean energy within [low_hz, high_hz]
        def band_energy(low_hz: float, high_hz: float) -> float:
            mask = (freqs >= low_hz) & (freqs <= high_hz)
            return float(np.mean(magnitude[mask])) if np.any(mask) else 0.0

        # Speech-relevant bands (approximate):
        #   20–100: sub-bass, 100–250: bass, 250–500: low-mid,
        #   500–1000: mid, 1–2 kHz: upper-mid, 2–4 kHz: presence, 4–8 kHz: brilliance
        sub_bass   = band_energy(20,   100)
        bass       = band_energy(100,  250)
        low_mid    = band_energy(250,  500)
        mid        = band_energy(500,  1000)
        upper_mid  = band_energy(1000, 2000)
        presence   = band_energy(2000, 4000)
        brilliance = band_energy(4000, 8000)

        # Spectral centroid (brightness)
        total = float(np.sum(magnitude))
        centroid = float(np.sum(freqs[:len(magnitude)] * magnitude) / total) if total > 0 else 0.0

        # Zero-crossing rate (fricative indicator)
        zcr = float(np.sum(np.abs(np.diff(np.sign(chunk)))) / (2 * len(chunk)))

        # Spectral rolloff (85% cumulative energy)
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff_freq = float(freqs[idx[0]]) if len(idx) > 0 else 0.0
        else:
            rolloff_freq = 0.0

        # Decision logic
        confidence = 0.6

        # High-frequency fricatives (S/Z/SH)
        if zcr > 0.1 and brilliance > 0.3:
            if centroid > 4000:
                viseme = 'SZ'
                confidence = min(0.8, brilliance + 0.2)
            elif centroid > 3000:
                viseme = 'FV'
                confidence = min(0.8, presence + 0.2)
            else:
                viseme = 'SZ'
                confidence = 0.6

        # Plosives / closures (MBP)
        elif rms < 0.03 and low_mid > upper_mid:
            viseme = 'MBP'
            confidence = 0.7

        # Vowel-like regions (based on broad spectral shape)
        elif rms > 0.02:
            if bass > 0.4 and low_mid > 0.3:
                # Lower formant emphasis (open/back vowels)
                if centroid < 1000:
                    viseme = 'O'   # "aw/oh"
                    confidence = 0.7
                else:
                    viseme = 'AA'  # mapped to 'O' below
                    confidence = 0.7
            elif upper_mid > 0.4 and presence > 0.2:
                # Front vowels (higher F2)
                if centroid > 2000:
                    viseme = 'AI'  # "ee/i"
                    confidence = 0.7
                else:
                    viseme = 'E'   # "eh/ay"
                    confidence = 0.7
            elif mid > 0.4:
                viseme = 'E'
                confidence = 0.6
            elif bass > 0.5 and centroid < 800:
                viseme = 'U'       # "oo/u"
                confidence = 0.7
            else:
                # Default vowel bucket
                viseme = 'E' if centroid > 1500 else 'O'
                confidence = 0.5

        # Liquid 'L' (moderate mid/upper-mid, limited presence)
        elif mid > 0.3 and upper_mid > 0.2 and presence < 0.3:
            viseme = 'L'
            confidence = 0.6

        # Fallbacks
        else:
            viseme = 'E' if rms > 0.015 else 'REST'
            confidence = 0.4 if viseme == 'E' else 0.6

        # Small label normalization to keep the set compact
        if viseme == 'AA':
            viseme = 'O'

        # Short history smoothing to dampen flicker
        self.viseme_history.append(viseme)
        if len(self.viseme_history) > 3:
            self.viseme_history.pop(0)

        # Boost confidence if the last two labels agree
        if len(self.viseme_history) >= 2:
            if self.viseme_history[-1] == self.viseme_history[-2]:
                confidence = min(1.0, confidence + 0.1)

        # Penalize rapid three-way switches
        if len(self.viseme_history) >= 3:
            if len(set(self.viseme_history[-3:])) == 3:
                confidence *= 0.8

        self.prev_viseme = viseme
        return viseme, float(confidence)


class RealtimeVisemeStream:
    """
    Real-time helper that feeds viseme detections to a mask controller.

    Parameters
    ----------
    mask_controller : object
        An object exposing: set_viseme_pose(viseme: str, weight: float, emotion: str) -> None
    emotion : str, default "neutral"
        Label forwarded with each viseme to the mask controller.

    Methods
    -------
    process_wav_file(wav_path: str) -> dict[str, int]
        Processes a WAV file in ~50 ms frames. Returns a count of frames per viseme.
    """

    def __init__(self, mask_controller, emotion: str = 'neutral'):
        self.mask = mask_controller
        self.emotion = emotion
        self.detector = PhonemeVisemeDetector()

    def process_wav_file(self, wav_path: str) -> Dict[str, int]:
        """
        Run viseme detection over a WAV file and stream poses to the mask controller.

        Frames are processed in ~50 ms chunks with a 50% overlap for analysis to
        improve temporal resolution while maintaining stable estimates.

        Parameters
        ----------
        wav_path : str
            Path to a PCM WAV file (mono or stereo; if stereo, only raw frames
            are read and interpreted as int16 PCM by NumPy).

        Returns
        -------
        viseme_stats : dict[str, int]
            A dictionary mapping viseme label → number of frames detected.

        Notes
        -----
        - Each frame’s intensity is estimated from the short-term RMS relative to
          a slowly adapting reference, then multiplied by the detector’s confidence
          to form a weight in [0, 1] for `set_viseme_pose`.
        """
        import wave

        print("[VisemeStream] Starting audio-based viseme detection...")

        try:
            wf = wave.open(wav_path, 'rb')
            rate = wf.getframerate()

            # 50 ms frames
            chunk_duration = 0.05
            chunk_frames = int(rate * chunk_duration)

            viseme_stats: Dict[str, int] = {}
            frame_count = 0
            ref_energy = 0.03  # EMA baseline for intensity normalization
            prev_chunk: np.ndarray | None = None

            while True:
                raw = wf.readframes(chunk_frames)
                if not raw:
                    break

                # Convert little-endian int16 PCM to float32 in [-1, 1]
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                # Overlapped analysis window: last half of previous + current
                if prev_chunk is not None and len(prev_chunk) > 0:
                    analysis_chunk = np.concatenate([prev_chunk[-len(prev_chunk) // 2:], audio])
                else:
                    analysis_chunk = audio

                # Estimate short-term intensity relative to a slowly moving reference
                rms = float(np.sqrt(np.mean(audio**2)))
                ref_energy = max(0.001, 0.95 * ref_energy + 0.05 * rms)
                intensity = min(1.0, rms / ref_energy) if ref_energy > 0 else 0.0

                # Detect viseme on the analysis chunk
                viseme, confidence = self.detector.detect(analysis_chunk)

                # Update stats
                viseme_stats[viseme] = viseme_stats.get(viseme, 0) + 1

                # Send to mask (weight = intensity × confidence)
                weight = float(intensity) * float(confidence)
                self.mask.set_viseme_pose(viseme, weight, self.emotion)

                frame_count += 1

                # Optional: periodic debug print (every ~200 ms)
                if frame_count % 4 == 0:
                    print(f"[Viseme] {viseme:5s} (conf: {confidence:.2f}, intensity: {intensity:.2f})")

                # Keep current chunk for next overlap
                prev_chunk = audio

                # Maintain frame cadence
                time.sleep(chunk_duration)

            wf.close()

            # Summary
            total = sum(viseme_stats.values())
            if total > 0:
                print("\n[Viseme Statistics]")
                for v, count in sorted(viseme_stats.items(), key=lambda x: x[1], reverse=True):
                    pct = 100.0 * count / total
                    print(f"  {v:5s}: {pct:5.1f}% ({count} frames)")

                variety = sum(1 for v, c in viseme_stats.items() if c > total * 0.01)
                print(f"\nViseme variety: {variety} different visemes detected")
                if variety < 4:
                    print("  Note: Low variety might indicate detection issues")

            return viseme_stats

        except Exception as e:
            print(f"[VisemeStream Error] {e}")
            import traceback
            traceback.print_exc()
            return {}


if __name__ == "__main__":
    # Self-test with synthetic signals
    print("Simple Viseme Detector (No ML Dependencies)")
    print("=" * 50)

    detector = PhonemeVisemeDetector()

    print("\nTesting with synthetic audio signals...")
    sr = 16000
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration), endpoint=False).astype(np.float32)

    # Test cases: rough vowel proxies, fricative-like noise, plosive-like burst, and silence
    test_cases = [
        (np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 2200 * t),
         "I-like (F1=300, F2=2200)"),
        (np.sin(2 * np.pi * 700 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t),
         "A-like (F1=700, F2=1200)"),
        (np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 700 * t),
         "U-like (F1=300, F2=700)"),
        (np.random.randn(int(sr * duration)).astype(np.float32) * 0.1,
         "Noise (fricative-like)"),
        (np.concatenate([
            np.zeros(int(sr * 0.02), dtype=np.float32),
            np.random.randn(int(sr * 0.02)).astype(np.float32) * 0.3,
            np.zeros(int(sr * 0.06), dtype=np.float32)
        ]),
         "Burst (plosive-like)"),
        (np.zeros(int(sr * duration), dtype=np.float32),
         "Silence"),
    ]

    for sig, desc in test_cases:
        viseme, conf = detector.detect(sig.astype(np.float32))
        print(f"  {desc:30s} -> {viseme:5s} (conf: {conf:.2f})")

    print("\nExpected visemes for speech:")
    print("  AI - 'ee', 'i' sounds (beat, see)")
    print("  E  - 'eh', 'ay' sounds (bet, say)")
    print("  O  - 'oh', 'ah' sounds (boat, hot)")
    print("  U  - 'oo' sounds (boot, food)")
    print("  MBP - m, b, p (lips closed)")
    print("  FV  - f, v (teeth on lip)")
    print("  SZ  - s, z (teeth together)")
    print("  L   - l sound")
    print("  REST - silence/pause")

    print("\nDetector ready!")
