"""
speaker_acoustics.py

Clip-level acoustic comparison across speakers using ns1-ns3 stimuli
(the only set recorded by all three speakers).

Measures per clip:
  - duration (seconds)
  - F0 range (max - min of voiced frames, in Hz)
  - F0 std (standard deviation of voiced frames, in Hz)
  - intensity range (max - min, in dB)
  - intensity std

Results are broken down by speaker and focus position (1 vs 2).

Usage:
  python3 code/speaker_acoustics.py
"""

import json
import os
import statistics

import parselmouth
from parselmouth.praat import call

STIMULI_DIR = "data/stimuli"
CLIPS_BASE  = "data/speakers"
SPEAKERS    = ["speaker0", "speaker1", "speaker2"]
NS_FILES    = ["ns1", "ns2", "ns3"]


def extract_acoustics(wav_path: str) -> dict:
    snd = parselmouth.Sound(wav_path)
    duration = snd.duration

    # F0 via autocorrelation, typical speech range
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    f0_values = [
        pitch.get_value_at_time(t)
        for t in [pitch.get_time_from_frame_number(i)
                  for i in range(1, pitch.get_number_of_frames() + 1)]
    ]
    voiced = [v for v in f0_values if v and not (v != v)]  # drop NaN/unvoiced

    f0_range = (max(voiced) - min(voiced)) if len(voiced) >= 2 else None
    f0_std   = statistics.stdev(voiced)    if len(voiced) >= 2 else None
    f0_mean  = statistics.mean(voiced)     if voiced else None

    # Intensity
    intensity = call(snd, "To Intensity", 75, 0.0)
    n_frames  = call(intensity, "Get number of frames")
    int_values = [
        call(intensity, "Get value in frame", i)
        for i in range(1, n_frames + 1)
    ]
    int_values = [v for v in int_values if v and not (v != v)]

    int_range = (max(int_values) - min(int_values)) if len(int_values) >= 2 else None
    int_std   = statistics.stdev(int_values)         if len(int_values) >= 2 else None

    return {
        "duration":    duration,
        "f0_mean":     f0_mean,
        "f0_range":    f0_range,
        "f0_std":      f0_std,
        "int_range":   int_range,
        "int_std":     int_std,
        "n_voiced":    len(voiced),
    }


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def fmt(v, decimals=1):
    return f"{v:.{decimals}f}" if v is not None else "—"


def main():
    # Load all ns metadata
    items_meta = {}  # (ns_file, item_index) -> focus position
    for ns in NS_FILES:
        path = os.path.join(STIMULI_DIR, f"{ns}.json")
        with open(path) as f:
            items = json.load(f)
        for i, item in enumerate(items):
            items_meta[(ns, i)] = item.get("focus")  # 1 or 2

    # Collect acoustic measurements per speaker
    all_rows = []
    for speaker in SPEAKERS:
        clips_dir = os.path.join(CLIPS_BASE, speaker, "clips")
        for ns in NS_FILES:
            meta_keys = [(ns, i) for i in range(len([k for k in items_meta if k[0] == ns]))]
            for ns_file, item_idx in meta_keys:
                wav_name = f"{ns_file}_item{item_idx}.wav"
                wav_path = os.path.join(clips_dir, wav_name)
                if not os.path.exists(wav_path):
                    print(f"  Missing: {wav_path}")
                    continue
                focus = items_meta[(ns_file, item_idx)]
                try:
                    acoustics = extract_acoustics(wav_path)
                    acoustics.update({
                        "speaker": speaker,
                        "ns_file": ns_file,
                        "item_idx": item_idx,
                        "focus": focus,
                    })
                    all_rows.append(acoustics)
                except Exception as e:
                    print(f"  Error processing {wav_path}: {e}")

    # ── Summary by speaker ──────────────────────────────────────────
    print("\n=== ACOUSTIC SUMMARY BY SPEAKER (ns1–ns3 clips) ===\n")
    print(f"{'Speaker':<12} {'N':>4}  {'Duration':>10}  {'F0 mean':>9}  {'F0 range':>10}  {'F0 std':>8}  {'Int range':>10}  {'Int std':>8}")
    print("-" * 85)
    for speaker in SPEAKERS:
        rows = [r for r in all_rows if r["speaker"] == speaker]
        print(f"{speaker:<12} {len(rows):>4}  "
              f"{fmt(mean_or_none([r['duration']   for r in rows]), 2):>10}s  "
              f"{fmt(mean_or_none([r['f0_mean']    for r in rows])):>9}Hz  "
              f"{fmt(mean_or_none([r['f0_range']   for r in rows])):>10}Hz  "
              f"{fmt(mean_or_none([r['f0_std']     for r in rows])):>8}Hz  "
              f"{fmt(mean_or_none([r['int_range']  for r in rows])):>10}dB  "
              f"{fmt(mean_or_none([r['int_std']    for r in rows])):>8}dB")

    # ── Summary by speaker × focus position ────────────────────────
    print("\n=== BY SPEAKER × FOCUS POSITION ===\n")
    print(f"{'Speaker':<12} {'Focus':>6} {'N':>4}  {'F0 range':>10}  {'F0 std':>8}  {'Duration':>10}")
    print("-" * 60)
    for speaker in SPEAKERS:
        for focus_pos in [1, 2]:
            rows = [r for r in all_rows if r["speaker"] == speaker and r["focus"] == focus_pos]
            print(f"{speaker:<12} {focus_pos:>6} {len(rows):>4}  "
                  f"{fmt(mean_or_none([r['f0_range'] for r in rows])):>10}Hz  "
                  f"{fmt(mean_or_none([r['f0_std']   for r in rows])):>8}Hz  "
                  f"{fmt(mean_or_none([r['duration'] for r in rows]), 2):>10}s")
        print()


if __name__ == "__main__":
    main()
