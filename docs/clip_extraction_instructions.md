# Clip Extraction Instructions

These instructions explain how to extract individual item clips from the raw recordings for all three speakers. The goal is to produce one `.wav` clip per item per speaker, covering all 24 items in the ns1–ns3 recording sets.

## Overview

The script `code/chunk_creation_of_any_audio.py` transcribes a raw recording using Whisper, aligns the transcript to the stimulus JSON, and cuts out one clip per item. You run it once per recording file per speaker.

Clips are saved to `data/speakers/{speaker}/clips/` named `{source_basename}_item{index}.wav`.

---

## Prerequisites

From the root of the repository:

```bash
pip install openai-whisper
```

ffmpeg must also be installed (`brew install ffmpeg` on Mac).

---

## Speaker 1 (3 runs)

Speaker 1's recordings cover ns1, ns2, ns3.

```bash
python code/chunk_creation_of_any_audio.py --speaker speaker1 --file ns1
python code/chunk_creation_of_any_audio.py --speaker speaker1 --file ns2
python code/chunk_creation_of_any_audio.py --speaker speaker1 --file ns3
```

---

## Speaker 2 (3 runs)

```bash
python code/chunk_creation_of_any_audio.py --speaker speaker2 --file ns1
python code/chunk_creation_of_any_audio.py --speaker speaker2 --file ns2
python code/chunk_creation_of_any_audio.py --speaker speaker2 --file ns3
```

---

## Speaker 0 (9 runs)

Speaker 0 recorded all 100 items across f1–f10. The 24 ns items are drawn from all 10 files. Run the script on each:

```bash
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f1
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f2
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f3 --model small
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f4
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f5
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f6
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f7
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f8 --model small
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f9
python code/chunk_creation_of_any_audio.py --speaker speaker0 --file f10
```

Note: f3 and f8 require `--model small` because the default `medium` model fails to fully transcribe those recordings.

This will extract clips for all items in each file. Only the clips corresponding to the 24 ns items will be used downstream — see `data/stimuli/ns1–3.json` (`original_file` and `original_index` fields) to identify which ones.

---

## Output

After all runs, clips will be in:

```
data/speakers/speaker0/clips/    e.g. f1_item0.wav, f1_item3.wav, ...
data/speakers/speaker1/clips/    e.g. ns1_item0.wav, ns1_item1.wav, ...
data/speakers/speaker2/clips/    e.g. ns1_item0.wav, ns1_item1.wav, ...
```

## Notes

- Each run takes a few minutes (Whisper transcription). The `medium` model is used by default.
- If a clip is not found (low alignment score), a warning is printed and that item is skipped. Check the output JSON in `data/output/json_world_matches_transcript/` to diagnose any failures.
- Run all commands from the **root of the repository**.
- After extraction, speaker0 clips should be copied to `ns{X}_item{Y}.wav` names (matching speakers 1 & 2) using the `original_file`/`original_index` mappings in the ns JSON files. This ensures uniform naming across all speakers for downstream scripts.
