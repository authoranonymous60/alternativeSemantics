# Data Organisation

## Directory Structure

```
data/
  stimuli/              JSON stimulus files defining sentence pairs
  speakers/
    speaker0/           Original speaker (100 items, files f1–f10)
      raw/              Full session recordings (f1–f10.wav)
      clips/            Individual item clips extracted by chunk_creation_of_any_audio.py
    speaker1/           New speaker 1 (24 items, files ns1–ns3)
      raw/              Full session recordings (ns1–3.wav)
      clips/            Individual item clips (to be extracted)
    speaker2/           New speaker 2 (24 items, files ns1–ns3)
      raw/              Full session recordings (not yet recorded)
      clips/            Individual item clips (to be extracted)
  output/               All pipeline outputs: CSVs, logs, match JSONs
```

## Stimuli

Stimulus files are JSON arrays. Each item defines a sentence pair and its structural properties:

| Field            | Description |
|------------------|-------------|
| `S1`             | First sentence, with the focus word in UPPERCASE |
| `S2`             | Second sentence (the continuation to be judged) |
| `A`              | Answer key: `A`, `B`, or `C` |
| `focus`          | `1` = recipient focused, `2` = object focused |
| `logic`          | `POS` (also) or `NEG` (didn't) |
| `alternative`    | `1` = different person / same object, `2` = same person / different object |
| `source`         | `original` or `new` (ns files only) |
| `original_file`  | Source file, e.g. `f1` (original items in ns files only) |
| `original_index` | 0-based index in source file (original items in ns files only) |

### Stimulus files

| File(s)       | Speaker(s) | Items | Description |
|---------------|------------|-------|-------------|
| `f1–f10.json` | speaker0   | 100   | Full original stimulus set |
| `ns1.json`    | speaker0, speaker1, speaker2 | 8 | Recording set 1 |
| `ns2.json`    | speaker0, speaker1, speaker2 | 8 | Recording set 2 |
| `ns3.json`    | speaker0, speaker1, speaker2 | 8 | Recording set 3 |

The ns files together form a 24-item subset, stratified across all 8 structures of the 2×2×2 design (focus × alternative × logic), drawn from the original 100 items. All three speakers record these same 24 items, enabling direct cross-speaker comparison.

## Experimental Design

The stimuli follow a **2×2×2 factorial structure**:

| Dimension     | Levels | Description |
|---------------|--------|-------------|
| `focus`       | 1 / 2  | Which constituent is focused: recipient (1) or object (2) |
| `alternative` | 1 / 2  | What S2 varies: different person (1) or same person/different object (2) |
| `logic`       | POS / NEG | S2 uses *also* (POS) or *didn't* (NEG) |

Answer key is fully determined by structure:
- `focus = alternative`, POS → **C**
- `focus = alternative`, NEG → **A**
- `focus ≠ alternative` → **B** (regardless of polarity)

## Audio

### Recording protocol

Speakers read each sentence pair naturally, placing emphasis on the uppercased word. See `speakerInstructions.txt` and `recordingProtocol.md` for full instructions.

### File naming

- Speaker0 full recordings: `f{n}.wav` (n = 1–10), one file per stimulus file
- New speaker full recordings: `ns{n}.wav` (n = 1–3), one file per stimulus file
- Extracted clips: `{basename}_item{index}.wav`, e.g. `f1_item0.wav`

### Processing pipeline

Full recordings are processed by `code/chunk_creation_of_any_audio.py`, which:
1. Transcribes the audio using Whisper
2. Aligns transcription tokens to stimulus items using forward matching
3. Extracts individual item clips with padding into `speakers/{speaker}/clips/`

## Scripts

| Script | Purpose | Key paths |
|--------|---------|-----------|
| `code/chunk_creation_of_any_audio.py` | Transcribe + align + extract clips | reads `data/stimuli/`, `data/speakers/speaker0/raw/`; writes `data/speakers/speaker0/clips/`, `data/output/` |
| `code/Task1.py` | Build stratified sets from stimulus JSON | reads `data/stimuli/`; writes `data/output/` |
| `code/audioInput.py` | Run inference pipeline (LLM evaluation) | writes `data/output/` |
