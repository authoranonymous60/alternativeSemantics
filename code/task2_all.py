import os
import re
import json
import whisper
import subprocess
from datetime import datetime
from difflib import SequenceMatcher, get_close_matches

# =========================================================
# CONFIG
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

SET1_FILE = os.path.join(REPO_ROOT, "data", "output_milan", "set1.json")
SET2_FILE = os.path.join(REPO_ROOT, "data", "output_milan", "set2.json")

INPUT_DIR = os.path.join(REPO_ROOT, "data", "input")
OUTPUT_MATCH_DIR = os.path.join(REPO_ROOT, "data", "output_milan", "matches")
OUTPUT_CLIPS_BASE = os.path.join(REPO_ROOT, "data", "clips")
LOG_DIR = os.path.join(REPO_ROOT, "data", "output_milan", "logs")

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"task2_all_{RUN_TIMESTAMP}.log")

WHISPER_MODEL = "medium"   # ha túl lassú: "small"
START_PADDING = 0.03
END_PADDING = 0.30

WINDOW_MARGIN = 3
MAX_FORWARD_WORDS = 100
MIN_SCORE = 0.88

VOCAB = [
    "sam", "only", "gave", "give", "also", "didnt",
    "rob", "sue", "mary", "ellen", "tom", "bill",
    "apples", "oranges", "bananas", "grapes"
]

MANUAL_MAP = {
    "didn't": "didnt",
    "didnt": "didnt",
    "did": "didnt",
    "married": "mary",
    "merry": "mary",
    "alan": "ellen",
    "helen": "ellen",
    "robb": "rob",
    "bell": "bill",
    "thom": "tom",
    "grape": "grapes",
    "banana": "bananas",
    "orange": "oranges",
    "apple": "apples",
}


def log(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# =========================================================
# HELPERS
# =========================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("didn't", "didnt")
    text = text.replace("did not", "didnt")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    t = normalize_text(text)
    return t.split() if t else []


def parse_s1(s1: str):
    toks = tokenize(s1)
    if len(toks) < 5:
        return None, None
    return toks[3], toks[4]


def parse_s2(s2: str):
    toks = tokenize(s2)
    if len(toks) < 5:
        return None, None, None
    logic = "didnt" if "didnt" in toks else "also" if "also" in toks else None
    return logic, toks[3], toks[4]


def get_audio_duration(audio_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def cut_clip(input_file, start, end, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ss", str(start),
        "-to", str(end),
        output_file,
    ]
    subprocess.run(cmd, check=True)


def snap_token(token: str):
    token = normalize_text(token)
    if not token:
        return None

    if token in MANUAL_MAP:
        return MANUAL_MAP[token]

    if token in VOCAB:
        return token

    matches = get_close_matches(token, VOCAB, n=1, cutoff=0.75)
    if matches:
        return matches[0]

    return None


# =========================================================
# SET LOADING
# =========================================================

def load_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_required_map():
    """
    Returns:
    {
      "f2": {"set1": {6}, "set2": {3}},
      "f3": {"set1": {0,4}, "set2": set()}
    }
    """
    required = {}

    for set_name, path in [("set1", SET1_FILE), ("set2", SET2_FILE)]:
        data = load_set(path)

        for item in data:
            source = item["source_file"].replace(".json", "")
            idx = item["item_index"]

            if source not in required:
                required[source] = {"set1": set(), "set2": set()}

            required[source][set_name].add(idx)

    return required


# =========================================================
# SOURCE FILE PROCESSING
# =========================================================

def load_source_items(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    for idx, item in enumerate(raw):
        s1_rec, s1_fruit = parse_s1(item["S1"])
        logic, s2_rec, s2_fruit = parse_s2(item["S2"])

        target_text = f"{item['S1']} {item['S2']}"
        target_norm = normalize_text(target_text)

        items.append({
            "item_index": idx,
            "S1": item["S1"],
            "S2": item["S2"],
            "target_text": target_text,
            "target_norm": target_norm,
            "target_tokens": target_norm.split(),
            "s1_recipient": s1_rec,
            "s1_fruit": s1_fruit,
            "logic": logic,
            "s2_recipient": s2_rec,
            "s2_fruit": s2_fruit,
        })

    return items


def transcribe_words(audio_file, model):
    # Keep decoding deterministic and CPU-safe across runs.
    # - language/task pin Whisper to English transcription mode
    # - temperature=0 removes sampling randomness
    # - fp16=False avoids CPU fp16 warnings/fallback behavior
    result = model.transcribe(
        audio_file,
        word_timestamps=True,
        language="en",
        task="transcribe",
        temperature=0,
        fp16=False,
        verbose=False
    )

    words = []
    raw_word_count = 0
    normalized_word_count = 0
    dropped_unsnapped = 0

    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            raw = w.get("word", "").strip()
            if not raw:
                continue
            raw_word_count += 1

            raw_norm = normalize_text(raw)
            if not raw_norm:
                continue
            normalized_word_count += 1

            snapped = snap_token(raw_norm)
            if snapped is None:
                dropped_unsnapped += 1
                continue

            words.append({
                "raw": raw,
                "norm": raw_norm,
                "snapped": snapped,
                "start": w["start"],
                "end": w["end"],
            })

    diagnostics = {
        "raw_word_count": raw_word_count,
        "normalized_word_count": normalized_word_count,
        "kept_word_count": len(words),
        "dropped_unsnapped": dropped_unsnapped,
        "last_word_end_time": words[-1]["end"] if words else None,
    }

    return words, diagnostics


def score_candidate(item, candidate_tokens):
    candidate_text = " ".join(candidate_tokens)
    text_score = SequenceMatcher(None, item["target_norm"], candidate_text).ratio()

    s1_rec_ok = item["s1_recipient"] in candidate_tokens if item["s1_recipient"] else False
    s1_fruit_ok = item["s1_fruit"] in candidate_tokens if item["s1_fruit"] else False
    s2_rec_ok = item["s2_recipient"] in candidate_tokens if item["s2_recipient"] else False
    s2_fruit_ok = item["s2_fruit"] in candidate_tokens if item["s2_fruit"] else False
    logic_ok = item["logic"] in candidate_tokens if item["logic"] else False

    keyword_hits = sum([s1_rec_ok, s1_fruit_ok, s2_rec_ok, s2_fruit_ok, logic_ok])
    keyword_score = keyword_hits / 5.0

    final_score = 0.35 * text_score + 0.65 * keyword_score

    return final_score, {
        "text_score": round(text_score, 4),
        "keyword_score": round(keyword_score, 4),
        "keyword_hits": keyword_hits,
        "s1_rec_ok": s1_rec_ok,
        "s1_fruit_ok": s1_fruit_ok,
        "s2_rec_ok": s2_rec_ok,
        "s2_fruit_ok": s2_fruit_ok,
        "logic_ok": logic_ok,
        "final_score": round(final_score, 4),
    }


def find_best_match_forward(item, words, search_start_idx):
    target_len = len(item["target_tokens"])
    if target_len == 0:
        return None

    search_end_idx = min(len(words) - 1, search_start_idx + MAX_FORWARD_WORDS)
    best = None

    min_len = max(6, target_len - WINDOW_MARGIN)
    max_len = target_len + WINDOW_MARGIN

    for start_idx in range(search_start_idx, search_end_idx + 1):
        for cand_len in range(min_len, max_len + 1):
            end_idx = start_idx + cand_len - 1
            if end_idx >= len(words) or end_idx > search_end_idx:
                continue

            candidate_tokens = [words[k]["snapped"] for k in range(start_idx, end_idx + 1)]
            score, debug = score_candidate(item, candidate_tokens)

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "debug": debug,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": words[start_idx]["start"],
                    "end_time": words[end_idx]["end"],
                    "matched_text": " ".join(candidate_tokens),
                }

    return best


def match_items_forward(items, words):
    results = []
    current_idx = 0

    for item in items:
        best = find_best_match_forward(item, words, current_idx)
        found = (
            best is not None and (
                best["score"] >= MIN_SCORE
                or (
                    best["score"] >= 0.83
                    and best["debug"]["keyword_hits"] >= 4
                    and best["debug"]["logic_ok"]
                )
            )
        )

        result = {
            "item_index": item["item_index"],
            "S1": item["S1"],
            "S2": item["S2"],
            "target_text": item["target_text"],
            "found": found,
            "final_score": round(best["score"], 4) if best else None,
            "start_time": round(best["start_time"], 3) if best else None,
            "end_time": round(best["end_time"], 3) if best else None,
            "matched_text": best["matched_text"] if best else None,
            "word_start_idx": best["start_idx"] if best else None,
            "word_end_idx": best["end_idx"] if best else None,
            "debug": best["debug"] if best else None,
        }

        results.append(result)

        if found and best["debug"]["keyword_hits"] >= 5:
            current_idx = best["end_idx"] + 1

    return results


def process_source_file(source_basename, needed_for_sets, model):
    audio_file_m4a = os.path.join(INPUT_DIR, f"{source_basename}.m4a")
    audio_file_wav = os.path.join(INPUT_DIR, f"{source_basename}.wav")
    json_file = os.path.join(INPUT_DIR, f"{source_basename}.json")

    if os.path.exists(audio_file_m4a):
        audio_file = audio_file_m4a
        audio_kind = "m4a"
    elif os.path.exists(audio_file_wav):
        audio_file = audio_file_wav
        audio_kind = "wav"
    else:
        log(f"[SKIP] Missing audio: {audio_file_m4a} and {audio_file_wav}")
        return

    if not os.path.exists(json_file):
        log(f"[SKIP] Missing json: {json_file}")
        return

    log(f"=== Processing {source_basename} ===")
    log(f"Audio source: {audio_file} ({audio_kind})")
    log(f"Needed in set1: {sorted(needed_for_sets['set1'])}")
    log(f"Needed in set2: {sorted(needed_for_sets['set2'])}")

    items = load_source_items(json_file)
    words, diag = transcribe_words(audio_file, model)
    total_duration = get_audio_duration(audio_file)

    # If m4a transcription is suspiciously short, retry with wav when available.
    if (
        audio_kind == "m4a"
        and os.path.exists(audio_file_wav)
        and (
            not words
            or diag["last_word_end_time"] is None
            or diag["last_word_end_time"] < 0.75 * total_duration
        )
    ):
        log(
            "[WARN] m4a transcript coverage looks short; retrying with wav "
            f"({diag['last_word_end_time']}s of {total_duration:.2f}s)."
        )
        audio_file = audio_file_wav
        audio_kind = "wav"
        words, diag = transcribe_words(audio_file, model)
        total_duration = get_audio_duration(audio_file)

    last_end = diag["last_word_end_time"]
    last_end_str = f"{last_end:.2f}s" if last_end is not None else "None"
    log(
        "Transcription diagnostics: "
        f"kept={diag['kept_word_count']}/{diag['normalized_word_count']} "
        f"(raw={diag['raw_word_count']}), "
        f"dropped_unsnapped={diag['dropped_unsnapped']}, "
        f"last_word_end={last_end_str}, "
        f"audio_duration={total_duration:.2f}s"
    )

    results = match_items_forward(items, words)

    os.makedirs(OUTPUT_MATCH_DIR, exist_ok=True)
    match_out = os.path.join(OUTPUT_MATCH_DIR, f"{source_basename}_matches.json")
    with open(match_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for result in results:
        if not result["found"]:
            continue

        item_idx = result["item_index"]

        for set_name in ("set1", "set2"):
            if item_idx not in needed_for_sets[set_name]:
                continue

            start = max(0, result["start_time"] - START_PADDING)
            end = min(total_duration, result["end_time"] + END_PADDING)

            output_file = os.path.join(
                OUTPUT_CLIPS_BASE,
                set_name,
                f"{source_basename}_item{item_idx}.wav"
            )

            log(f"Cutting {set_name}: {source_basename}_item{item_idx}.wav")
            cut_clip(audio_file, start, end, output_file)

    log(f"Saved match file: {match_out}")


# =========================================================
# MAIN
# =========================================================

def main():
    required = build_required_map()
    model = whisper.load_model(WHISPER_MODEL)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    log(f"Task2 run started. Log file: {LOG_FILE}")
    log(f"Whisper model: {WHISPER_MODEL}")
    log("Required source files:")
    for source, sets in sorted(required.items()):
        log(
            f"  {source}: "
            f"set1={sorted(sets['set1'])}, "
            f"set2={sorted(sets['set2'])}"
        )

    for source_basename, needed_for_sets in sorted(required.items()):
        process_source_file(source_basename, needed_for_sets, model)

    log("Done.")


if __name__ == "__main__":
    main()