import os
import re
import json
import argparse
import whisper
import subprocess
from difflib import SequenceMatcher, get_close_matches

# =========================================================
# CONFIG
# =========================================================

parser = argparse.ArgumentParser(description="Transcribe, align and extract clips from a speaker recording.")
parser.add_argument("--speaker", required=True, help="Speaker ID, e.g. speaker0, speaker1, speaker2")
parser.add_argument("--file", required=True, help="Source basename, e.g. f1, ns1")
parser.add_argument("--model", default="medium", help="Whisper model size (default: medium)")
parser.add_argument("--no-cut", action="store_true", help="Skip clip extraction")
args = parser.parse_args()

SOURCE_BASENAME = args.file
SPEAKER = args.speaker

AUDIO_FILE = f"data/speakers/{SPEAKER}/raw/{SOURCE_BASENAME}.wav"
JSON_FILE = f"data/stimuli/{SOURCE_BASENAME}.json"

OUTPUT_RESULTS = f"data/output/json_world_matches_transcript/{SPEAKER}_{SOURCE_BASENAME}_vocab_forward_matches.json"
OUTPUT_CLIPS_DIR = f"data/speakers/{SPEAKER}/clips"

WHISPER_MODEL = args.model
CUT_CLIPS = not args.no_cut

START_PADDING = 0.03
END_PADDING = 0.30

WINDOW_MARGIN = 3
MAX_FORWARD_WORDS = 100
MIN_SCORE = 0.88

# zárt szókészlet
VOCAB = [
    "sam", "only", "gave", "give", "also", "didnt",
    "rob", "sue", "mary", "ellen", "tom", "bill",
    "apples", "oranges", "bananas", "grapes"
]

RECIPIENTS = {"rob", "sue", "mary", "ellen", "tom", "bill"}
FRUITS = {"apples", "oranges", "bananas", "grapes"}
LOGICS = {"also", "didnt"}

# kézi javítások tipikus whisper félrehallásokra
MANUAL_MAP = {
    "didn't": "didnt",
    "didnt": "didnt",
    "did": "didnt",
    "married": "mary",
    "merry": "mary",
    "ellenn": "ellen",
    "grape": "grapes",
    "banana": "bananas",
    "orange": "oranges",
    "apple": "apples",
}

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


def load_items(json_file):
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


# =========================================================
# VOCAB SNAPPING
# =========================================================

def snap_token(token: str):
    token = normalize_text(token)
    if not token:
        return None

    if token in MANUAL_MAP:
        return MANUAL_MAP[token]

    if token in VOCAB:
        return token

    # közeli egyezés a zárt vocabhoz
    matches = get_close_matches(token, VOCAB, n=1, cutoff=0.75)
    if matches:
        return matches[0]

    return None


def transcribe_words(audio_file):
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(
        audio_file,
        word_timestamps=True,
        verbose=False
    )

    words = []

    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            raw = w.get("word", "").strip()
            if not raw:
                continue

            raw_norm = normalize_text(raw)
            if not raw_norm:
                continue

            snapped = snap_token(raw_norm)

            # ha nem tudjuk értelmes vocab tokenre húzni, eldobjuk
            if snapped is None:
                continue

            words.append({
                "raw": raw,
                "norm": raw_norm,
                "snapped": snapped,
                "start": w["start"],
                "end": w["end"],
            })

    return words


def join_snapped(words, start_idx, end_idx):
    return " ".join(w["snapped"] for w in words[start_idx:end_idx + 1])


# =========================================================
# SCORING
# =========================================================

def contains_token(candidate_tokens, token):
    return token is not None and token in candidate_tokens


def score_candidate(item, candidate_tokens):
    candidate_text = " ".join(candidate_tokens)
    text_score = SequenceMatcher(None, item["target_norm"], candidate_text).ratio()

    s1_rec_ok = contains_token(candidate_tokens, item["s1_recipient"])
    s1_fruit_ok = contains_token(candidate_tokens, item["s1_fruit"])
    s2_rec_ok = contains_token(candidate_tokens, item["s2_recipient"])
    s2_fruit_ok = contains_token(candidate_tokens, item["s2_fruit"])
    logic_ok = contains_token(candidate_tokens, item["logic"])

    keyword_hits = sum([s1_rec_ok, s1_fruit_ok, s2_rec_ok, s2_fruit_ok, logic_ok])
    keyword_score = keyword_hits / 5.0

    # content words fontosabbak, mint a sima string hasonlóság
    final_score = 0.35 * text_score + 0.65 * keyword_score

    debug = {
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

    return final_score, debug


# =========================================================
# MATCHING
# =========================================================

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
            if end_idx >= len(words):
                continue
            if end_idx > search_end_idx:
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

        found = best is not None and best["score"] >= MIN_SCORE

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

        print(f"Item {item['item_index']}: found={found} score={result['final_score']}")
        if found and best["score"] >= 0.90:
            print(f"  start={result['start_time']} end={result['end_time']}")
            print(f"  matched={result['matched_text']}")
            current_idx = best["end_idx"] + 1
        else:
            print("  no reliable forward match")
        print()

    return results


# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Processing {SOURCE_BASENAME}")
    print("Loading JSON...")
    items = load_items(JSON_FILE)

    print(f"Transcribing with Whisper model: {WHISPER_MODEL}")
    words = transcribe_words(AUDIO_FILE)

    print(f"Snapped words kept: {len(words)}")
    print(f"Items in JSON: {len(items)}\n")

    results = match_items_forward(items, words)

    os.makedirs(os.path.dirname(OUTPUT_RESULTS), exist_ok=True)
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {OUTPUT_RESULTS}")

    if CUT_CLIPS:
        total_duration = get_audio_duration(AUDIO_FILE)
        print("\nCutting clips...")

        for r in results:
            if not r["found"]:
                continue

            start = max(0, r["start_time"] - START_PADDING)
            end = min(total_duration, r["end_time"] + END_PADDING)

            output_file = os.path.join(
                OUTPUT_CLIPS_DIR,
                f"{SOURCE_BASENAME}_item{r['item_index']}.wav"
            )

            cut_clip(AUDIO_FILE, start, end, output_file)

        print(f"Saved clips to: {OUTPUT_CLIPS_DIR}")


if __name__ == "__main__":
    main()