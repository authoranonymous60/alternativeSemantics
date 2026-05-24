# audioInput.py
#
# Examples:
#   python audioInput.py --backend openai --model gpt-audio --mode audio f1 f2 f3
#   python audioInput.py --backend openai --model gpt-audio --mode audio f11 f12 f13 --fewshot 0
#   python audioInput.py --backend gemini --model gemini-2.0-flash --mode audio f1 --fewshot 2 --cv
#
# WAV files are looked up in --wav-dir (default: data/speakers/speaker0/raw)
# JSON files are looked up in --json-dir (default: data/stimuli)
# Total number of examples is determined automatically from each JSON file.
#
# Notes:
# - Few-shot blocks are OPTIONAL for coverage (we warn if missing).
# - Test blocks are REQUIRED for coverage (we retry if missing).
# - In --cv mode, we rotate which indices are few-shot. For each fold, "test" = all other indices.

import argparse
import base64
import csv
import json
import os
import re
from datetime import datetime

import openai
import google.generativeai as genai
from openai import OpenAI

# ---------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------
CSV_COLUMNS = [
    # Identity & condition
    "example_index",
    "is_few_shot",

    # Gold
    "true_S1",
    "true_S2",
    "true_A",

    # Inference
    "inf_correct",
    "model_A",

    # Focus evaluation
    "trans_correct",

    # Model
    "model_S1",
    "model_S2",
    "model_explanation",

    "s1_edit_norm",
    "s2_edit_norm",

    # Linguistic features
    "focus",
    "logic",
    "alternative",

    # Metadata and condition
    "file_id",
    "mode",
    "backend",
    "model_name",
    "run_timestamp_utc",
    "response_id",
    "resolved_model",

    # NEW (optional): which CV fold produced this row (empty string if not --cv)
    "cv_fold",

    "speaker",
]

# ---------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------
def load_audio(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_examples(json_path):
    with open(json_path, "r") as f:
        examples = json.load(f)

    # Assign stable sequential indices 0..N-1
    for new_idx, ex in enumerate(examples):
        ex["idx"] = new_idx
    return examples


# -----------------------------
# Baseline helpers (no accent)
# -----------------------------
NAMES = {"mary", "ellen", "sue", "tom", "rob", "bill"}  # extend if needed


def normalize_for_edit(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[.,!?]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[-1][-1]


def normalized_edit_distance(gold: str, pred: str) -> float:
    gold_n = normalize_for_edit(gold)
    pred_n = normalize_for_edit(pred)
    dist = levenshtein(gold_n, pred_n)
    denom = max(len(gold_n), len(pred_n), 1)
    return dist / denom


def remove_focus_from_S1(S1: str) -> str:
    tokens = re.findall(r"\b\w+\b|\S", S1)

    new_tokens = []
    for tok in tokens:
        if tok.isalpha() and tok.isupper():
            lower = tok.lower()
            if lower in NAMES:
                tok = lower.capitalize()
            else:
                tok = lower
        new_tokens.append(tok)

    cleaned = ""
    for i, tok in enumerate(new_tokens):
        if i > 0 and tok.isalnum():
            cleaned += " "
        cleaned += tok
    return cleaned


# ---------------------------------------------------------------
# Prompt blocks
# ---------------------------------------------------------------
TASK_BLOCK_AUDIO = """
Your task has three parts:
1. Transcribe S1 and S2 from audio.
2. Mark prosodic focus in S1 using UPPERCASE.
3. Classify S2 relative to S1:
   A = entailed
   B = independent
   C = contradicted.

IMPORTANT:
- The audio for ALL examples is ALREADY INCLUDED with this message.
- You must NOT ask for audio or wait for additional input.
- You must infer S1 and S2 ONLY from the provided audio.
"""

TASK_BLOCK_INFERENCE = """
Your task:
1. Listen to S1 and S2 from audio.
2. Classify S2 relative to S1:
   A = entailed
   B = independent
   C = contradicted.

IMPORTANT:
- The audio for ALL examples is ALREADY INCLUDED with this message.
- You must NOT ask for audio or wait for additional input.
- Pay close attention to which word is prosodically focused in S1.
"""

TASK_BLOCK_TRANSCRIPTION = """
Your task:
1. Transcribe S1 and S2 from audio.
2. Mark the prosodically focused word in S1 using UPPERCASE. All other words must be in normal casing.

IMPORTANT:
- The audio for ALL examples is ALREADY INCLUDED with this message.
- You must NOT ask for audio or wait for additional input.
- You must infer S1 and S2 ONLY from the provided audio.
"""

TASK_BLOCK_TEXT = """
You are given text for S1 and S2.
Your task is to classify S2 relative to S1:
   A = entailed
   B = independent
   C = contradicted.
"""

TASK_BLOCK_ORACLE = """
You are given text for S1 and S2.
The prosodically focused word in S1 is marked with UPPERCASE. Use this focus marking to determine the correct inference.
Your task is to classify S2 relative to S1:
   A = entailed
   B = independent
   C = contradicted.
"""

TASK_BLOCK_TEXT_FOCUS = """
You are given text for S1 and S2.
Your task is to identify the prosodically focused word in S1 and mark it with UPPERCASE. All other words must be in normal casing.
"""

FOCUS_HINT_TEXT = """
---------------------------------------------------------------
FOCUS GUIDANCE
---------------------------------------------------------------
The classification depends on the focused element in S1, because of
the presence of 'only', in the following way: 'Sam only gave TOM
oranges' entails that Sam did not give anyone else oranges. On the
other hand, 'Sam only gave Tom ORANGES' entails that Sam didn't give
anything else to Tom.

You must follow this logic in determining the inference. You must
also refer to this logic in producing the explanation.
"""

OUTPUT_FORMAT_BOTH = """<index>
S1: ...
S2: ...
A
Because <explanation>"""

OUTPUT_FORMAT_INFERENCE = """<index>
A
Because <explanation>"""

OUTPUT_FORMAT_TRANSCRIPTION = """<index>
S1: ...
S2: ..."""


def build_base_prompt(task_block: str, fewshot_text: str, focus_block: str, new_item_block: str,
                      output_format: str = OUTPUT_FORMAT_BOTH) -> str:
    sep = "---------------------------------------------------------------"

    pre = f"{fewshot_text}\n{sep}\n" if fewshot_text else f"{sep}\n"

    return f"""
You are performing a semantic classification task.

{task_block}

{focus_block}

Your output must follow this structure:

{output_format}

Do not add meta-comments or tool-use descriptions.

{pre}TEST ITEMS
{sep}
<BEGIN_NEW>
{new_item_block}
<END_NEW>

Begin now.
"""


def make_new_item_block_audio(examples, fewshot_indices=None):
    fewshot_set = set(fewshot_indices or [])
    block = ""
    for ex in examples:
        if ex["idx"] in fewshot_set:
            continue
        block += f"{ex['idx']}\nS1:\nS2:\nA\nBecause...\n\n"
    return block


def make_new_item_block_inference(examples, fewshot_indices=None):
    fewshot_set = set(fewshot_indices or [])
    block = ""
    for ex in examples:
        if ex["idx"] in fewshot_set:
            continue
        block += f"{ex['idx']}\nA\nBecause...\n\n"
    return block


def make_new_item_block_transcription(examples, fewshot_indices=None):
    fewshot_set = set(fewshot_indices or [])
    block = ""
    for ex in examples:
        if ex["idx"] in fewshot_set:
            continue
        block += f"{ex['idx']}\nS1:\nS2:\n\n"
    return block


def make_new_item_block_text(examples, clean=False):
    block = ""
    for ex in examples:
        S1 = remove_focus_from_S1(ex["S1"]) if clean else ex["S1"]
        block += f"{ex['idx']}\nS1: {S1}\nS2: {ex['S2']}\nA\nBecause...\n\n"
    return block


def make_new_item_block_text_focus(examples):
    block = ""
    for ex in examples:
        S1 = remove_focus_from_S1(ex["S1"])
        block += f"{ex['idx']}\nS1: {S1}\nS2: {ex['S2']}\n\n"
    return block


def make_fewshot_item_block(fewshot_examples, task: str = "both", style: str = "simple"):
    """Demonstration examples shown before the test items.

    style only affects the inference task:
      simple   - label only (no S1/S2 text)
      compound - S1 uppercase text + label (reveals focus)
    """
    n = len(fewshot_examples)
    if n == 0:
        return ""

    block = (
        f"The following {n} example{'s' if n > 1 else ''} "
        f"show{'s' if n == 1 else ''} the correct answer format:\n\n"
    )

    for i, ex in enumerate(fewshot_examples, start=1):
        block += f"Example {i}\n"
        if task == "transcription":
            block += (
                f"S1: {ex['S1']}\n"
                f"S2: {ex['S2']}\n\n"
            )
        elif task == "inference":
            if style == "compound":
                block += (
                    f"S1: {ex['S1']}\n"
                    f"S2: {ex['S2']}\n"
                    f"{ex['A']}\n"
                    f"Because...\n\n"
                )
            else:
                # simple: label only — showing S1 uppercase would reveal focus
                block += (
                    f"{ex['A']}\n"
                    f"Because...\n\n"
                )
        else:  # both
            block += (
                f"S1: {ex['S1']}\n"
                f"S2: {ex['S2']}\n"
                f"{ex['A']}\n"
                f"Because...\n\n"
            )
    return block


# ---------------------------------------------------------------
# Model callers
# ---------------------------------------------------------------
def call_gemini(prompt, encoded_audio, model_name):
    import time
    from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded
    model = genai.GenerativeModel(model_name)
    contents = [
        {"text": prompt},
        {"mime_type": "audio/wav", "data": base64.b64decode(encoded_audio)},
    ]
    for attempt in range(8):
        try:
            response = model.generate_content(contents, generation_config={"temperature": 0.0})
            return response.text or ""
        except ResourceExhausted:
            wait = 30 * (2 ** attempt)
            print(f"  ⚠ Gemini 429 rate limit — waiting {wait}s before retry {attempt + 1}/8...")
            time.sleep(wait)
        except DeadlineExceeded:
            wait = 30 * (2 ** attempt)
            print(f"  ⚠ Gemini 504 timeout — waiting {wait}s before retry {attempt + 1}/8...")
            time.sleep(wait)
    raise RuntimeError(f"Gemini API not resolved after 8 retries for model {model_name}")


client = OpenAI()

def call_openai(prompt, encoded_audio, model):
    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": encoded_audio, "format": "wav"}},
                ],
            }
        ],
    )


def extract_output_text(resp):
    return resp.choices[0].message.content if resp and resp.choices else ""


def run_text_model(prompt, model):
    completion = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    msg = completion.choices[0].message
    if hasattr(msg, "content") and isinstance(msg.content, str):
        return msg.content
    if hasattr(msg, "content"):
        return "".join(part.text for part in msg.content if hasattr(part, "text") and part.text)
    return ""


# ---------------------------------------------------------------
# Focus position (LAST TWO TOKENS)
# ---------------------------------------------------------------
FOCUS_NONE   = 0
FOCUS_FIRST  = 1
FOCUS_SECOND = 2
FOCUS_BOTH   = 3

FOCUS_LABELS = {
    FOCUS_NONE: "focus_none",
    FOCUS_FIRST: "focus_first",
    FOCUS_SECOND: "focus_second",
    FOCUS_BOTH: "focus_both",
}

def focus_position(sentence):
    def is_focused(tok):
        return tok.isupper()

    tokens = re.findall(r"\b\w+\b", sentence)
    if len(tokens) < 2:
        return FOCUS_NONE

    last = is_focused(tokens[-1])
    second_last = is_focused(tokens[-2])

    if last and second_last:
        return FOCUS_BOTH
    elif second_last:
        return FOCUS_FIRST
    elif last:
        return FOCUS_SECOND
    else:
        return FOCUS_NONE


# ---------------------------------------------------------------
# Output parsing + coverage validation
# ---------------------------------------------------------------
MAX_RETRIES = 4  # number of retries *after* the first attempt


def _strip_leading(s):
    s = re.sub(r"^\s*\d+[\.\):]\s*", "", s).strip()
    s = re.sub(r"^\s*(S1|S2|A)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^\s*Because\s*:? ?", "", s, flags=re.IGNORECASE).strip()
    return s


def _prep_lines(output_text):
    lines = [ln.rstrip("\n") for ln in output_text.splitlines()]
    return [ln for ln in lines if ln.strip() != ""]


def _split_blocks_both(lines):
    """Parse 5-line blocks: index, S1, S2, A, explanation."""
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        nums = re.findall(r"\d+", lines[i].strip())
        if nums and i + 4 < n:
            model_index = int(nums[-1])
            s1_line   = _strip_leading(lines[i + 1])
            s2_line   = _strip_leading(lines[i + 2])
            ans_line  = _strip_leading(lines[i + 3])
            expl_line = _strip_leading(lines[i + 4])
            if ans_line in {"A", "B", "C"}:
                blocks.append({"index": model_index, "S1": s1_line, "S2": s2_line,
                                "A": ans_line, "explanation": expl_line})
                i += 5
                continue
        i += 1
    return blocks


def _split_blocks_inference(lines):
    """Parse 3-line blocks: index, A, explanation."""
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        nums = re.findall(r"\d+", lines[i].strip())
        if nums and i + 2 < n:
            model_index = int(nums[-1])
            ans_line  = _strip_leading(lines[i + 1])
            expl_line = _strip_leading(lines[i + 2])
            if ans_line in {"A", "B", "C"}:
                blocks.append({"index": model_index, "S1": "", "S2": "",
                                "A": ans_line, "explanation": expl_line})
                i += 3
                continue
        i += 1
    return blocks


def _split_blocks_transcription(lines):
    """Parse 3-line blocks: index, S1, S2."""
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        nums = re.findall(r"\d+", lines[i].strip())
        if nums and i + 2 < n:
            model_index = int(nums[-1])
            s1_line = _strip_leading(lines[i + 1])
            s2_line = _strip_leading(lines[i + 2])
            if s1_line not in {"A", "B", "C"} and s2_line not in {"A", "B", "C"}:
                blocks.append({"index": model_index, "S1": s1_line, "S2": s2_line,
                                "A": "", "explanation": ""})
                i += 3
                continue
        i += 1
    return blocks


def split_into_blocks(output_text, task="both"):
    lines = _prep_lines(output_text)
    if task == "inference":
        return _split_blocks_inference(lines)
    elif task == "transcription":
        return _split_blocks_transcription(lines)
    else:
        return _split_blocks_both(lines)


def validate_block_count(output_text, total_num, fewshot_indices, file_id, attempt, log_f,
                         task="both"):
    """
    Validate model output coverage.

    Rules:
    - REQUIRED = all indices in 0..total_num-1 except fewshot_indices.
    - Few-shot items are demonstrations only and must not appear in the output.
      If the model produces them anyway they are warned about and ignored.
    - Duplicate indices are ALWAYS an error.

    Returns:
      True if all required test indices are present, else False.
    """
    blocks = split_into_blocks(output_text, task=task)

    if not blocks:
        print(f"❌ ERROR [attempt {attempt}] for {file_id}: no parseable blocks found.", file=log_f)
        return False

    model_indices = [b.get("index", None) for b in blocks]
    if any(idx is None for idx in model_indices):
        bad = [i for i, idx in enumerate(model_indices) if idx is None]
        print(f"❌ ERROR [attempt {attempt}] for {file_id}: some blocks missing indices (block positions: {bad}).", file=log_f)
        return False

    if len(model_indices) != len(set(model_indices)):
        seen = set()
        dups = []
        for idx in model_indices:
            if idx in seen:
                dups.append(idx)
            seen.add(idx)
        print(f"❌ ERROR [attempt {attempt}] for {file_id}: duplicate indices found: {sorted(set(dups))}.", file=log_f)
        print(f"   Model indices: {sorted(model_indices)}", file=log_f)
        return False

    model_index_set = set(model_indices)
    fewshot_set = set(fewshot_indices)
    required_test = set(range(0, total_num)) - fewshot_set

    # Warn if model echoed any few-shot demonstration items
    echoed_fs = model_index_set & fewshot_set
    if echoed_fs:
        print(f"⚠️ WARNING [attempt {attempt}] for {file_id}: model produced output for demonstration items {sorted(echoed_fs)} — ignored.", file=log_f)

    # Warn on indices completely outside expected range (not few-shot, not test)
    extra = model_index_set - set(range(0, total_num))
    if extra:
        print(f"⚠️ WARNING [attempt {attempt}] for {file_id}: extra indices outside 0..{total_num-1}: {sorted(extra)}", file=log_f)

    missing_test = required_test - model_index_set
    if missing_test:
        print(f"❌ ERROR [attempt {attempt}] for {file_id}: missing REQUIRED test indices: {sorted(missing_test)}", file=log_f)
        print(f"   Model indices found: {sorted(model_indices)}", file=log_f)
        return False

    print(
        f"✓ Output coverage OK [attempt {attempt}] for {file_id}: "
        f"{len(blocks)} parsed blocks; all {len(required_test)} required test indices present.",
        file=log_f,
    )
    return True


def parse_model_outputs(output_text, examples, task="both"):
    """
    Align model output blocks to gold examples by the model-declared index.
    examples must have ex["idx"] = 0..total_num-1.
    """
    blocks = split_into_blocks(output_text, task=task)
    if not blocks:
        return []

    ex_by_idx = {ex["idx"]: ex for ex in examples}

    results = []
    for b in blocks:
        idx = b["index"]
        if idx not in ex_by_idx:
            continue
        original = ex_by_idx[idx]
        results.append({
            "index": original["idx"],
            "S1_true": original["S1"],
            "S2_true": original["S2"],
            "A_true":  original["A"],
            "focus":   original["focus"],
            "logic":   original["logic"],
            "alternative": original["alternative"],
            "S1_model": b["S1"],
            "S2_model": b["S2"],
            "A_model":  b["A"],
            "explanation": b.get("explanation", ""),
        })

    results.sort(key=lambda r: r["index"])
    return results


# ---------------------------------------------------------------
# Writing results to CSV
# ---------------------------------------------------------------
def write_results_csv(results, csv_path, fieldnames=None):
    if not results:
        return

    if fieldnames is None:
        fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Wrote CSV results to: {csv_path}")


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
def evaluate(parsed_examples, fewshot_indices_set, task="both"):
    """
    parsed_examples: list from parse_model_outputs()
    fewshot_indices_set: set of indices that were few-shot for this run/fold
    task: "both", "inference", or "transcription" — controls which fields are scored
    """
    results = []

    for ex in parsed_examples:
        dataset_index = ex["index"]
        if dataset_index in fewshot_indices_set:
            continue  # demonstrations are never scored

        true_S1 = ex["S1_true"].strip()
        true_S2 = ex["S2_true"].strip()
        true_A  = ex["A_true"]

        model_S1 = ex["S1_model"].strip()
        model_S2 = ex["S2_model"].strip()
        model_A  = ex["A_model"].strip()
        model_explanation = ex.get("explanation", "").strip()

        if task in ("both", "transcription"):
            gold_pos  = ex["focus"]
            model_pos = focus_position(model_S1)
            trans_correct = int(
                model_pos in (FOCUS_FIRST, FOCUS_SECOND)
                and gold_pos in (FOCUS_FIRST, FOCUS_SECOND)
                and model_pos == gold_pos
            )
            s1_edit_norm = normalized_edit_distance(true_S1, model_S1)
            s2_edit_norm = normalized_edit_distance(true_S2, model_S2)
        else:
            trans_correct = ""
            s1_edit_norm  = ""
            s2_edit_norm  = ""

        if task in ("both", "inference"):
            inf_correct = int(model_A == true_A)
        else:
            inf_correct = ""
            model_A = ""
            model_explanation = ""

        results.append({
            "example_index": dataset_index,
            "is_few_shot": 1 if dataset_index in fewshot_indices_set else 0,

            "true_S1": true_S1,
            "true_S2": true_S2,
            "true_A":  true_A,

            "inf_correct": inf_correct,
            "model_A": model_A,

            "trans_correct": trans_correct,

            "model_S1": model_S1,
            "model_S2": model_S2,
            "model_explanation": model_explanation,

            "s1_edit_norm": s1_edit_norm,
            "s2_edit_norm": s2_edit_norm,

            "focus": ex["focus"],
            "logic": ex["logic"],
            "alternative": ex["alternative"],
        })

    return results


# ---------------------------------------------------------------
# CV fold construction
# ---------------------------------------------------------------
def make_cv_folds(total_num: int, fewshot_num: int):
    """
    Deterministic folds:
      fold 0 => few-shot indices [0..fewshot_num-1]
      fold 1 => [fewshot_num..2*fewshot_num-1], etc.

    Requires total_num % fewshot_num == 0.
    """
    if fewshot_num <= 0:
        return [(0, [])]

    if total_num % fewshot_num != 0:
        raise ValueError(f"--cv requires total_num ({total_num}) divisible by fewshot_num ({fewshot_num}).")

    folds = []
    num_folds = total_num // fewshot_num
    for fold in range(num_folds):
        start = fold * fewshot_num
        idxs = list(range(start, start + fewshot_num))
        folds.append((fold, idxs))
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_paths", nargs="+", help="File IDs (e.g. f11 f12 f13) or path prefixes")

    parser.add_argument("--backend", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--model", required=True, help="Model name for the chosen backend")

    parser.add_argument("--fewshot", type=int, default=0, dest="fewshot_num",
                        help="Number of few-shot examples (default: 0)")

    parser.add_argument(
        "--mode",
        choices=["audio", "baseline", "oracle", "text_focus"],
        default="audio",
        help="Experiment mode: audio (default), baseline (no accent), oracle (uppercase accent), text_focus (focus ID from text).",
    )

    parser.add_argument("--wav-dir", default="data/speakers/speaker0/raw",
                        help="Directory containing .wav files (default: data/speakers/speaker0/raw)")
    parser.add_argument("--json-dir", default="data/stimuli",
                        help="Directory containing .json files (default: data/stimuli)")
    parser.add_argument("--speaker", default="",
                        help="Speaker label included in output filenames and CSV (e.g. speaker0, speaker1)")

    parser.add_argument(
        "--task",
        choices=["both", "inference", "transcription"],
        default="both",
        help="Which task to run: both (default), inference only, or transcription only.",
    )
    parser.add_argument("--use_focus_hint", action="store_true")
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable n-fold rotation of few-shot indices (requires total_num divisible by fewshot_num).",
    )
    parser.add_argument(
        "--fewshot-style",
        choices=["simple", "compound"],
        default="simple",
        dest="fewshot_style",
        help="Few-shot example style for inference task: simple (label only) or compound (S1 text + label).",
    )

    args = parser.parse_args()

    # API keys
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if args.backend == "gemini":
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    os.makedirs("data/output", exist_ok=True)

    master_results = []
    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for prefix in args.input_paths:
        print(f"\n=== Processing {prefix} ===")

        file_id = os.path.basename(prefix)

        # Resolve paths: if prefix already contains a directory separator use it directly,
        # otherwise look in the configured directories.
        if os.sep in prefix or "/" in prefix:
            audio_path = prefix + ".wav"
            json_path = prefix + ".json"
        else:
            audio_path = os.path.join(args.wav_dir, prefix + ".wav")
            json_path = os.path.join(args.json_dir, prefix + ".json")

        # Load JSON once and determine total_num from actual file length
        examples_all = load_examples(json_path)
        total_num = len(examples_all)

        # Load audio once
        encoded_audio = load_audio(audio_path) if args.mode == "audio" else None

        # Decide folds (per-file, since total_num varies)
        if args.cv and args.fewshot_num > 0:
            folds = make_cv_folds(total_num, args.fewshot_num)
        else:
            folds = [(None, list(range(0, args.fewshot_num)))]

        for (fold_id, fewshot_indices) in folds:
            fold_tag = f"_cv{fold_id}" if fold_id is not None else ""
            cv_label = str(fold_id) if fold_id is not None else ""

            FH_TEXT = "_focusHint" if args.use_focus_hint else ""
            CV_TEXT = "_CV" if (args.cv and args.fewshot_num > 0) else ""
            SP_TEXT = f"_SP{args.speaker}" if args.speaker else ""
            STYLE_TEXT = f"_{args.fewshot_style}" if args.fewshot_num > 0 else ""

            runID = f"{args.task}_{args.mode}_{args.backend}_{args.model}{SP_TEXT}_FS{args.fewshot_num}{STYLE_TEXT}{FH_TEXT}{CV_TEXT}{fold_tag}_{run_timestamp}"
            log_path = f"data/output/{file_id}_{runID}.log"

            log_f = open(log_path, "w", encoding="utf-8")
            print(
                f"=== Log for {file_id} (backend={args.backend}, model={args.model}, run={run_timestamp}, fold={cv_label}) ===",
                file=log_f,
            )

            # Build few-shot/test sets for this fold (only affects labeling + prompts)
            fewshot_set = set(fewshot_indices)
            few_shot = [ex for ex in examples_all if ex["idx"] in fewshot_set]

            focus_block = FOCUS_HINT_TEXT if (args.mode == "oracle" or (args.mode == "audio" and args.use_focus_hint)) else ""

            if args.mode == "audio":
                if args.task == "inference":
                    prompt = build_base_prompt(
                        TASK_BLOCK_INFERENCE,
                        make_fewshot_item_block(few_shot, task="inference", style=args.fewshot_style),
                        focus_block,
                        make_new_item_block_inference(examples_all, fewshot_indices=fewshot_indices),
                        output_format=OUTPUT_FORMAT_INFERENCE,
                    )
                elif args.task == "transcription":
                    prompt = build_base_prompt(
                        TASK_BLOCK_TRANSCRIPTION,
                        make_fewshot_item_block(few_shot, task="transcription"),
                        "",
                        make_new_item_block_transcription(examples_all, fewshot_indices=fewshot_indices),
                        output_format=OUTPUT_FORMAT_TRANSCRIPTION,
                    )
                else:  # both
                    prompt = build_base_prompt(
                        TASK_BLOCK_AUDIO,
                        make_fewshot_item_block(few_shot, task="both"),
                        focus_block,
                        make_new_item_block_audio(examples_all, fewshot_indices=fewshot_indices),
                        output_format=OUTPUT_FORMAT_BOTH,
                    )
            elif args.mode == "baseline":
                prompt = build_base_prompt(
                    TASK_BLOCK_TEXT,
                    "",
                    "",
                    make_new_item_block_text(examples_all, clean=True),
                )
            elif args.mode == "oracle":
                prompt = build_base_prompt(
                    TASK_BLOCK_ORACLE,
                    "",
                    focus_block,
                    make_new_item_block_text(examples_all, clean=False),
                )
            elif args.mode == "text_focus":
                prompt = build_base_prompt(
                    TASK_BLOCK_TEXT_FOCUS,
                    "",
                    "",
                    make_new_item_block_text_focus(examples_all),
                    output_format=OUTPUT_FORMAT_TRANSCRIPTION,
                )
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            # Attempt loop
            success = False
            completion = None
            output_text = ""

            for attempt in range(1, MAX_RETRIES + 2):
                print(f"\n--- Attempt {attempt} for {file_id}{fold_tag} ---", file=log_f)
                print(f"\n--- Attempt {attempt} for {file_id}{fold_tag} ---")

                print("\n--- Prompt sent to model ---\n", file=log_f)
                print(prompt, file=log_f)
                print("\n--- End of prompt ---\n", file=log_f)

                if args.mode == "audio":
                    if args.backend == "openai":
                        completion = call_openai(prompt, encoded_audio, args.model)
                        output_text = extract_output_text(completion)
                    else:
                        completion = None
                        output_text = call_gemini(prompt, encoded_audio, args.model)
                else:
                    completion = None
                    output_text = run_text_model(prompt, args.model)

                # Log resolved model version (OpenAI returns this in completion.model)
                resolved = getattr(completion, "model", None) if completion else None
                if resolved and resolved != args.model:
                    print(f"\n--- Resolved model: {resolved} ---\n", file=log_f)

                # Log raw output
                print("\n--- Raw Model Output ---\n", file=log_f)
                print(output_text, file=log_f)
                print("\n--- End of Raw Model Output ---\n", file=log_f)

                # Validate coverage (test indices required; few-shot optional)
                ok = validate_block_count(
                    output_text=output_text,
                    total_num=total_num,
                    fewshot_indices=fewshot_indices,
                    file_id=file_id + fold_tag,
                    attempt=attempt,
                    log_f=log_f,
                    task=args.task,
                )
                if not ok:
                    if attempt <= MAX_RETRIES:
                        print(
                            f"🔁 Incomplete output for {file_id}{fold_tag}, retrying (attempt {attempt + 1})...",
                            file=log_f,
                        )
                        continue
                    else:
                        print(f"❌ Giving up on {file_id}{fold_tag} after {attempt} attempts.", file=log_f)
                        break

                # Parse + evaluate
                parsed = parse_model_outputs(output_text, examples_all, task=args.task)
                results = evaluate(parsed, fewshot_set, task=args.task)
                success = True
                break

            if not success:
                log_f.close()
                print(f"❌ Skipped {prefix}{fold_tag} due to incomplete model output.")
                continue

            # Attach metadata & write per-run CSV
            csv_path = f"data/output/{file_id}_{runID}.csv"
            extra_fields = {
                "file_id": file_id,
                "mode": args.mode,
                "backend": args.backend,
                "model_name": args.model,
                "run_timestamp_utc": run_timestamp,
                "response_id": getattr(completion, "id", "") if completion else "",
                "resolved_model": getattr(completion, "model", "") if completion else "",
                "cv_fold": cv_label,
                "speaker": args.speaker,
            }

            results_with_meta = []
            for r in results:
                row = dict(r)
                row.update(extra_fields)
                results_with_meta.append(row)

            write_results_csv(results_with_meta, csv_path, fieldnames=CSV_COLUMNS)
            master_results.extend(results_with_meta)

            print(f"\n✓ Finished processing {prefix}{fold_tag}", file=log_f)
            print(f"  Results saved to: {csv_path}", file=log_f)
            print(f"  Log saved to:     {log_path}\n", file=log_f)
            log_f.close()

    # ------------------------------------------------------------
    # Write MASTER CSV (once, after all prefixes + folds)
    # ------------------------------------------------------------
    if master_results:
        parts = [
            args.task,
            args.mode,
            args.backend,
            args.model,
        ]
        if args.speaker:
            parts.append(f"SP{args.speaker}")
        parts.append(f"FS{args.fewshot_num}")
        if args.fewshot_num > 0:
            parts.append(args.fewshot_style)

        if args.use_focus_hint:
            parts.append("FH")

        if args.cv and args.fewshot_num > 0:
            parts.append("CV")

        master_id = "_".join(parts)
        master_csv_path = f"data/output/master_{master_id}_{run_timestamp}.csv"

        write_results_csv(master_results, master_csv_path, fieldnames=CSV_COLUMNS)
        print(f"\n✓ Master CSV saved to: {master_csv_path}\n")
    else:
        print("\n⚠️ No successful runs — master CSV not created.\n")



if __name__ == "__main__":
    main()

