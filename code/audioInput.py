# audioInput.py
#
# Examples (UNCHANGED):
#   python audioInput.py --backend openai --model gpt-4o --mode baseline input/f1 input/f2 ... 0 10
#   python audioInput.py --backend openai --model gpt-audio --mode audio input/f2 0 10
#   python audioInput.py --backend openai --model gpt-4o --mode oracle input/f2 0 10
#
# CV mode (NEW, optional):
#   python audioInput.py --cv --backend openai --model gpt-audio --mode audio input/f1 ... 2 10
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

    # NEW (optional): which CV fold produced this row (empty string if not --cv)
    "cv_fold",
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

TASK_BLOCK_TEXT = """
You are given text for S1 and S2.
Your task is to classify S2 relative to S1:
   A = entailed
   B = independent
   C = contradicted.
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

def build_base_prompt(task_block: str, fewshot_text: str, focus_block: str, new_item_block: str) -> str:
    return f"""
You are performing a semantic classification task.

{task_block}

{focus_block}

Your output must follow this structure:

<index>
S1: ...
S2: ...
A
Because <explanation>

Do not add meta-comments or tool-use descriptions.

{fewshot_text}

---------------------------------------------------------------
NEW INPUT EXAMPLES
---------------------------------------------------------------
<BEGIN_NEW>
{new_item_block}
<END_NEW>

Begin now.
"""


def make_new_item_block_audio(examples):
    # Always ask for all indices, so the model *can* answer everything.
    block = ""
    for ex in examples:
        block += f"{ex['idx']}\nS1:\nS2:\nA\nBecause...\n\n"
    return block


def make_new_item_block_text(examples, clean=False):
    block = ""
    for ex in examples:
        S1 = remove_focus_from_S1(ex["S1"]) if clean else ex["S1"]
        block += f"{ex['idx']}\nS1: {S1}\nS2: {ex['S2']}\nA\nBecause...\n\n"
    return block


def make_fewshot_item_block(fewshot_examples, total_num: int):
    """
    Few-shot items are *task items* with an answer shown.
    We still require the model to answer them (but coverage validation treats them as optional).
    """
    n = len(fewshot_examples)
    if n == 0:
        return ""

    idxs = [ex["idx"] for ex in fewshot_examples]
    idxs_str = ", ".join(str(i) for i in idxs)

    block = (
        f"The following numbered examples ALREADY INCLUDE a correct answer: {idxs_str}\n\n"
        f"IMPORTANT:\n"
        f"- These examples are NOT demonstrations.\n"
        f"- They are full task items, just like the later ones.\n"
        f"- The presence of an answer does NOT mean you should skip them.\n"
        f"- You MUST produce a complete output block for EVERY numbered example (0–{total_num-1}).\n\n"
        f"For each numbered example (0–{total_num-1}), you must:\n"
        f"- Transcribe S1 and S2 from the audio\n"
        f"- Mark the focused element in S1 using UPPERCASE\n"
        f"- Provide your own classification (A, B, or C)\n"
        f"- Provide an explanation\n\n"
    )

    for ex in fewshot_examples:
        block += (
            f"{ex['idx']}\n"
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
    model = genai.GenerativeModel(model_name)
    contents = [
        {"text": prompt},
        {"mime_type": "audio/wav", "data": base64.b64decode(encoded_audio)},
    ]
    response = model.generate_content(contents, generation_config={"temperature": 0.0})
    return response.text or ""


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

def split_into_blocks(output_text):
    lines = [ln.rstrip("\n") for ln in output_text.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]  # remove blank lines

    blocks = []
    i = 0
    n = len(lines)

    def strip_leading_numbering(s):
        s = re.sub(r"^\s*\d+[\.\):]\s*", "", s).strip()
        s = re.sub(r"^\s*(S1|S2|A)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^\s*Because\s*:? ?", "", s, flags=re.IGNORECASE).strip()
        return s

    while i < n:
        line = lines[i].strip()
        nums = re.findall(r"\d+", line)

        if nums:
            try:
                model_index = int(nums[-1])   # LAST integer on the line
            except ValueError:
                model_index = None

            if i + 4 >= n:
                break

            s1_line = strip_leading_numbering(lines[i + 1])
            s2_line = strip_leading_numbering(lines[i + 2])
            ans_line = strip_leading_numbering(lines[i + 3])
            expl_line = strip_leading_numbering(lines[i + 4])

            if ans_line in {"A", "B", "C"}:
                blocks.append({
                    "index": model_index,
                    "S1": s1_line,
                    "S2": s2_line,
                    "A": ans_line,
                    "explanation": expl_line,
                })
                i += 5
                continue

        i += 1

    return blocks


def validate_block_count(output_text, total_num, fewshot_indices, file_id, attempt, log_f):
    """
    Validate model output coverage.

    Rules:
    - Indices come from split_into_blocks() (model's own index).
    - REQUIRED = all indices in 0..total_num-1 except fewshot_indices.
    - OPTIONAL = fewshot_indices (warn only if missing).
    - Duplicate indices are ALWAYS an error.
    - Extra indices outside 0..total_num-1 are allowed (warn), ignored for coverage.

    Returns:
      True if coverage is acceptable, else False.
    """
    blocks = split_into_blocks(output_text)

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

    expected_all = set(range(0, total_num))
    fewshot_set = set(fewshot_indices)
    required_test = expected_all - fewshot_set
    optional_fs = fewshot_set

    extra = model_index_set - expected_all
    if extra:
        print(f"⚠️ WARNING [attempt {attempt}] for {file_id}: extra indices outside 0..{total_num-1}: {sorted(extra)}", file=log_f)

    missing_test = required_test - model_index_set
    if missing_test:
        print(f"❌ ERROR [attempt {attempt}] for {file_id}: missing REQUIRED test indices: {sorted(missing_test)}", file=log_f)
        print(f"   Model indices found: {sorted(model_indices)}", file=log_f)
        return False

    missing_fs = optional_fs - model_index_set
    if missing_fs:
        print(f"⚠️ WARNING [attempt {attempt}] for {file_id}: missing few-shot indices (optional): {sorted(missing_fs)}", file=log_f)

    num_in_range = len(model_index_set & expected_all)
    print(
        f"✓ Output coverage OK [attempt {attempt}] for {file_id}: "
        f"{len(blocks)} parsed blocks; {num_in_range}/{total_num} indices in-range; "
        f"all {len(required_test)} required test indices present.",
        file=log_f,
    )
    return True


def parse_model_outputs(output_text, examples):
    """
    Align model output blocks to gold examples by the model-declared index.
    examples must have ex["idx"] = 0..total_num-1.
    """
    blocks = split_into_blocks(output_text)
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
def evaluate(parsed_examples, fewshot_indices_set):
    """
    parsed_examples: list from parse_model_outputs()
    fewshot_indices_set: set of indices that were few-shot for this run/fold
    """
    results = []

    for ex in parsed_examples:
        dataset_index = ex["index"]

        true_S1 = ex["S1_true"].strip()
        true_S2 = ex["S2_true"].strip()
        true_A  = ex["A_true"]

        model_S1 = ex["S1_model"].strip()
        model_S2 = ex["S2_model"].strip()
        model_A  = ex["A_model"].strip()
        model_explanation = ex.get("explanation", "").strip()

        gold_pos = ex["focus"]
        model_pos = focus_position(model_S1)

        trans_correct = int(
            model_pos in (FOCUS_FIRST, FOCUS_SECOND)
            and gold_pos in (FOCUS_FIRST, FOCUS_SECOND)
            and model_pos == gold_pos
        )

        s1_edit_norm = normalized_edit_distance(true_S1, model_S1)
        s2_edit_norm = normalized_edit_distance(true_S2, model_S2)

        inf_correct = int(model_A == true_A)

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
    parser.add_argument("input_paths", nargs="+", help="Path prefix(es) for .wav and .json files")

    parser.add_argument("--backend", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--model", required=True, help="Model name for the chosen backend")

    parser.add_argument("fewshot_num", type=int, help="Number of few-shot examples")
    parser.add_argument("total_num", type=int, help="Total number of examples per file")

    parser.add_argument(
        "--mode",
        choices=["audio", "baseline", "oracle"],
        default="audio",
        help="Experiment mode: audio (default), baseline (no accent), oracle (uppercase accent).",
    )

    parser.add_argument("--use_focus_hint", action="store_true")
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable n-fold rotation of few-shot indices (requires total_num divisible by fewshot_num).",
    )

    args = parser.parse_args()

    # API keys
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if args.backend == "gemini":
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    os.makedirs("results", exist_ok=True)

    master_results = []
    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Decide folds
    if args.cv and args.fewshot_num > 0:
        folds = make_cv_folds(args.total_num, args.fewshot_num)
    else:
        folds = [(None, list(range(0, args.fewshot_num)))]  # classic: first fewshot_num are few-shot

    for prefix in args.input_paths:
        print(f"\n=== Processing {prefix} ===")

        audio_path = prefix + ".wav"
        json_path = prefix + ".json"

        file_id = os.path.basename(prefix)

        # Load JSON once
        examples_all = load_examples(json_path)

        if len(examples_all) != args.total_num:
            print(
                f"⚠️ WARNING: {file_id} has {len(examples_all)} examples but CLI total_num is {args.total_num}. "
                "Using CLI total_num for coverage."
            )

        # Truncate/align if needed (safety)
        examples_all = examples_all[:args.total_num]

        # Load audio once
        encoded_audio = load_audio(audio_path) if args.mode == "audio" else None

        for (fold_id, fewshot_indices) in folds:
            fold_tag = f"_cv{fold_id}" if fold_id is not None else ""
            cv_label = str(fold_id) if fold_id is not None else ""

            FH_TEXT = "_focusHint" if args.use_focus_hint else ""
            CV_TEXT = "_CV" if (args.cv and args.fewshot_num > 0) else ""

            runID = f"{args.mode}_{args.backend}_{args.model}_FS{args.fewshot_num}{FH_TEXT}{CV_TEXT}{fold_tag}_{run_timestamp}"
            log_path = f"results/{file_id}_{runID}.log"

            log_f = open(log_path, "w", encoding="utf-8")
            print(
                f"=== Log for {file_id} (backend={args.backend}, model={args.model}, run={run_timestamp}, fold={cv_label}) ===",
                file=log_f,
            )

            # Build few-shot/test sets for this fold (only affects labeling + prompts)
            fewshot_set = set(fewshot_indices)
            few_shot = [ex for ex in examples_all if ex["idx"] in fewshot_set]

            focus_block = FOCUS_HINT_TEXT if (args.mode == "audio" and args.use_focus_hint) else ""

            if args.mode == "audio":
                prompt = build_base_prompt(
                    TASK_BLOCK_AUDIO,
                    make_fewshot_item_block(few_shot, total_num=args.total_num),
                    focus_block,
                    make_new_item_block_audio(examples_all),
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
                    TASK_BLOCK_TEXT,
                    "",
                    "",
                    make_new_item_block_text(examples_all, clean=False),
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

                # Log raw output
                print("\n--- Raw Model Output ---\n", file=log_f)
                print(output_text, file=log_f)
                print("\n--- End of Raw Model Output ---\n", file=log_f)

                # Validate coverage (test indices required; few-shot optional)
                ok = validate_block_count(
                    output_text=output_text,
                    total_num=args.total_num,
                    fewshot_indices=fewshot_indices,
                    file_id=file_id + fold_tag,
                    attempt=attempt,
                    log_f=log_f,
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
                parsed = parse_model_outputs(output_text, examples_all)
                results = evaluate(parsed, fewshot_set)
                success = True
                break

            if not success:
                log_f.close()
                print(f"❌ Skipped {prefix}{fold_tag} due to incomplete model output.")
                continue

            # Attach metadata & write per-run CSV
            csv_path = f"data/results/{file_id}_{runID}.csv"
            extra_fields = {
                "file_id": file_id,
                "mode": args.mode,
                "backend": args.backend,
                "model_name": args.model,
                "run_timestamp_utc": run_timestamp,
                "response_id": getattr(completion, "id", "") if completion else "",
                "cv_fold": cv_label,
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
            args.mode,
            args.backend,
            args.model,
            f"FS{args.fewshot_num}",
        ]

        if args.use_focus_hint:
            parts.append("FH")

        if args.cv and args.fewshot_num > 0:
            parts.append("CV")

        master_id = "_".join(parts)
        master_csv_path = f"data/results/master_{master_id}_{run_timestamp}.csv"

        write_results_csv(master_results, master_csv_path, fieldnames=CSV_COLUMNS)
        print(f"\n✓ Master CSV saved to: {master_csv_path}\n")
    else:
        print("\n⚠️ No successful runs — master CSV not created.\n")



if __name__ == "__main__":
    main()

