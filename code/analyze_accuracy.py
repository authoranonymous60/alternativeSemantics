"""
Accuracy analysis of audio-focus inference model runs.

Groups all CSVs in data/output/ by model_name and few-shot setting (FS0/FS2/FS5),
computes accuracy (% inf_correct == 1), and prints a sorted table.
Also reports available breakdown variables.
"""

import os
import re
import pandas as pd

DATA_DIR = "data/output"

# ── 1. Load all CSVs and annotate with filename-derived fields ────────────────

records = []
skipped = []

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, fname)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        skipped.append((fname, str(e)))
        continue

    if "inf_correct" not in df.columns:
        skipped.append((fname, "no inf_correct column"))
        continue

    # Extract FS setting from filename (FS0 / FS2 / FS5)
    fs_match = re.search(r"(FS\d+)", fname)
    fs_setting = fs_match.group(1) if fs_match else "FS?"

    # Extract speaker tag (e.g. SPspeaker0, SPspeaker1, SPspeaker2)
    sp_match = re.search(r"(SP\w+?)(?:_FS|_CV|_\d{8}|$)", fname)
    speaker_tag = sp_match.group(1) if sp_match else "NONE"

    # focusHint present in filename?
    focus_hint = "focusHint" in fname

    # compound few-shot examples?
    compound = "compound" in fname

    # cross-validation fold
    cv_match = re.search(r"_(cv\d+)_", fname)
    cv_fold_tag = cv_match.group(1) if cv_match else None

    df["_filename"] = fname
    df["_fs_setting"] = fs_setting
    df["_speaker_tag"] = speaker_tag
    df["_focus_hint"] = focus_hint
    df["_compound"] = compound
    df["_cv_fold_tag"] = cv_fold_tag

    records.append(df)

print(f"Loaded {len(records)} CSV files ({len(skipped)} skipped)\n")
if skipped:
    print("Skipped files:")
    for f, reason in skipped[:10]:
        print(f"  {f}: {reason}")
    print()

all_data = pd.concat(records, ignore_index=True)
print(f"Total rows: {len(all_data):,}")
print(f"Columns: {all_data.columns.tolist()}\n")

# ── 2. Main table: model_name × FS setting ────────────────────────────────────

def accuracy_table(df, group_cols, label=""):
    grp = (
        df.groupby(group_cols, dropna=False)["inf_correct"]
        .agg(n="count", correct="sum")
        .reset_index()
    )
    grp["accuracy_%"] = (grp["correct"] / grp["n"] * 100).round(1)
    return grp


print("=" * 70)
print("  ACCURACY BY MODEL × FEW-SHOT SETTING")
print("=" * 70)

main_tbl = accuracy_table(all_data, ["model_name", "_fs_setting"])
main_tbl = main_tbl.sort_values(["model_name", "_fs_setting"]).reset_index(drop=True)

# Pretty-print
col_w = {"model_name": 28, "_fs_setting": 10, "n": 8, "correct": 9, "accuracy_%": 12}
header = (
    f"{'Model':<28}  {'FS':>6}  {'N':>8}  {'Correct':>9}  {'Accuracy %':>10}"
)
print(header)
print("-" * 70)
prev_model = None
for _, row in main_tbl.iterrows():
    sep = "" if row["model_name"] == prev_model else ""
    print(
        f"{row['model_name']:<28}  {row['_fs_setting']:>6}  "
        f"{int(row['n']):>8,}  {int(row['correct']):>9,}  {row['accuracy_%']:>9.1f}%"
    )
    prev_model = row["model_name"]

print()

# ── 3. Secondary breakdown: model × FS × focusHint ───────────────────────────

print("=" * 80)
print("  ACCURACY BY MODEL × FS SETTING × FOCUS-HINT")
print("=" * 80)

tbl2 = accuracy_table(all_data, ["model_name", "_fs_setting", "_focus_hint"])
tbl2 = tbl2.sort_values(["model_name", "_fs_setting", "_focus_hint"]).reset_index(drop=True)
tbl2["hint"] = tbl2["_focus_hint"].map({True: "hint", False: "no-hint"})
print(f"{'Model':<28}  {'FS':>6}  {'Hint':>8}  {'N':>8}  {'Acc%':>6}")
print("-" * 65)
for _, row in tbl2.iterrows():
    print(
        f"{row['model_name']:<28}  {row['_fs_setting']:>6}  {row['hint']:>8}  "
        f"{int(row['n']):>8,}  {row['accuracy_%']:>5.1f}%"
    )

print()

# ── 4. Breakdown: logic (POS/NEG) ────────────────────────────────────────────

if "logic" in all_data.columns:
    print("=" * 70)
    print("  ACCURACY BY MODEL × FS SETTING × LOGIC (POS/NEG)")
    print("=" * 70)
    tbl3 = accuracy_table(all_data, ["model_name", "_fs_setting", "logic"])
    tbl3 = tbl3.sort_values(["model_name", "_fs_setting", "logic"]).reset_index(drop=True)
    print(f"{'Model':<28}  {'FS':>6}  {'Logic':>6}  {'N':>8}  {'Acc%':>6}")
    print("-" * 62)
    for _, row in tbl3.iterrows():
        print(
            f"{row['model_name']:<28}  {row['_fs_setting']:>6}  {str(row['logic']):>6}  "
            f"{int(row['n']):>8,}  {row['accuracy_%']:>5.1f}%"
        )
    print()

# ── 5. Breakdown: focus position ─────────────────────────────────────────────

if "focus" in all_data.columns:
    print("=" * 70)
    print("  ACCURACY BY MODEL × FS SETTING × FOCUS POSITION")
    print("=" * 70)
    tbl4 = accuracy_table(all_data, ["model_name", "_fs_setting", "focus"])
    tbl4 = tbl4.sort_values(["model_name", "_fs_setting", "focus"]).reset_index(drop=True)
    print(f"{'Model':<28}  {'FS':>6}  {'Focus':>6}  {'N':>8}  {'Acc%':>6}")
    print("-" * 62)
    for _, row in tbl4.iterrows():
        print(
            f"{row['model_name']:<28}  {row['_fs_setting']:>6}  {str(row['focus']):>6}  "
            f"{int(row['n']):>8,}  {row['accuracy_%']:>5.1f}%"
        )
    print()

# ── 6. Breakdown: speaker ────────────────────────────────────────────────────

if "speaker" in all_data.columns:
    print("=" * 70)
    print("  ACCURACY BY MODEL × FS SETTING × SPEAKER (from CSV column)")
    print("=" * 70)
    tbl5 = accuracy_table(all_data, ["model_name", "_fs_setting", "speaker"])
    tbl5 = tbl5.sort_values(["model_name", "_fs_setting", "speaker"]).reset_index(drop=True)
    print(f"{'Model':<28}  {'FS':>6}  {'Speaker':>10}  {'N':>8}  {'Acc%':>6}")
    print("-" * 66)
    for _, row in tbl5.iterrows():
        print(
            f"{row['model_name']:<28}  {row['_fs_setting']:>6}  {str(row['speaker']):>10}  "
            f"{int(row['n']):>8,}  {row['accuracy_%']:>5.1f}%"
        )
    print()

# ── 7. Report available grouping variables ───────────────────────────────────

print("=" * 70)
print("  AVAILABLE GROUPING VARIABLES")
print("=" * 70)
check_cols = ["model_name", "logic", "focus", "alternative", "is_few_shot",
              "speaker", "cv_fold", "mode", "backend", "file_id", "trans_correct"]
for col in check_cols:
    if col in all_data.columns:
        vals = all_data[col].dropna().unique()
        n_unique = len(vals)
        sample = sorted([str(v) for v in vals])[:10]
        print(f"  {col:<18} ({n_unique:>4} unique): {', '.join(sample)}")

print()
fname_vars = {
    "FS setting (filename)": "_fs_setting",
    "Speaker tag (filename)": "_speaker_tag",
    "Focus hint (filename)": "_focus_hint",
    "Compound examples (filename)": "_compound",
    "CV fold (filename)": "_cv_fold_tag",
}
for label, col in fname_vars.items():
    vals = all_data[col].dropna().unique()
    n_unique = len(vals)
    sample = sorted([str(v) for v in vals])[:10]
    print(f"  {label:<30} ({n_unique:>3} unique): {', '.join(sample)}")
