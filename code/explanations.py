import csv
import re
import sys
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

EXPLANATION_COL = "model_explanation"

# Regex: 3 or more consecutive uppercase letters (A–Z)
UPPERCASE_RE = re.compile(r"[A-Z]{3,}")

# Case-insensitive search for "focus"
FOCUS_RE = re.compile(r"focus", re.IGNORECASE)


# --------------------------------------------------
# Core logic
# --------------------------------------------------

def is_focus_oriented(text: str) -> bool:
    if not text:
        return False
    return bool(UPPERCASE_RE.search(text) or FOCUS_RE.search(text))


def analyze_file(csv_path: Path):
    total = 0
    focus_count = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if EXPLANATION_COL not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: missing column '{EXPLANATION_COL}'"
            )

        for row in reader:
            total += 1
            explanation = row.get(EXPLANATION_COL, "")
            if is_focus_oriented(explanation):
                focus_count += 1

    return total, focus_count


# --------------------------------------------------
# Entry point
# --------------------------------------------------

def main(csv_files):
    print("=== Focus-oriented explanation analysis ===\n")

    grand_total = 0
    grand_focus = 0

    for file in csv_files:
        path = Path(file)
        total, focus = analyze_file(path)

        grand_total += total
        grand_focus += focus

        pct = 100 * focus / total if total > 0 else 0.0
        print(f"{path.name}")
        print(f"  total explanations: {total}")
        print(f"  focus-oriented:     {focus} ({pct:.1f}%)\n")

    if len(csv_files) > 1:
        pct = 100 * grand_focus / grand_total if grand_total > 0 else 0.0
        print("=== Overall ===")
        print(f"Total explanations: {grand_total}")
        print(f"Focus-oriented:     {grand_focus} ({pct:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python focus_explanations.py file1.csv file2.csv ...")
        sys.exit(1)

    main(sys.argv[1:])
