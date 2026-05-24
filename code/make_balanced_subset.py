import json
import random
from collections import defaultdict

# -----------------------------
# Config
# -----------------------------
INPUT_FILE = "all_examples.json"     # your full JSON file
OUTPUT_FILE = "balanced_100.json"    # output subset file
TOTAL_DESIRED = 100
RANDOM_SEED = 42                     # change or remove for different samples

random.seed(RANDOM_SEED)


def load_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_condition(examples):
    """
    Group examples by the full 8-case condition:
       (A, focus, logic, alternative)
    """
    groups = defaultdict(list)
    for ex in examples:
        key = (ex["A"], ex["focus"], ex["logic"], ex["alternative"])
        groups[key].append(ex)
    return groups


def compute_target_counts(groups, total_desired):
    """
    Compute how many examples to sample from each (A, focus, logic, alternative) cell.
    """
    conditions = sorted(groups.keys())
    n_cells = len(conditions)

    base = total_desired // n_cells
    remainder = total_desired % n_cells

    target = {}
    for i, cond in enumerate(conditions):
        extra = 1 if i < remainder else 0
        requested = base + extra
        available = len(groups[cond])
        target[cond] = min(requested, available)

    current_total = sum(target.values())
    if current_total < total_desired:
        leftover = total_desired - current_total

        while leftover > 0:
            made_progress = False
            for cond in conditions:
                if leftover <= 0:
                    break
                available = len(groups[cond])
                if target[cond] < available:
                    target[cond] += 1
                    leftover -= 1
                    made_progress = True
            if not made_progress:
                break

    return target


def sample_balanced_subset(groups, target_counts):
    sampled = []
    for cond, examples in groups.items():
        k = target_counts.get(cond, 0)
        if k > 0:
            if k > len(examples):
                raise ValueError(
                    f"Requested {k} but only {len(examples)} available for condition {cond}"
                )
            sampled.extend(random.sample(examples, k))
    return sampled


def main():
    examples = load_examples(INPUT_FILE)
    print(f"Loaded {len(examples)} examples.")

    groups = group_by_condition(examples)

    print("Available per (A, focus, logic, alternative):")
    for cond in sorted(groups.keys()):
        print(f"  {cond}: {len(groups[cond])}")

    target_counts = compute_target_counts(groups, TOTAL_DESIRED)

    print("\nTarget counts:")
    total_target = 0
    for cond in sorted(target_counts.keys()):
        count = target_counts[cond]
        total_target += count
        print(f"  {cond}: {count}")
    print(f"Total in subset: {total_target}")

    subset = sample_balanced_subset(groups, target_counts)

    random.shuffle(subset)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=4)

    print(f"\nWrote {len(subset)} examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
