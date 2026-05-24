import os
import json
import random
from collections import defaultdict

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = "data/stimuli"
OUTPUT_DIR = "data/output"
RANDOM_SEED = 42



# ---------------------------
# HELPERS
# --------------------------- 
def infer_logic(s2: str) -> str:
    s2_lower = s2.lower()
    if "didn't give" in s2_lower or "did not give" in s2_lower:
        return "NEG"
    elif "also gave" in s2_lower:
        return "POS"
    else:
        raise ValueError(f"Could not infer logic from S2: {s2}")


def parse_s1(s1: str):
    tokens = s1.replace(".", "").split()
    if len(tokens) < 5:
        raise ValueError(f"Unexpected S1 format: {s1}")
    recipient = tokens[3]
    item = tokens[4]
    return recipient, item


def parse_s2(s2: str):
    tokens = s2.replace(".", "").split()
    if len(tokens) < 5:
        raise ValueError(f"Unexpected S2 format: {s2}")
    recipient = tokens[3]
    item = tokens[4]
    return recipient, item


def infer_alternative(s1: str, s2: str) -> str:
    s1_recipient, s1_item = parse_s1(s1)
    s2_recipient, s2_item = parse_s2(s2)

    recipient_changed = s1_recipient.lower() != s2_recipient.lower()
    item_changed = s1_item.lower() != s2_item.lower()

    if recipient_changed and not item_changed:
        return "1"
    elif item_changed and not recipient_changed:
        return "2"
    else:
        raise ValueError(
            f"Could not uniquely infer alternative.\nS1: {s1}\nS2: {s2}"
        )


def get_structure(item):
    focus = str(item["focus"])
    logic = item.get("logic", infer_logic(item["S2"]))
    alternative = str(item.get("alternative", infer_alternative(item["S1"], item["S2"])))
    return (focus, alternative, logic)


# ---------------------------
# LOAD ALL ITEMS
# ---------------------------
def load_all_items(data_dir):
    all_items = []

    print(f"Looking for input files in: {data_dir}")

    for i in range(1, 11):
        filename = f"f{i}.json"
        path = os.path.join(data_dir, filename)

        print(f"Loading: {path}")

        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        print(f"Loaded {len(items)} items from {filename}")

        for j, item in enumerate(items):
            item_copy = item.copy()

            if "logic" not in item_copy:
                item_copy["logic"] = infer_logic(item_copy["S2"])
            if "alternative" not in item_copy:
                item_copy["alternative"] = infer_alternative(item_copy["S1"], item_copy["S2"])

            item_copy["source_file"] = filename
            item_copy["item_index"] = j
            item_copy["structure"] = get_structure(item_copy)

            all_items.append(item_copy)

    print(f"\nTotal loaded items: {len(all_items)}")
    return all_items


# ---------------------------
# BUILD TWO STRATIFIED SETS
# ---------------------------
def build_two_sets(all_items, seed=42):
    random.seed(seed)

    grouped = defaultdict(list)
    for item in all_items:
        grouped[item["structure"]].append(item)

    print("\nStructure counts:")
    for structure in sorted(grouped.keys()):
        print(f"{structure}: {len(grouped[structure])}")

    expected_structures = {
        ("1", "1", "NEG"),
        ("1", "1", "POS"),
        ("1", "2", "NEG"),
        ("1", "2", "POS"),
        ("2", "1", "NEG"),
        ("2", "1", "POS"),
        ("2", "2", "NEG"),
        ("2", "2", "POS"),
    }

    found_structures = set(grouped.keys())
    missing = expected_structures - found_structures

    if missing:
        raise ValueError(f"Missing structures: {missing}")

    set1 = []
    set2 = []

    for structure in sorted(grouped.keys()):
        items = grouped[structure]

        if len(items) < 2:
            raise ValueError(
                f"Not enough items in structure {structure}. "
                f"Need at least 2, found {len(items)}"
            )

        chosen = random.sample(items, 2)
        set1.append(chosen[0])
        set2.append(chosen[1])

    random.shuffle(set1)
    random.shuffle(set2)

    return set1, set2


# ---------------------------
# SAVE OUTPUTS
# ---------------------------
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("Script started")

    all_items = load_all_items(DATA_DIR)

    if len(all_items) == 0:
        raise ValueError("No items were loaded. Check DATA_DIR and file names.")

    set1, set2 = build_two_sets(all_items, seed=RANDOM_SEED)

    set1_path = os.path.join(OUTPUT_DIR, "set1.json")
    set2_path = os.path.join(OUTPUT_DIR, "set2.json")

    save_json(set1, set1_path)
    save_json(set2, set2_path)

    print("\nDone.")
    print(f"Set 1 saved to: {set1_path}")
    print(f"Set 2 saved to: {set2_path}")