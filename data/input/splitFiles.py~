import json, os

# Load the 100-example balanced set
subset = json.load(open("balanced_100.json"))

for i in range(10):
    chunk = subset[i*10:(i+1)*10]

    # ---------------------------------------------
    # REMOVE 'position' field from each example
    # ---------------------------------------------
    cleaned_chunk = []
    for ex in chunk:
        ex = dict(ex)              # make a shallow copy
        ex.pop("position", None)   # safely remove if present
        cleaned_chunk.append(ex)

    # Write cleaned chunk
    with open(f"f{i+1}.json", "w") as f:
        json.dump(cleaned_chunk, f, indent=4)
