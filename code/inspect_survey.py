import json
import re
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python code/inspect_survey.py <path_to_qsf>")
    print("Example: python code/inspect_survey.py data/stimuli/Inference_Survey_24.qsf")
    sys.exit(1)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QSF = sys.argv[1]
STIMULI = os.path.join(BASE, "data/stimuli")

# Build lookup: ns_file_item -> item data
lookup = {}
for ns in ["ns1", "ns2", "ns3"]:
    with open(os.path.join(STIMULI, f"{ns}.json")) as f:
        items = json.load(f)
    for i, item in enumerate(items):
        lookup[f"{ns}_item{i}"] = item

# Parse QSF and print each clip with its structure info
with open(QSF) as f:
    qsf = json.load(f)

found = []
for elem in qsf["SurveyElements"]:
    if elem.get("Element") != "SQ":
        continue
    tag = elem.get("Payload", {}).get("DataExportTag", "")
    m = re.match(r"(speaker\d)_(ns\d_item\d+)", tag)
    if m:
        found.append((m.group(1), m.group(2)))

found.sort()
print(f"{'ID':<30} {'Focus':<8} {'Alt':<6} {'Logic':<6} {'Answer':<8} S1 / S2")
print("-" * 100)
for speaker, item_key in found:
    info = lookup.get(item_key, {})
    clip_id = f"{speaker}_{item_key}"
    print(f"{clip_id:<30} {str(info.get('focus')):<8} {info.get('alternative'):<6} {info.get('logic'):<6} {info.get('A'):<8} {info.get('S1')} / {info.get('S2')}")
