import json, re, os
base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base, 'Inference_Survey_24.qsf')) as f:
    qsf = json.load(f)
lookup = {}
for ns in ['ns1', 'ns2', 'ns3']:
    with open(os.path.join(base, f'{ns}.json')) as f2:                                                                                                             
        items = json.load(f2)
    for i, item in enumerate(items):
        lookup[f'{ns}_item{i}'] = item
for elem in qsf['SurveyElements']:
    if elem.get('Element') != 'SQ':
        continue
    text = elem.get('Payload', {}).get('QuestionText', '')
    clips = re.findall(r'(speaker\d)_(ns\d_item\d+)', text)
    for speaker, item_key in clips:
        info = lookup.get(item_key, {})
        print(f'{speaker}_{item_key}  focus={info.get("focus")} alt={info.get("alternative")} {info.get("logic")}  answer={info.get("A")}')
        print(f'  S1: {info.get("S1")}')
        print(f'  S2: {info.get("S2")}')
        print()
