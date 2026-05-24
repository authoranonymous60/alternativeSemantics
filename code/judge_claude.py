"""
LLM-as-judge using Claude directly, with semantically correct arg1/arg2 focus prompts.

E_FocusID: Does the explanation apply the semantics matching the actual prosodic focus?
  - focus=1 (arg1, PERSON capitalised): correct reasoning restricts WHO received the item
    e.g. "only BILL got grapes" → Sam gave grapes to no one else
  - focus=2 (arg2, FRUIT capitalised): correct reasoning restricts WHAT the person received
    e.g. "Rob only got ORANGES" → Rob received no other fruit

E_Inf: Does the explanation's reasoning lead to (support) the correct inference label?
  A=entailed, B=independent, C=contradicted
"""

import os
import re
import pandas as pd
import anthropic
from datetime import datetime

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

INF_FILES = {
    "gemini-3.1-pro-preview": "data/output/master_inference_audio_gemini_gemini-3.1-pro-preview_FS0_20260428_143342.csv",
    "gemini-2.5-flash": "data/output/master_inference_audio_gemini_gemini-2.5-flash_FS0_20260428_143340.csv",
    "gpt-4o-audio-preview": "data/output/master_inference_audio_openai_gpt-4o-audio-preview_FS0_20260428_152454.csv",
    "gpt-audio": "data/output/master_inference_audio_openai_gpt-audio_FS0_20260428_143339.csv",
}

FOCUSID_PROMPT = """\
S1: {true_S1}
S2: {true_S2}
Prosodic focus: the UPPERCASE word is "{focus_word}" ({focus_type}).
Model explanation: "{explanation}"

In S1, the UPPERCASE word marks prosodic focus. This determines the semantics of "only":
- If a PERSON is capitalised (arg1 focus): "only" restricts WHO received the item.
  Correct reasoning: "Sam gave [item] to no one else" / "only [PERSON] received [item]".
- If a FRUIT is capitalised (arg2 focus): "only" restricts WHAT the person received.
  Correct reasoning: "Sam gave [PERSON] no other fruit" / "[PERSON] only received [FRUIT]".

Does the model's explanation apply the semantics that match the actual focus ({focus_type})?
Answer YES if it does. Answer NO if it applies the opposite focus semantics, or is confused about which argument is focused.

Respond in exactly this format, nothing else:
E_FOCUSID: YES/NO"""

INF_PROMPT = """\
S1: {true_S1}
S2: {true_S2}
Correct label: {true_A}  (A=entailed, B=independent, C=contradicted)
Model explanation: "{explanation}"

Does the explanation's reasoning actually support the correct label ({true_A})?
Answer YES if the logic in the explanation leads to label {true_A}.
Answer NO if the logic leads to a different label, or is incoherent.

Respond in exactly this format, nothing else:
E_INF: YES/NO"""


def focus_info(row):
    import re
    m = re.search(r'\b([A-Z]{2,})\b', str(row['true_S1']))
    focus_word = m.group(1) if m else "?"
    if row['focus'] == 1:
        focus_type = "arg1, person focus"
    else:
        focus_type = "arg2, fruit focus"
    return focus_word, focus_type


def judge_one(prompt_text):
    msg = client.messages.create(
        model=MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return msg.content[0].text.strip()


def parse_focusid(raw):
    m = re.search(r"E_FOCUSID:\s*(YES|NO)", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper() == "YES"
    return None


def parse_einf(raw):
    m = re.search(r"E_INF:\s*(YES|NO)", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper() == "YES"
    return None


def run_judge():
    all_rows = []
    for model_name, path in INF_FILES.items():
        df = pd.read_csv(path)
        df['model_name'] = model_name
        n = len(df)
        print(f"\n{model_name} ({n} items)...")
        for i, row in df.iterrows():
            expl = str(row.get('model_explanation', '')).strip()
            if not expl or expl == 'nan':
                all_rows.append({
                    'model_name': model_name,
                    'file_id': row['file_id'],
                    'example_index': row['example_index'],
                    'focus': row['focus'],
                    'true_A': row['true_A'],
                    'model_A': row['model_A'],
                    'inf_correct': row['inf_correct'],
                    'E_FOCUSID': None,
                    'E_INF': None,
                })
                continue

            focus_word, focus_type = focus_info(row)

            # E_FocusID
            fid_prompt = FOCUSID_PROMPT.format(
                true_S1=row['true_S1'],
                true_S2=row['true_S2'],
                focus_word=focus_word,
                focus_type=focus_type,
                explanation=expl,
            )
            fid_raw = judge_one(fid_prompt)
            e_focusid = parse_focusid(fid_raw)

            # E_Inf
            einf_prompt = INF_PROMPT.format(
                true_S1=row['true_S1'],
                true_S2=row['true_S2'],
                true_A=row['true_A'],
                explanation=expl,
            )
            einf_raw = judge_one(einf_prompt)
            e_inf = parse_einf(einf_raw)

            all_rows.append({
                'model_name': model_name,
                'file_id': row['file_id'],
                'example_index': row['example_index'],
                'focus': row['focus'],
                'true_A': row['true_A'],
                'model_A': row['model_A'],
                'inf_correct': row['inf_correct'],
                'E_FOCUSID': e_focusid,
                'E_INF': e_inf,
            })

            if (i + 1) % 32 == 0:
                done = sum(1 for r in all_rows if r['model_name'] == model_name)
                print(f"  {done}/{n}")

    out = pd.DataFrame(all_rows)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outpath = f"data/output/judge_claude_FS0_{ts}.csv"
    out.to_csv(outpath, index=False)
    print(f"\nSaved to {outpath}")
    print("\nNulls:", out[['E_FOCUSID','E_INF']].isnull().sum().to_dict())
    print("\nBy model:")
    print(out.groupby('model_name')[['E_FOCUSID','E_INF']].mean().round(3))
    return outpath


if __name__ == "__main__":
    run_judge()
