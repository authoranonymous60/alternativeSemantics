# Focus-Sensitive Inference from Speech

This repository contains code and data for experiments on focus-sensitive semantic inference from spoken language.

## Contents

- `code/` — experimental scripts  
- `data/input/` — input audio + JSON files  
- `data/results/` — CSV outputs used in the paper  

## Requirements

This project runs on **Python 3.10+**.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

```API Keys
`export OPENAI_API_KEY="your_openai_key`
`export GOOGLE_API_KEY="your_gemini_key`

```Run
`python code/audioInput.py \
  --backend openai \
  --model gpt-audio \
  --mode audio \
  data/input/f2 0 10`




```Notes

This repository was originally provided for anonymous review purposes.