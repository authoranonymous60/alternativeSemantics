# Focus-Sensitive Inference from Speech

This repository contains code and data for experiments on
focus-sensitive semantic inference from spoken language.

## Contents
- `code/` — experimental scripts
- `data/input/` — input audio + JSON files
- `data/results/` — CSV outputs used in the paper

##
Here is a test example
python code/audioInput.py --backend openai --model gpt-audio --mode audio data/input/f2 0 10

models currently available: gpt-audio, gpt-4o-audio-preview, gemini-2.0-flash

## Reproducibility
The experiments require access to:
- OpenAI audio models
- Google Gemini audio models



## Notes
This repository is provided for anonymous review purposes.
