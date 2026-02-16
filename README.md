# Focus-Sensitive Inference from Speech

This repository contains code and data for experiments on
focus-sensitive semantic inference from spoken language.

## Contents

-   `code/` --- experimental scripts\
-   `data/input/` --- input audio and JSON files\
-   `data/results/` --- CSV outputs used in the paper

## Requirements

-   Python **3.10+**
-   Dependencies listed in `requirements.txt`

Install dependencies:

``` bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

    openai
    google-generativeai

## API Keys

Before running the scripts, set the required API keys as environment
variables:

``` bash
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_API_KEY="your_gemini_key"
```

## Example Run

``` bash
python code/audioInput.py   --backend openai   --model gpt-audio   --mode audio   data/input/f2 0 10
```

## Notes

This repository was originally prepared for anonymous review purposes.
