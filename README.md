# Welicare Content Classifier

A small fine-tuned model that flags LLM-generated activity content that is inappropriate
for people living with dementia, before it reaches caregivers in the
[Welicare](https://github.com/natashamykytiuk-bit/Welicare) app.

> 🚧 Work in progress. Sections marked TODO are filled as milestones complete.

## Problem
Welicare uses an LLM to generate conversation starters, trivia and activities from resident
life-story profiles. LLMs sometimes produce content that tests memory, is too complex, or
risks distress. This model screens and reranks that content: fast, cheap, and runs on CPU.

## Labels
See [`docs/rubric.md`](docs/rubric.md): `RECALL`, `COMPLEX`, `DISTRESS`, `INFANTILIZING`, `CORRECTIVE`, `OK`.

## Data
- **Synthetic only:** fictional resident profiles, no real resident data.
- **Noisy set:** LLM-generated items, weak-labeled by an LLM. TODO: size, labeler(s)
- **Gold set:** hand-labeled test set. TODO: size, LLM–human agreement (κ)

## Results
TODO: baselines vs. model table (macro-F1, per-class F1, latency, cost / 1k items)

## Integration
TODO: Dockerized FastAPI service on Cloud Run, called from Welicare's `generateSuggestions`.

## Limitations
- Rubric reflects published guidance and TODO practitioner review; it is not clinically validated.
- Synthetic data may not match real-world content distribution. English only.

## Reproduce
```bash
pip install -r requirements.txt
cp .env.example .env   # add your API key
python src/generate_data.py --n-profiles 5
```
