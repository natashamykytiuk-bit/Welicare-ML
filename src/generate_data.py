"""Generate synthetic candidate items for the dementia-appropriateness dataset.

Uses FICTIONAL resident profiles only. Never put real resident data here.

Usage:
    export ANTHROPIC_API_KEY=...   # or put it in .env (gitignored)
    python src/generate_data.py --n-profiles 50 --items-per-call 10
Output: data/raw/candidates.jsonl
"""
import argparse, json, os, random, uuid
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("GEN_MODEL", "claude-sonnet-5")
OUT = Path("data/raw/candidates.jsonl")

CONTENT_TYPES = ["conversation_starter", "trivia_question", "activity_instruction"]
STAGES = ["early", "mid", "late"]
# Ask for a mix so the dataset isn't all OK. Mirrors docs/rubric.md categories.
QUALITY_MODES = {
    "good": "Follow dementia-care communication best practice: open-ended, choice-based, simple, warm.",
    "subtly_bad": ("Write items a well-meaning but untrained volunteer might write. Each should contain "
                   "ONE subtle problem: testing memory, too many steps, a distressing topic, "
                   "talking down, or correcting the person. Do not label or explain the problem."),
}

PROFILE_PROMPT = """Invent one FICTIONAL care-home resident profile as JSON with keys:
name, birth_year (1930-1955), hometown, career, hobbies (list), favourite_music (list),
family_notes, dementia_stage ("{stage}"). Vary culture, gender and background. Output JSON only."""

ITEM_PROMPT = """Resident profile (fictional):
{profile}

Write {n} distinct {ctype} items for an activity worker to use with this resident.
{mode_instruction}
Output a JSON list of strings only."""


def ask_json(client, prompt):
    msg = client.messages.create(model=MODEL, max_tokens=1500,
                                 messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-profiles", type=int, default=5)
    ap.add_argument("--items-per-call", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    client = Anthropic()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for _ in range(args.n_profiles):
            stage = random.choice(STAGES)
            try:
                profile = ask_json(client, PROFILE_PROMPT.format(stage=stage))
            except Exception as e:  # malformed JSON etc. — skip, don't crash the run
                print("profile failed:", e); continue
            pid = str(uuid.uuid4())[:8]
            for ctype in CONTENT_TYPES:
                for mode, instr in QUALITY_MODES.items():
                    try:
                        items = ask_json(client, ITEM_PROMPT.format(
                            profile=json.dumps(profile), n=args.items_per_call,
                            ctype=ctype.replace("_", " "), mode_instruction=instr))
                    except Exception as e:
                        print("items failed:", e); continue
                    for text in items:
                        f.write(json.dumps({"profile_id": pid, "stage": stage,
                                            "content_type": ctype, "gen_mode": mode,
                                            "text": text, "model": MODEL}) + "\n")
            print(f"profile {pid} ({stage}) done")


if __name__ == "__main__":
    main()
