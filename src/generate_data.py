"""Generate synthetic candidate items for the dementia-appropriateness dataset.

Uses FICTIONAL resident profiles only. Never put real resident data here.
Training-data use only. Production Welicare uses Claude plus the trained classifier.

Usage:
    # keys go in .env (gitignored), see .env.example
    python src/generate_data.py --provider claude --n-profiles 50
    python src/generate_data.py --provider openai --n-profiles 25
Output: appends to data/raw/candidates.jsonl (each row records provider and model)
"""
import argparse, json, os, random, time, uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
OUT = Path("data/raw/candidates.jsonl")

CONTENT_TYPES = ["conversation_starter", "trivia_question", "activity_instruction"]
STAGES = ["early", "mid", "late"]
# Same good/subtly-bad mix for every provider, so the classifier can't learn
# "which model wrote this" as a shortcut for "is this bad".
GOOD_INSTRUCTION = ("Follow dementia-care communication best practice: open-ended, choice-based, "
                    "simple, warm, strength-based.")

# One flaw per subtle batch, mirroring docs/rubric.md. Descriptions are deliberately
# about the *subtle* form of each flaw.
FLAWS = {
    "RECALL": ("quietly relies on the person retrieving a specific memory or fact (a name, date, "
               "place, sequence), phrased warmly so it doesn't sound like a test. Avoid the words "
               "'remember' and 'recall'."),
    "COMPLEX": ("asks for slightly too much at once: two linked questions, a hypothetical, an "
                "abstract comparison, or an instruction with one step too many. Still friendly and short-ish."),
    "DISTRESS": ("gently touches something that could unsettle someone with dementia: a lost spouse "
                 "or home, a past job they can no longer do, time passing, 'going home', or illness. "
                 "Framed nostalgically, not grimly."),
    "INFANTILIZING": ("is slightly patronizing in word choice, praise, or activity: overly sing-song, "
                      "'we' language, praise for trivial things, or a childlike task presented sweetly."),
    "CORRECTIVE": ("nudges the person toward 'correct' reality or gently tests orientation (today's "
                   "date, where they are, who someone is), phrased as helpful or casual."),
}

SUBTLE_TEMPLATE = (
    "Write items that read like GOOD practice at first glance: warm, natural, well-phrased, "
    "personalized to the resident, and the kind of thing a kind volunteer would genuinely say. "
    "But each item contains ONE subtle problem: it {flaw} "
    "An experienced recreational therapist should only notice the problem on a second read. "
    "No obvious red flags, no exaggeration, and vary the wording across items. "
    "Do not label or explain the problem."
)

PROFILE_PROMPT = """Invent one FICTIONAL care-home resident profile as JSON with keys:
name, birth_year (1930-1955), hometown, career, hobbies (list), favourite_music (list),
family_notes, dementia_stage ("{stage}"). Vary culture, gender and background. Output JSON only."""

ITEM_PROMPT = """Resident profile (fictional):
{profile}

Write {n} distinct {ctype} items for an activity worker to use with this resident.
{mode_instruction}
Output a JSON list of strings only."""


class Claude:
    name = "claude"

    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

    def complete(self, prompt):
        msg = self.client.messages.create(model=self.model, max_tokens=4000,
                                          messages=[{"role": "user", "content": prompt}])
        # The response may start with a thinking block; keep only the text blocks.
        return "".join(b.text for b in msg.content if b.type == "text")


class OpenAIGen:
    name = "openai"

    def __init__(self):
        from openai import OpenAI
        self.model = os.getenv("OPENAI_MODEL")
        if not self.model:
            raise SystemExit("Set OPENAI_MODEL in .env (check openai.com for current model names).")
        self.client = OpenAI()

    def complete(self, prompt):
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content


PROVIDERS = {"claude": Claude, "openai": OpenAIGen}


def parse_json(text):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=PROVIDERS, default="claude")
    ap.add_argument("--n-profiles", type=int, default=5)
    ap.add_argument("--items-per-call", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)  # None = different each run
    args = ap.parse_args()
    random.seed(args.seed)

    gen = PROVIDERS[args.provider]()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        for _ in range(args.n_profiles):
            stage = random.choice(STAGES)
            print(f"[{gen.name}] generating profile ({stage} stage)...", flush=True)
            t0 = time.time()
            try:
                profile = parse_json(gen.complete(PROFILE_PROMPT.format(stage=stage)))
                print(f"  profile ok ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:  # malformed JSON etc.: skip, don't crash the run
                print("profile failed:", e); continue
            pid = f"{gen.name}-{uuid.uuid4().hex[:8]}"
            for ctype in CONTENT_TYPES:
                flaw = random.choice(list(FLAWS))
                batches = [("good", None, GOOD_INSTRUCTION),
                           ("subtly_bad", flaw, SUBTLE_TEMPLATE.format(flaw=FLAWS[flaw]))]
                for mode, intended_flaw, instr in batches:
                    print(f"  {ctype} / {mode}{' (' + intended_flaw + ')' if intended_flaw else ''}...", end=" ", flush=True)
                    t0 = time.time()
                    try:
                        items = parse_json(gen.complete(ITEM_PROMPT.format(
                            profile=json.dumps(profile), n=args.items_per_call,
                            ctype=ctype.replace("_", " "), mode_instruction=instr)))
                    except Exception as e:
                        print(f"failed ({time.time()-t0:.0f}s): {e}", flush=True); continue
                    print(f"{len(items)} items ({time.time()-t0:.0f}s)", flush=True)
                    for text in items:
                        f.write(json.dumps({"profile_id": pid, "stage": stage,
                                            "content_type": ctype, "gen_mode": mode,
                                            "intended_flaw": intended_flaw,
                                            "text": text, "provider": gen.name,
                                            "model": gen.model}) + "\n")
                    f.flush()
            print(f"profile {pid} ({stage}) done")


if __name__ == "__main__":
    main()
