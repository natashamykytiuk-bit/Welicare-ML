"""Print a random sample of generated items for manual review.

Usage: python src/peek.py [n] [--mode good|subtly_bad] [--provider claude|openai]
"""
import argparse, json, random
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int, nargs="?", default=20)
ap.add_argument("--mode")
ap.add_argument("--provider")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

with open("data/raw/candidates.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

print("Totals:", dict(Counter((r.get("provider", "?"), r["gen_mode"]) for r in rows)), "\n")

if args.mode:
    rows = [r for r in rows if r["gen_mode"] == args.mode]
if args.provider:
    rows = [r for r in rows if r.get("provider") == args.provider]

random.seed(args.seed)
for r in random.sample(rows, min(args.n, len(rows))):
    flaw = r.get("intended_flaw") or ""
    print(f"[{r['gen_mode']:10}] {flaw:13} {r['content_type']:20} ({r['stage']})\n    {r['text']}\n")
