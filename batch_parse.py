#!/usr/bin/env python3
"""
Batch-parse composite edit prompts into atomic sub-instructions (JSON per prompt).

Ollama (local, free):
  ./.venv/bin/python batch_parse.py \
    --backend ollama --ollama-model llama3.1:8b-instruct-q4_K_M \
    --prompts prompts.txt --outdir artifacts/parsed

OpenAI (optional):
  export OPENAI_API_KEY="sk-..."
  ./.venv/bin/python batch_parse.py \
    --backend openai --openai-model gpt-4o \
    --prompts prompts.txt --outdir artifacts/parsed
"""
import argparse, json, time
from pathlib import Path
from atomic_edits.parser.parse import parse_with_ollama, parse_with_openai

def main():
    ap = argparse.ArgumentParser(description="Batch parse prompts to atomic steps (JSON).")
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--outdir", default="artifacts/parsed")
    ap.add_argument("--backend", choices=["ollama","openai"], default="ollama")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--openai-model", default="gpt-4o")
    ap.add_argument("--sleep", type=float, default=0.10)
    args = ap.parse_args()

    lines = [l.strip() for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    faildir = Path("artifacts/failed"); faildir.mkdir(parents=True, exist_ok=True)


    for idx, instr in enumerate(lines, 1):
        try:
            if args.backend == "ollama":
                if not args.ollama_model:
                    raise SystemExit("Please provide --ollama-model for the ollama backend.")
                result = parse_with_ollama(instr, ollama_model=args.ollama_model)
            else:
                result = parse_with_openai(instr, openai_model=args.openai_model)

            out = outdir / f"{idx:03d}.json"
            out.write_text(json.dumps(result.to_dict(), indent=2))
            print(f"[{idx:03d}/{len(lines)}] saved -> {out}")
        except Exception as e:
            (faildir / f"{idx:03d}.prompt.txt").write_text(instr)
            (faildir / f"{idx:03d}.error.txt").write_text(repr(e))
            print(f"[{idx:03d}/{len(lines)}] FAILED -> {e}")
        time.sleep(args.sleep)

if __name__ == "__main__":
    main()
