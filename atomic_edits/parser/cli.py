#!/usr/bin/env python3
import argparse, json, sys
from .parse import parse_with_ollama, parse_with_openai

def main():
    ap = argparse.ArgumentParser(description="Atomic Edits - Instruction Parser (LLM planner)")
    ap.add_argument("--text", "-t", type=str, help="Instruction text to parse. If omitted, reads stdin.", default=None)
    ap.add_argument("--backend", "-b", choices=["ollama", "openai"], default="ollama",
                    help="Which LLM backend to use (open-source via Ollama, or GPT-4 via OpenAI).")
    ap.add_argument("--ollama-model", type=str, default=None, help="MODEL_NAME for Ollama (required if backend=ollama).")
    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama server URL.")
    ap.add_argument("--openai-model", type=str, default="gpt-4", help="OpenAI model name (if backend=openai).")

    args = ap.parse_args()

    text = args.text
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        print("No instruction text provided.", file=sys.stderr)
        sys.exit(2)

    if args.backend == "ollama":
        if not args.ollama_model:
            print("Please provide --ollama-model (your chosen open-source alternative).", file=sys.stderr)
            sys.exit(2)
        result = parse_with_ollama(text, ollama_model=args.ollama_model, url=args.ollama_url)
    else:
        result = parse_with_openai(text, openai_model=args.openai_model)

    print(json.dumps(result.to_dict(), indent=2))

if __name__ == "__main__":
    main()
