#!/usr/bin/env python3
"""
calculate_mt5_perplexity.py — Compute mT5 perplexity for all documents in
globallaws_cleaned/ and write results to a CSV file.

Perplexity is computed via teacher-forced reconstruction loss on non-overlapping
512-token chunks: PPL = exp(total_nll / total_tokens).

USAGE
-----
    python scripts/calculate_mt5_perplexity.py
    python scripts/calculate_mt5_perplexity.py --input-dir globallaws_cleaned --output-csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

# Set HuggingFace cache before importing transformers.
_HF_ROOT = os.path.expanduser("~/mg61_scratch/chunheng/chunheng/huggingface")
os.makedirs(_HF_ROOT, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_ROOT)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_HF_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_HF_ROOT, "transformers"))

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "globallaws_cleaned"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "globallaws_mt5_perplexity.csv"
CSV_FIELDS = ("filename", "perplexity", "num_tokens", "num_chunks", "status")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute mT5 perplexity for text documents.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing .txt documents (default: globallaws_cleaned/)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path (default: globallaws_mt5_perplexity.csv)",
    )
    parser.add_argument(
        "--model-name",
        default="google/mt5-base",
        help="HuggingFace model id (default: google/mt5-base)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Chunk size in tokens (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for forward passes (default: 1)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (default: cuda if available, else cpu)",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chunk_token_ids(token_ids: list[int], max_length: int) -> list[list[int]]:
    """Split token ids into non-overlapping chunks."""
    return [token_ids[i : i + max_length] for i in range(0, len(token_ids), max_length)]


def compute_document_perplexity(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> tuple[float, int, int]:
    text = text.strip()
    if not text:
        raise ValueError("empty document")

    token_ids = tokenizer.encode(text, add_special_tokens=True)
    if len(token_ids) < 2:
        raise ValueError("document too short after tokenization")

    chunks = chunk_token_ids(token_ids, max_length)
    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            input_ids = torch.tensor(batch_chunks, dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            if outputs.loss is None:
                raise RuntimeError("model did not return a loss")

            chunk_token_counts = [len(chunk) for chunk in batch_chunks]
            batch_tokens = sum(chunk_token_counts)
            total_nll += outputs.loss.item() * batch_tokens
            total_tokens += batch_tokens

    perplexity = math.exp(total_nll / total_tokens)
    return perplexity, total_tokens, len(chunks)


def load_model_and_tokenizer(model_name: str, device: torch.device):
    print(f"Loading tokenizer and model: {model_name}")
    print(f"HuggingFace cache root: {_HF_ROOT}")
    # Use Auto* APIs for compatibility with transformers 5.x (MT5Tokenizer
    # is no longer exported from the top-level transformers package).
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model


def iter_text_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted(input_dir.glob("*.txt"))


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Input directory: {args.input_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Device: {device}")

    files = iter_text_files(args.input_dir)
    if not files:
        print(f"No .txt files found in {args.input_dir}", file=sys.stderr)
        return 1

    tokenizer, model = load_model_and_tokenizer(args.model_name, device)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        for path in tqdm(files, desc="Computing perplexity", unit="file"):
            row = {
                "filename": path.name,
                "perplexity": "",
                "num_tokens": "",
                "num_chunks": "",
                "status": "",
            }
            try:
                text = path.read_text(encoding="utf-8")
                perplexity, num_tokens, num_chunks = compute_document_perplexity(
                    text=text,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                )
                row["perplexity"] = f"{perplexity:.6f}"
                row["num_tokens"] = num_tokens
                row["num_chunks"] = num_chunks
                row["status"] = "ok"
                print(f"{path.name}: ppl={perplexity:.4f}, tokens={num_tokens}, chunks={num_chunks}")
            except Exception as exc:
                row["status"] = str(exc)
                print(f"{path.name}: ERROR - {exc}", file=sys.stderr)

            writer.writerow(row)
            csv_file.flush()

    print(f"Done. Wrote {len(files)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
