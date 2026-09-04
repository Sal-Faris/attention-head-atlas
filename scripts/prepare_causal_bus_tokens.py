"""Prepare a fresh held-out document sample for causal bus confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def excluded_rows(paths: list[Path]) -> set[int]:
    rows: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as bundle:
            for key in bundle.files:
                if key.endswith("dataset_rows"):
                    rows.update(int(value) for value in np.asarray(bundle[key]).ravel())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "D:/Laptop/AI/model-cache/huggingface/"
            "datasets--NeelNanda--pile-10k/snapshots/"
            "127bfedcd5047750df5ccf3a12979a47bfa0bafa/"
            "data/train-00000-of-00001-4746b8785c874cc7.parquet"
        ),
    )
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260904)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/causal_bus_fresh_tokens_v1.npz"),
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == "step143000")
    tokenizer = AutoTokenizer.from_pretrained(Path(record["snapshot"]), local_files_only=True)
    exclusions = excluded_rows(
        [
            Path("artifacts/pythia-70m-deduped/activation_validation_pilot.npz"),
            Path(
                "artifacts/pythia-70m-deduped/"
                "qk_bilinear_margin_confirmation_v1.npz"
            ),
            Path("artifacts/pythia-70m-deduped/qk_conditional_events_v1.npz"),
        ]
    )
    texts = pd.read_parquet(args.dataset, columns=["text"])["text"]
    prefix = tokenizer.bos_token_id
    if prefix is None:
        prefix = tokenizer.eos_token_id
    if prefix is None:
        raise ValueError("tokenizer has neither BOS nor EOS token")
    rng = np.random.default_rng(args.seed)
    selected_tokens = []
    selected_rows = []
    for value in rng.permutation(len(texts)):
        row = int(value)
        if row in exclusions:
            continue
        token_ids = tokenizer(
            str(texts.iloc[row])[:8192],
            add_special_tokens=False,
            truncation=True,
            max_length=args.sequence_length - 1,
        )["input_ids"]
        if len(token_ids) != args.sequence_length - 1:
            continue
        selected_tokens.append([int(prefix), *(int(token) for token in token_ids)])
        selected_rows.append(row)
        if len(selected_tokens) == args.sequences:
            break
    if len(selected_tokens) != args.sequences:
        raise RuntimeError("not enough unused full-length documents")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        confirmation_tokens=np.asarray(selected_tokens, dtype=np.int64),
        confirmation_dataset_rows=np.asarray(selected_rows, dtype=np.int64),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "dataset": "NeelNanda/pile-10k",
                    "dataset_revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
                    "excluded_prior_rows": len(exclusions),
                    "seed": args.seed,
                    "sequence_length": args.sequence_length,
                },
                sort_keys=True,
            )
        ),
    )
    print(f"saved {len(selected_tokens)} unused documents to {args.output}")


if __name__ == "__main__":
    main()
