"""Download selected Pythia checkpoints and extract compact operator factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from head_atlas.factor_io import (
    extract_processed_factors_from_safetensors,
    save_factor_bundle,
)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pythia70m_deduped_pilot.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument("--only", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = config["checkpoints"] if args.only is None else args.only
    unknown = sorted(set(selected) - set(config["checkpoints"]))
    if unknown:
        raise ValueError(f"checkpoints are outside the frozen pilot: {unknown}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit('Install model dependencies with: pip install -e ".[models]"') from error

    records = []
    expected = config["expected_architecture"]
    for revision in selected:
        print(f"acquiring {config['model']} at {revision}", flush=True)
        snapshot = Path(
            snapshot_download(
                repo_id=config["model"],
                revision=revision,
                cache_dir=config["cache_root"],
                allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"],
            )
        )
        checkpoint_output = args.output_root / revision
        record = {
            "revision": revision,
            "snapshot_commit": snapshot.name,
            "snapshot": str(snapshot.resolve()),
            "factors": {},
        }
        for kind in config["operator_views"]:
            operators, snapshot_metadata = extract_processed_factors_from_safetensors(
                snapshot, kind
            )
            observed = {
                "layers": snapshot_metadata["n_layers"],
                "heads": snapshot_metadata["n_heads"],
                "model_width": snapshot_metadata["d_model"],
                "head_width": snapshot_metadata["d_head"],
            }
            if observed != expected:
                raise RuntimeError(
                    f"architecture mismatch at {revision}: {observed} != {expected}"
                )
            output = checkpoint_output / f"{kind.lower()}_factors.npz"
            save_factor_bundle(
                output,
                operators,
                {
                    **snapshot_metadata,
                    "model": config["model"],
                    "revision": revision,
                    "snapshot_commit": snapshot.name,
                    "git_commit": git_commit(),
                },
            )
            record["factors"][kind] = {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "operator_count": len(operators),
            }
            print(f"extracted {len(operators)} {kind} operators", flush=True)
        records.append(record)

    manifest = {
        "experiment_id": config["experiment_id"],
        "model": config["model"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config": str(args.config),
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved acquisition manifest to {args.manifest}")


if __name__ == "__main__":
    main()
