#!/usr/bin/env python3
"""Download the OpenNeuro ds004022 EEG-fNIRS dataset.

The script stores the dataset under ``data/openneuro/ds004022/`` so that raw
data are kept outside normal source-control history. For reproducibility, the
preferred backend is DataLad, which preserves OpenNeuro's git-annex metadata.
If DataLad is unavailable, the script falls back to the optional
``openneuro-py`` package.

Examples
--------
Download or update the dataset:

    python scripts/download_openneuro.py

Download a specific OpenNeuro snapshot:

    python scripts/download_openneuro.py --snapshot 1.0.0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_ID = "ds004022"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "openneuro" / DATASET_ID


def run_command(command: list[str]) -> None:
    """Run an external command and surface a concise failure message."""
    print("+ " + " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {command}") from exc


def download_with_datalad(output_dir: Path, snapshot: str | None) -> bool:
    """Download the dataset with DataLad when it is available.

    DataLad is the most reproducible option for OpenNeuro because it can track
    exact dataset snapshots and retrieve git-annexed file contents on demand.
    """
    if shutil.which("datalad") is None:
        return False

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_url = f"https://github.com/OpenNeuroDatasets/{DATASET_ID}.git"

    if not (output_dir / ".git").exists():
        run_command(["datalad", "clone", dataset_url, str(output_dir)])

    if snapshot:
        run_command(["datalad", "-C", str(output_dir), "checkout", snapshot])

    # Retrieve all annexed data files so downstream scripts can read locally.
    run_command(["datalad", "-C", str(output_dir), "get", "."])
    return True


def download_with_openneuro_py(output_dir: Path, snapshot: str | None) -> bool:
    """Download the dataset with openneuro-py when installed."""
    try:
        import openneuro  # type: ignore[import-not-found]
    except ImportError:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {
        "dataset": DATASET_ID,
        "target_dir": str(output_dir),
    }
    if snapshot:
        kwargs["snapshot"] = snapshot

    openneuro.download(**kwargs)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Download OpenNeuro dataset {DATASET_ID} to data/openneuro/{DATASET_ID}/."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for the dataset.",
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Optional OpenNeuro snapshot/tag, for example 1.0.0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    print(f"Dataset: {DATASET_ID}")
    print(f"Destination: {output_dir}")

    if download_with_datalad(output_dir, args.snapshot):
        print("Download completed with DataLad.")
        return 0

    if download_with_openneuro_py(output_dir, args.snapshot):
        print("Download completed with openneuro-py.")
        return 0

    print(
        "\nNo supported OpenNeuro download backend was found.\n"
        "Install one of the following and rerun this script:\n"
        "  - datalad:      https://www.datalad.org/\n"
        "  - openneuro-py:  pip install openneuro-py\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
