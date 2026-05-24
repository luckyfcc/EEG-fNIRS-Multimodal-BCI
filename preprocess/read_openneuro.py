#!/usr/bin/env python3
"""Read OpenNeuro ds004022 EEG data with MNE.

This module keeps raw data loading separate from analysis code so experiments
can reuse the same, documented import path. EEG is read through ``mne-bids``
when possible, which preserves BIDS metadata. A fNIRS entry point is included
as a placeholder for future NIRx/SNIRF support.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids


DATASET_ID = "ds004022"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIDS_ROOT = REPO_ROOT / "data" / "openneuro" / DATASET_ID


def list_subjects(bids_root: Path) -> list[str]:
    """Return available BIDS subject labels without the ``sub-`` prefix."""
    if not bids_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {bids_root}. "
            "Run scripts/download_openneuro.py first."
        )
    return sorted(get_entity_vals(bids_root, "subject"))


def infer_first_eeg_recording(bids_root: Path, subject: str | None = None) -> BIDSPath:
    """Build a BIDSPath for the first EEG recording found in the dataset."""
    subjects = list_subjects(bids_root)
    if not subjects:
        raise FileNotFoundError(f"No BIDS subjects found under {bids_root}")

    subject = subject or subjects[0]
    if subject not in subjects:
        raise ValueError(f"Subject {subject!r} not found. Available subjects: {subjects}")

    eeg_dir = bids_root / f"sub-{subject}" / "eeg"
    eeg_files = sorted(
        path
        for ext in ("*.vhdr", "*.edf", "*.bdf", "*.set", "*.fif")
        for path in eeg_dir.glob(ext)
    )
    if not eeg_files:
        raise FileNotFoundError(f"No EEG recording files found in {eeg_dir}")

    return BIDSPath(root=bids_root, subject=subject, datatype="eeg", suffix="eeg").match()[0]


def read_eeg(
    bids_root: Path = DEFAULT_BIDS_ROOT,
    subject: str | None = None,
    preload: bool = False,
) -> mne.io.BaseRaw:
    """Read an EEG recording as an MNE Raw object.

    Parameters
    ----------
    bids_root:
        Root directory of the downloaded BIDS dataset.
    subject:
        BIDS subject label without the ``sub-`` prefix. When omitted, the first
        available subject is used for a quick smoke test.
    preload:
        Whether to preload signal data into memory. Leave ``False`` for large
        raw files unless downstream processing requires random access.
    """
    bids_path = infer_first_eeg_recording(bids_root, subject)
    raw = read_raw_bids(bids_path=bids_path, verbose=True)

    if preload:
        raw.load_data()

    return raw


def read_fnirs(
    bids_root: Path = DEFAULT_BIDS_ROOT,
    subject: str | None = None,
    preload: bool = False,
) -> mne.io.BaseRaw:
    """Reserved fNIRS reader interface for ds004022.

    Future implementation can dispatch to ``mne.io.read_raw_nirx`` or
    ``mne.io.read_raw_snirf`` after confirming the dataset's fNIRS file format
    and channel metadata. Keeping the function signature aligned with
    ``read_eeg`` makes multimodal preprocessing scripts easier to reproduce.
    """
    del bids_root, subject, preload
    raise NotImplementedError(
        "fNIRS reading is reserved for a future implementation after verifying "
        "the ds004022 fNIRS file format and metadata."
    )


def summarize_raw(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Create a compact channel summary for reproducibility logs."""
    channel_types = raw.get_channel_types()
    return pd.DataFrame(
        {
            "channel": raw.ch_names,
            "type": channel_types,
            "sfreq": raw.info["sfreq"],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Read EEG data from OpenNeuro {DATASET_ID}.")
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=DEFAULT_BIDS_ROOT,
        help="Path to the downloaded BIDS dataset.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Subject label without the sub- prefix. Defaults to the first available subject.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load raw EEG samples into memory.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional path to save a channel summary CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = read_eeg(bids_root=args.bids_root.resolve(), subject=args.subject, preload=args.preload)

    print(raw)
    print(raw.info)

    summary = summarize_raw(raw)
    print(summary.head())

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)
        print(f"Saved channel summary: {args.summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
