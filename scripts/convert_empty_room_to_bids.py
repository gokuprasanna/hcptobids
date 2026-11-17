#!/usr/bin/env python3
"""
Convert only the unprocessed empty room (Rnoise/Pnoise) HCP MEG recordings to
BIDS, without the task MEG or MRI. Uses the same conversion as hcp2bids.py.

Usage:
    python scripts/convert_empty_room_to_bids.py <emptyroom_root> <bids_root>

Where:
    <emptyroom_root> contains <subject>/unprocessed/MEG/*-Rnoise and *-Pnoise
    <bids_root> is the root directory for the BIDS dataset
"""

import argparse
from pathlib import Path

from hcp2bids import convert_subject_noise


def convert_empty_room_to_bids(emptyroom_root: Path, bids_root: Path):
    """Convert the empty room MEG data of every subject to BIDS."""
    subjects = sorted(d.name for d in emptyroom_root.iterdir() if d.is_dir() and d.name.isdigit())
    if not subjects:
        print(f"No subject directories found in {emptyroom_root}")
        return

    for subject in subjects:
        convert_subject_noise(emptyroom_root, subject, bids_root)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HCP empty room (Rnoise/Pnoise) MEG recordings to BIDS"
    )
    parser.add_argument(
        "emptyroom_root", type=Path,
        help="Directory containing <subject>/unprocessed/MEG/*-Rnoise and *-Pnoise",
    )
    parser.add_argument("bids_root", type=Path, help="Output BIDS root directory")
    args = parser.parse_args()

    if not args.emptyroom_root.exists():
        parser.error(f"Emptyroom root directory {args.emptyroom_root} does not exist")

    args.bids_root.mkdir(parents=True, exist_ok=True)
    convert_empty_room_to_bids(args.emptyroom_root, args.bids_root)


if __name__ == "__main__":
    main()
