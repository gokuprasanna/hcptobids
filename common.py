"""
Constants and helpers shared by the hcptobids scripts.

Coordinate frames: ``read_raw_bti(convert=True)`` moves the sensors from the
4D device frame into a Neuromag-style device frame with a 90 deg rotation
about z plus ``BTI_DEV_TRANSLATION`` (m, MNE's default). The same rotation
takes the 4D head frame (x to nasion, y to left ear, z up) to an RAS head
frame (x to right ear, y to nasion, z up). That RAS frame is the "head" frame
of ``dev_head_t`` and ``trans.fif`` throughout this project.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np


BTI_DEV_TRANSLATION = (0.0, 0.02, 0.11)
BTI_HEAD_TO_HEAD = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def get_subjects_dir(bids_root: Path, subjects_dir: Optional[Path] = None) -> Path:
    """``subjects_dir`` if given, else ``<bids_root>/derivatives/freesurfer``."""
    return Path(subjects_dir) if subjects_dir else Path(bids_root) / "derivatives" / "freesurfer"


def find_subjects(bids_root: Path) -> List[str]:
    """IDs (without ``sub-``) of the BIDS subjects, excluding ``sub-emptyroom``."""
    return [d.name[4:] for d in sorted(Path(bids_root).glob("sub-*"))
            if d.is_dir() and d.name != "sub-emptyroom"]


def setup_freesurfer(path: Optional[str] = None, subjects_dir: Optional[Path] = None) -> str:
    """
    Point the environment at FreeSurfer and return its install folder.

    The folder is ``path`` if given, else ``$FREESURFER_HOME``. Sets
    ``FREESURFER_HOME``, puts its ``bin/`` on ``PATH`` and, if given, sets
    ``SUBJECTS_DIR``.

    Raises RuntimeError if the folder doesn't contain SetUpFreeSurfer.sh.
    """
    home = path or os.environ.get("FREESURFER_HOME")
    if not home or not (Path(home) / "SetUpFreeSurfer.sh").exists():
        raise RuntimeError(
            "FreeSurfer not found. Set the FREESURFER_HOME environment variable or pass "
            "--freesurfer-home (the folder containing SetUpFreeSurfer.sh). "
            "See docs/INSTALL.md."
        )
    home = str(home)
    os.environ["FREESURFER_HOME"] = home
    fs_bin = f"{home}/bin"
    if fs_bin not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = fs_bin + os.pathsep + os.environ["PATH"]
    if subjects_dir is not None:
        os.environ["SUBJECTS_DIR"] = str(subjects_dir)
    return home
