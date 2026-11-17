"""
Build the MNE head model for subjects reconstructed by run_freesurfer.py.

For each subject, writes to ``<SUBJECTS_DIR>/sub-<id>/bem/`` (same names as
the Appleseed dataset):

    watershed/, inner_skull.surf, outer_skull.surf, outer_skin.surf
    sub-<id>-head.fif                     scalp, used for visual checks
    sub-<id>-inner_skull-bem.fif          single-layer BEM (enough for MEG)
    sub-<id>-inner_skull-bem-sol.fif      its solution, for the forward model
    sub-<id>-ico-4-src.fif                cortical source space

It also links FreeSurfer's fsaverage into SUBJECTS_DIR so source estimates
can be morphed to a common space.

Usage:
    python make_head_model.py <bids_root> [--subjects-dir <SUBJECTS_DIR>]
        [--subjects ID ...] [--freesurfer-home <FREESURFER_HOME>]

SUBJECTS_DIR defaults to <bids_root>/derivatives/freesurfer. FreeSurfer (for
the watershed BEM) is found through --freesurfer-home or the FREESURFER_HOME
environment variable.
"""

import argparse
from pathlib import Path

import mne
from mne.bem import make_watershed_bem

from common import find_subjects, get_subjects_dir, setup_freesurfer


BEM_ICO = 4                 # 5120 triangles, as in Appleseed's BEM
BEM_CONDUCTIVITY = (0.3,)   # single layer: brain only
SRC_SPACING = "ico4"


def link_fsaverage(subjects_dir: Path, freesurfer_home: str):
    """Symlink FreeSurfer's fsaverage into ``subjects_dir`` if it isn't there."""
    target = subjects_dir / "fsaverage"
    if not target.exists():
        target.symlink_to(Path(freesurfer_home) / "subjects" / "fsaverage")
        print(f"Linked fsaverage into {subjects_dir}")


def make_head_model(subject_id: str, subjects_dir: Path):
    """Make the BEM surfaces, BEM solution and source space for one subject."""
    fs_subject = f"sub-{subject_id}"
    bem_dir = subjects_dir / fs_subject / "bem"
    print(f"\n=== Head model for {fs_subject} ===")

    print(" Watershed BEM surfaces")
    make_watershed_bem(fs_subject, subjects_dir=subjects_dir, overwrite=True)

    print(" Single-layer BEM")
    surfaces = mne.make_bem_model(fs_subject, ico=BEM_ICO, conductivity=BEM_CONDUCTIVITY,
                                  subjects_dir=subjects_dir)
    mne.write_bem_surfaces(bem_dir / f"{fs_subject}-inner_skull-bem.fif", surfaces, overwrite=True)
    solution = mne.make_bem_solution(surfaces)
    mne.write_bem_solution(bem_dir / f"{fs_subject}-inner_skull-bem-sol.fif", solution,
                           overwrite=True)

    print(f" Source space ({SRC_SPACING})")
    src = mne.setup_source_space(fs_subject, spacing=SRC_SPACING, add_dist="patch",
                                 subjects_dir=subjects_dir)
    mne.write_source_spaces(bem_dir / f"{fs_subject}-ico-4-src.fif", src, overwrite=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build BEM and source space for subjects reconstructed by run_freesurfer.py"
    )
    parser.add_argument("bids_root", type=Path, help="BIDS root directory")
    parser.add_argument("--subjects-dir", type=Path, default=None,
                        help="FreeSurfer SUBJECTS_DIR (default: <bids_root>/derivatives/freesurfer)")
    parser.add_argument("--subjects", nargs="+",
                        help="Subject IDs (default: every BIDS subject with a finished recon-all)")
    parser.add_argument("--freesurfer-home", default=None,
                        help="FreeSurfer install folder (default: $FREESURFER_HOME)")
    args = parser.parse_args()

    subjects_dir = get_subjects_dir(args.bids_root, args.subjects_dir)
    try:
        freesurfer_home = setup_freesurfer(args.freesurfer_home, subjects_dir)
    except RuntimeError as e:
        parser.error(str(e))
    subjects = args.subjects or [
        s for s in find_subjects(args.bids_root)
        if (subjects_dir / f"sub-{s}" / "surf" / "lh.white").exists()
    ]
    link_fsaverage(subjects_dir, freesurfer_home)
    for subject_id in subjects:
        make_head_model(subject_id, subjects_dir)


if __name__ == "__main__":
    main()
