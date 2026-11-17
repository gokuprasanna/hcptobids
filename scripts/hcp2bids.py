"""
Convert unprocessed HCP MEG (StoryM task + Rnoise/Pnoise empty room) and
structural MRI (T1w/T2w) data to BIDS.

Usage:
    python scripts/hcp2bids.py <meg_root> <mri_root> <bids_root>
        [--noise-root <emptyroom_root>] [--subjects ID ...]

Head position: unprocessed HCP MEG has no digitized head shape or fiducials,
so MNE can't build the device->head transform itself. It is rebuilt here from
the 4D device->head transform stored in each run (see restore_head_position).
The head->MRI transform comes from HCP's MEG anatomy package; see
create_transforms.py.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import List, Optional

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, write_anat, write_raw_bids

from common import BTI_DEV_TRANSLATION, BTI_HEAD_TO_HEAD


TASK_NAME = "storymath"

EVENT_COLUMN_DESCRIPTIONS = {
    "trial_type": "Experimental condition or task label for each event",
    "trigger_code": "Numeric trigger code associated with the task event",
}


# ---------------------------------------------------------------------------
# Reading HCP 4D/BTi data
# ---------------------------------------------------------------------------

def read_hcp_bti(run_dir: Path) -> mne.io.Raw:
    """Read the 4D/BTi MEG data in ``<run_dir>/4D``, with bytes decoded to str."""
    fourd_dir = run_dir / "4D"
    raw = mne.io.read_raw_bti(
        pdf_fname=fourd_dir / "c,rfDC",
        config_fname=fourd_dir / "config",
        head_shape_fname=None,
        rotation_x=0.0,
        translation=BTI_DEV_TRANSLATION,
        preload=True,
    )
    sanitize_bti_raw(raw)
    return raw


def restore_head_position(raw: mne.io.Raw):
    """
    Set ``raw.info['dev_head_t']`` from the 4D device->head transform, in place.

    Without a head shape file, ``read_raw_bti`` sets ``dev_head_t`` to the
    identity, as if the head sat at the device origin. The real head position
    of the run is in ``raw.info['dev_ctf_t']`` (4D device -> 4D head). This
    composes it with the reader's device conversion (4D -> Neuromag device:
    the 4D -> RAS rotation plus BTI_DEV_TRANSLATION) and the 4D -> RAS head
    rotation, the same way MNE does when a head shape is available.
    """
    bti_dev_to_neuromag_dev = BTI_HEAD_TO_HEAD.copy()
    bti_dev_to_neuromag_dev[:3, 3] = BTI_DEV_TRANSLATION
    dev_head = (BTI_HEAD_TO_HEAD @ raw.info["dev_ctf_t"]["trans"]
                @ np.linalg.inv(bti_dev_to_neuromag_dev))
    raw.info["dev_head_t"] = mne.transforms.Transform("meg", "head", dev_head)


def _to_str(x):
    """Recursively decode bytes (also inside lists, dicts and arrays) to str."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, list):
        return [_to_str(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_str(v) for k, v in x.items()}
    if isinstance(x, np.ndarray) and x.dtype.kind in ("S", "a"):
        return np.array([_to_str(v) for v in x], dtype=object)
    return x


def sanitize_bti_raw(raw: mne.io.Raw):
    """Decode bytes left in a BTi ``raw.info`` and annotations, in place."""
    read_only_keys = set(getattr(raw.info, "_attributes", {}))
    for key, val in raw.info.items():
        if key not in read_only_keys:
            raw.info[key] = _to_str(val)

    if raw.annotations is not None:
        raw.annotations.description = [_to_str(d) for d in raw.annotations.description]


# ---------------------------------------------------------------------------
# Writing BIDS
# ---------------------------------------------------------------------------

def write_bids_fif(raw: mne.io.Raw, bids_path: BIDSPath, **kwargs):
    """
    Write ``raw`` to BIDS in FIF format; kwargs go to ``write_raw_bids``.

    The BTi reader keeps the data as float64, which MNE-BIDS would write as
    double precision, splitting long runs into several files. Saving to a
    temporary FIF first stores it as float32, like any other FIF recording.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        fif_fname = Path(tmp_dir) / "intermediate_raw.fif"
        raw.save(fif_fname)
        write_raw_bids(mne.io.read_raw_fif(fif_fname), bids_path=bids_path, overwrite=True,
                       format="FIF", **kwargs)


def update_json_sidecar(json_path: Path, fields: dict):
    """Add ``fields`` to a JSON sidecar, keeping whatever is already in it."""
    sidecar = json.loads(json_path.read_text()) if json_path.exists() else {}
    sidecar.update(fields)
    json_path.write_text(json.dumps(sidecar, indent=4))


# ---------------------------------------------------------------------------
# Task MEG
# ---------------------------------------------------------------------------

def load_events(raw: mne.io.Raw, eprime_tab: Path):
    """
    Get task events, from the MEG trigger channel if present, else EPRIME.

    Parameters
    ----------
    raw : mne.io.Raw
        Task MEG recording.
    eprime_tab : Path
        EPRIME ``.tab`` file for the run, used when ``raw`` has no stim channel.

    Returns
    -------
    events : ndarray, shape (n_events, 3)
        MNE events array.
    event_metadata : DataFrame
        ``trial_type`` and ``trigger_code`` for each event.
    timing_source : str
        ``"meg_trigger"`` or ``"eprime"``.
    """
    if len(mne.pick_types(raw.info, stim=True)) > 0:
        print("  → Using MEG trigger channel")
        events = mne.find_events(raw)
        codes = events[:, 2]
        event_metadata = pd.DataFrame({
            "trial_type": codes.astype(str),
            "trigger_code": codes.astype(int),
        })
        return events, event_metadata, "meg_trigger"

    print("  → Using EPRIME timing (assumed aligned)")
    df = pd.read_csv(eprime_tab, sep="\t")
    onsets_sec = df["Trigger.OnsetTime"] / 1000.0
    codes = df["Trigger.Code"].astype(int)

    events = np.zeros((len(df), 3), dtype=int)
    events[:, 0] = (onsets_sec * raw.info["sfreq"]).astype(int)
    events[:, 2] = codes

    trial_type = df["Condition"] if "Condition" in df.columns else codes
    event_metadata = pd.DataFrame({
        "trial_type": trial_type.astype(str),
        "trigger_code": codes,
    }).fillna("n/a")
    return events, event_metadata, "eprime"


def convert_subject_meg(meg_subject_dir: Path, bids_root: Path,
                        empty_room: Optional[BIDSPath] = None):
    """
    Convert the StoryM task MEG runs of one subject.

    If ``empty_room`` is given, it is recorded as ``AssociatedEmptyRoom`` in
    each run's MEG sidecar.
    """
    subject = meg_subject_dir.name
    print(f"\n=== Converting MEG for subject {subject} ===")

    for run_dir in sorted((meg_subject_dir / "unprocessed" / "MEG").glob("*-StoryM")):
        run_label = run_dir.name.split("-")[0]  # "8-StoryM" -> "8"
        print(f" Run {run_label}")

        eprime_files = list((run_dir / "EPRIME").glob("*.tab"))
        if len(eprime_files) != 1:
            raise RuntimeError(f"Expected 1 EPRIME file in {run_dir}")

        raw = read_hcp_bti(run_dir)
        restore_head_position(raw)
        events, event_metadata, timing_source = load_events(raw, eprime_files[0])

        bids_path = BIDSPath(
            subject=subject,
            task=TASK_NAME,
            run=run_label,
            datatype="meg",
            root=bids_root,
        )
        write_bids_fif(
            raw,
            bids_path,
            events=events,
            event_metadata=event_metadata,
            extra_columns_descriptions=EVENT_COLUMN_DESCRIPTIONS,
            empty_room=empty_room,
        )

        events_json = bids_path.copy().update(suffix="events", extension=".json").fpath
        update_json_sidecar(events_json, {
            "TimingSource": timing_source,
            "ClockAlignmentAssumption": "EPRIME timestamps are assumed aligned to the MEG acquisition clock",
            "HCPNote": "HCP MEG data do not include digitized headshape or fiducials",
        })


# ---------------------------------------------------------------------------
# Empty room MEG
# ---------------------------------------------------------------------------

def convert_subject_noise(noise_root: Path, subject: str, bids_root: Path) -> dict:
    """
    Convert one subject's noise recordings to BIDS.

    HCP records two per session: Rnoise, the empty room, and Pnoise, with the
    participant in the MEG but no task (mne-hcp calls them
    ``noise_empty_room`` and ``noise_subject``).

    Uses the MNE-BIDS empty room convention: ``subject='emptyroom'``,
    ``task='noise'``, ``session=<recording date>``, ``acq=rnoise|pnoise``.
    Rnoise and Pnoise are recorded on the same day, so they are told apart by
    ``acq`` rather than ``run``: they are different kinds of noise recording,
    not repeats of one.

    Expects raw 4D/BTi data at::

        <noise_root>/<subject>/unprocessed/MEG/*-Rnoise/4D/{c,rfDC,config}
        <noise_root>/<subject>/unprocessed/MEG/*-Pnoise/4D/{c,rfDC,config}

    The HCP ``<subject>/MEG/{Rnoise,Pnoise}/datacheck`` folders are QC output
    only and contain no raw signal.

    Returns
    -------
    dict
        The BIDSPath written for each recording, keyed ``"rnoise"``/``"pnoise"``.
        Empty if nothing was found.
    """
    written = {}
    print(f"\n=== Converting noise (emptyroom) recordings for subject {subject} ===")

    unprocessed_meg = noise_root / subject / "unprocessed" / "MEG"
    if not unprocessed_meg.exists():
        print(f" No unprocessed MEG directory found for subject {subject} under {noise_root}")
        return written

    noise_dirs = sorted(unprocessed_meg.glob("*-Rnoise")) + sorted(unprocessed_meg.glob("*-Pnoise"))
    if not noise_dirs:
        print(f" No Rnoise/Pnoise 4D data found under {unprocessed_meg}")
        return written

    for run_dir in noise_dirs:
        noise_type = run_dir.name.split("-")[1]  # "1-Rnoise" -> "Rnoise"
        print(f" Noise run: {run_dir.name}")

        raw = read_hcp_bti(run_dir)

        meas_date = raw.info.get("meas_date")
        session = meas_date.strftime("%Y%m%d") if meas_date is not None else "1"

        bids_path = BIDSPath(
            subject="emptyroom",
            session=session,
            task="noise",
            acquisition=noise_type.lower(),
            datatype="meg",
            root=bids_root,
        )
        write_bids_fif(raw, bids_path)
        written[noise_type.lower()] = bids_path
    return written


# ---------------------------------------------------------------------------
# Structural MRI
# ---------------------------------------------------------------------------

def find_anat_images(mri_root: Path) -> dict:
    """
    Find the T1w and T2w NIfTI images under ``mri_root``.

    Returns
    -------
    dict
        ``{"T1w": [paths], "T2w": [paths]}``, each list sorted.
    """
    images = {"T1w": [], "T2w": []}
    for nii in sorted(mri_root.rglob("*.nii*")):
        # Match "t1w"/"t2w" rather than "t1"/"t2": every HCP filename contains
        # "3T", including the AFI/BIAS/FieldMap files next to the real scans.
        name = nii.name.lower()
        if "t1w" in name:
            images["T1w"].append(nii)
        elif "t2w" in name:
            images["T2w"].append(nii)
        else:
            print(f" Skipping non-standard MRI: {nii.name}")
    return images


def convert_subject_mri(structural_root: Path, subject: str, bids_root: Path):
    """
    Convert one subject's raw T1w/T2w scans to BIDS anat.

    run_freesurfer.py reconstructs the T1w from here, so the FreeSurfer
    subject shares the scanner space of these images.
    """
    print(f"\n=== Converting MRI for subject {subject} ===")

    mri_root = structural_root / subject / "unprocessed"
    if not mri_root.exists():
        print(f" No MRI directory found for subject {subject}")
        return

    for suffix, images in find_anat_images(mri_root).items():
        if not images:
            print(f" No {suffix} found for subject {subject}")
            continue

        # HCP often has 2 MPRAGE (and sometimes 2 SPACE) scans per subject.
        # Each gets its own run- entity.
        multi_run = len(images) > 1
        for run, nii in enumerate(images, start=1):
            bids_path = BIDSPath(subject=subject, root=bids_root, suffix=suffix, datatype="anat")
            if multi_run:
                bids_path.update(run=run)

            write_anat(image=nii, bids_path=bids_path, overwrite=True)
            run_note = f" run-{run}" if multi_run else ""
            print(f" Wrote {suffix}{run_note} from {nii.name}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def convert_subject(meg_subject_dir: Path, structural_root: Path, bids_root: Path,
                    noise_root: Optional[Path] = None):
    """Convert the task MEG, empty room MEG (optional) and MRI of one subject."""
    subject = meg_subject_dir.name
    noise_paths = {}
    if noise_root is not None:
        noise_paths = convert_subject_noise(noise_root, subject, bids_root)
    # Rnoise is the true empty room recording (Pnoise has the participant in
    # the MEG), so it's the one linked to the task runs.
    convert_subject_meg(meg_subject_dir, bids_root, empty_room=noise_paths.get("rnoise"))
    convert_subject_mri(structural_root, subject, bids_root)


def convert_dataset(meg_root: Path, mri_root: Path, bids_root: Path,
                    noise_root: Optional[Path] = None,
                    subjects: Optional[List[str]] = None):
    """
    Convert every subject under ``meg_root`` to BIDS.

    Parameters
    ----------
    meg_root : Path
        HCP MEG root, containing ``<subject>/unprocessed/MEG/*-StoryM``.
    mri_root : Path
        HCP structural root, containing ``<subject>/unprocessed/3T``.
    bids_root : Path
        Output BIDS root.
    noise_root : Path, optional
        HCP empty room root, containing
        ``<subject>/unprocessed/MEG/*-Rnoise`` and ``*-Pnoise``. If None,
        noise recordings are not converted.
    subjects : list of str, optional
        Subject IDs to convert. If None, every subject under ``meg_root``.
    """
    for subject_dir in sorted(meg_root.iterdir()):
        if subject_dir.is_dir() and (subjects is None or subject_dir.name in subjects):
            convert_subject(subject_dir, mri_root, bids_root, noise_root)


def main():
    parser = argparse.ArgumentParser(
        description="Convert unprocessed HCP MEG and MRI data to BIDS format"
    )
    parser.add_argument("meg_root", type=Path, help="HCP MEG root directory")
    parser.add_argument("mri_root", type=Path, help="HCP structural/MRI root directory")
    parser.add_argument("bids_root", type=Path, help="Output BIDS root directory")
    parser.add_argument(
        "--noise-root", type=Path, default=None,
        help="HCP empty room root containing <subject>/unprocessed/MEG/*-Rnoise "
             "and *-Pnoise (default: noise recordings are not converted)",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Subject IDs to convert (default: every subject under meg_root)",
    )
    args = parser.parse_args()

    convert_dataset(
        meg_root=args.meg_root,
        mri_root=args.mri_root,
        bids_root=args.bids_root,
        noise_root=args.noise_root,
        subjects=args.subjects,
    )


if __name__ == "__main__":
    main()
