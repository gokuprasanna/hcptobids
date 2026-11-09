"""
Create MEG-to-MRI transformation files (trans.fif) from fiducial markers.

Fiducials (LPA, RPA, Nasion) are the anatomical landmarks used to align the
MEG head and MRI coordinate systems. The rigid transform that best maps the
MEG fiducials onto the MRI fiducials is found with the Procrustes method.

Workflow:
    1. Define fiducials in MRI space (Freeview or manual entry).
    2. Save them per subject as sub-<id>_fiducials.json (or .fif).
    3. Run this script to create trans files mapping MEG head -> MRI.
    4. Check the result with mne.viz.plot_alignment().

NOTE: MEG-side fiducials are read from raw.info['dig'] of the BIDS MEG file.
Unprocessed HCP MEG has no digitization, so for HCP data this step fails
until MEG fiducials are supplied some other way.

Expected layout:
    hcp-bids/
    ├── sub-100307/
    │   ├── anat/
    │   ├── meg/
    │   └── sub-100307_fiducials.json  ← created manually
    ...

Output:
    hcp-bids/derivatives/trans/
    ├── sub-100307_from-meg_to-mri_trans.fif
    ├── sub-100307_from-meg_to-mri_trans.json
    ...
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from mne.io.constants import FIFF
from mne.transforms import Transform


FIDUCIAL_NAMES = ["LPA", "RPA", "Nasion"]
_FIFF_IDENT_TO_NAME = {
    FIFF.FIFFV_POINT_LPA: "LPA",
    FIFF.FIFFV_POINT_RPA: "RPA",
    FIFF.FIFFV_POINT_NASION: "Nasion",
}


# ---------------------------------------------------------------------------
# Reading fiducials
# ---------------------------------------------------------------------------

def create_fiducial_template(output_path: Path, description: str = "MRI-space fiducials") -> dict:
    """
    Write an example fiducial JSON file with placeholder coordinates.

    Coordinates are in mm, FreeSurfer surface RAS (tkr) space, and must be
    replaced with real measurements.
    """
    template = {
        "description": description,
        "units": "mm",
        "coordinate_system": "FreeSurfer surface RAS (tkr)",
        "fiducials": {
            "LPA": {"x": -60.0, "y": -10.0, "z": 0.0, "description": "Left Preauricular point"},
            "RPA": {"x": 60.0, "y": -10.0, "z": 0.0, "description": "Right Preauricular point"},
            "Nasion": {"x": 0.0, "y": 100.0, "z": 0.0, "description": "Nasion (bridge of nose)"},
        },
        "instructions": [
            "1. Open the subject's T1 in Freeview: freeview -v $SUBJECTS_DIR/<id>/mri/T1.mgz",
            "2. Place the cursor on each fiducial",
            "3. Record the 'TkReg coordinates' (x, y, z) in millimeters",
            "4. Update this file with measured coordinates",
            "5. Run create_transforms.py to generate trans files",
        ],
    }
    output_path.write_text(json.dumps(template, indent=2))
    print(f"✓ Created fiducial template: {output_path}")
    return template


def _cardinal_points_mm(dig) -> Dict[str, np.ndarray]:
    """Get the LPA/RPA/Nasion points (mm) from a list of digitization points."""
    fiducials = {}
    for point in dig or []:
        if point["kind"] == FIFF.FIFFV_POINT_CARDINAL and point["ident"] in _FIFF_IDENT_TO_NAME:
            fiducials[_FIFF_IDENT_TO_NAME[point["ident"]]] = point["r"] * 1000
    return fiducials


def read_fiducials_from_fif(fif_path: Path) -> Dict[str, np.ndarray]:
    """Read fiducials (mm) from an MNE ``<subject>-fiducials.fif`` file."""
    if not fif_path.exists():
        raise FileNotFoundError(f"Fiducial file not found: {fif_path}")

    dig, _ = mne.io.read_fiducials(str(fif_path))
    fiducials = _cardinal_points_mm(dig)

    missing = set(FIDUCIAL_NAMES) - set(fiducials)
    if missing:
        raise ValueError(f"Missing fiducials: {missing}")
    return fiducials


def read_fiducials_from_json(json_path: Path) -> Dict[str, np.ndarray]:
    """
    Read fiducials (mm) from a JSON file.

    Each fiducial may be ``{"x": .., "y": .., "z": ..}`` or ``[x, y, z]``, and
    they may sit under a top-level ``"fiducials"`` key or at the top level::

        {"fiducials": {"LPA": {"x": .., "y": .., "z": ..}, "RPA": .., "Nasion": ..}}
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Fiducial file not found: {json_path}")

    data = json.loads(json_path.read_text())
    fid_data = data.get("fiducials", data)

    fiducials = {}
    for name in FIDUCIAL_NAMES:
        if name not in fid_data:
            raise ValueError(f"Missing fiducial: {name}")
        coords = fid_data[name]
        if isinstance(coords, dict):
            fiducials[name] = np.array([coords["x"], coords["y"], coords["z"]])
        elif isinstance(coords, (list, tuple)):
            fiducials[name] = np.array(coords)
    return fiducials


def read_fiducials_from_file(fiducial_path: Path) -> Dict[str, np.ndarray]:
    """Read fiducial coordinates (mm) from a .json or .fif file."""
    suffix = fiducial_path.suffix.lower()
    if suffix == ".json":
        return read_fiducials_from_json(fiducial_path)
    if suffix == ".fif":
        return read_fiducials_from_fif(fiducial_path)
    raise ValueError(f"Unsupported fiducial file format: {fiducial_path.suffix}")


def get_meg_fiducials_from_raw(raw_path: Path) -> Dict[str, np.ndarray]:
    """
    Get the MEG fiducials (mm, head coordinates) from a raw .fif file.

    Raises
    ------
    ValueError
        If the file doesn't hold all three fiducials.
    """
    raw = mne.io.read_raw_fif(str(raw_path), preload=False)
    fiducials = _cardinal_points_mm(raw.info["dig"])
    if len(fiducials) < 3:
        raise ValueError(f"Only {len(fiducials)} fiducials found in {raw_path.name} "
                         "(unprocessed HCP MEG has no digitization)")
    return fiducials


# ---------------------------------------------------------------------------
# Computing and writing the transform
# ---------------------------------------------------------------------------

def create_transformation_matrix(meg_fiducials: Dict[str, np.ndarray],
                                 mri_fiducials: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Find the rigid MEG -> MRI transform that best fits the fiducials.

    Uses the Kabsch/Procrustes method: rotation and translation, no scaling.

    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix, in the same units as the input.
    """
    meg_coords = np.array([meg_fiducials[name] for name in FIDUCIAL_NAMES])
    mri_coords = np.array([mri_fiducials[name] for name in FIDUCIAL_NAMES])

    meg_center = meg_coords.mean(axis=0)
    mri_center = mri_coords.mean(axis=0)

    H = (meg_coords - meg_center).T @ (mri_coords - mri_center)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Make it a proper rotation (det = +1), not a reflection.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    trans = np.eye(4)
    trans[:3, :3] = R
    trans[:3, 3] = mri_center - R @ meg_center
    return trans


def write_trans(subject_id: str, output_dir: Path, trans_m: np.ndarray,
                meg_fiducials: Dict[str, np.ndarray], mri_fiducials: Dict[str, np.ndarray]):
    """Write the trans .fif and a JSON sidecar with the fiducials used."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"sub-{subject_id}_from-meg_to-mri_trans"

    trans_path = stem.with_suffix(".fif")
    mne.write_trans(trans_path, Transform("head", "mri", trans_m), overwrite=True)
    print(f"  ✓ Created: {trans_path.name}")

    meta = {
        "Description": "Transform from MEG head to MRI (FreeSurfer surface RAS) coordinates",
        "Subject": f"sub-{subject_id}",
        "Method": "Fiducial-based coregistration (Procrustes)",
        "Fiducials": {
            "MEG": {k: v.tolist() for k, v in meg_fiducials.items()},
            "MRI": {k: v.tolist() for k, v in mri_fiducials.items()},
        },
        "FiducialUnits": "mm",
        "TransformationMatrix": trans_m.tolist(),
        "TransformationUnits": "m",
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def create_trans_from_fiducials(subject_id: str, bids_root: Path, output_dir: Path,
                                mri_fiducial_file: Optional[Path] = None,
                                meg_fiducial_file: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Create the MEG -> MRI transform for one subject.

    Parameters
    ----------
    subject_id : str
        Subject ID (without ``sub-``).
    bids_root : Path
        BIDS root directory.
    output_dir : Path
        Where to write the trans file.
    mri_fiducial_file : Path, optional
        MRI fiducial file. If None, uses ``sub-<id>/sub-<id>_fiducials.*``.
    meg_fiducial_file : Path, optional
        MEG fiducial file. If None, reads them from the subject's BIDS MEG data.

    Returns
    -------
    tuple
        (success, message)
    """
    subject_bids_dir = bids_root / f"sub-{subject_id}"

    try:
        if mri_fiducial_file is None:
            candidates = sorted(subject_bids_dir.glob(f"sub-{subject_id}_fiducials.*"))
            if not candidates:
                return False, f"No MRI fiducial file found for sub-{subject_id}"
            mri_fiducial_file = candidates[0]
        print(f"  Reading MRI fiducials from {mri_fiducial_file.name}")
        mri_fiducials = read_fiducials_from_file(mri_fiducial_file)

        if meg_fiducial_file is None:
            meg_files = sorted((subject_bids_dir / "meg").glob("*_meg.fif"))
            if not meg_files:
                return False, f"No MEG file found for sub-{subject_id}"
            print(f"  Reading MEG fiducials from {meg_files[0].name}")
            meg_fiducials = get_meg_fiducials_from_raw(meg_files[0])
        else:
            meg_fiducials = read_fiducials_from_file(meg_fiducial_file)

        print("  Computing transformation matrix")
        trans_m = create_transformation_matrix(meg_fiducials, mri_fiducials)
        trans_m[:3, 3] *= 1e-3  # translation mm -> m

        write_trans(subject_id, output_dir, trans_m, meg_fiducials, mri_fiducials)
        return True, "Transform created successfully"

    except Exception as e:
        return False, f"Error: {e}"


def create_all_transforms(bids_root: Path, subjects: Optional[List[str]] = None,
                          output_dir: Optional[Path] = None) -> dict:
    """
    Create transform files for every subject that has fiducial data.

    Parameters
    ----------
    bids_root : Path
        BIDS root directory.
    subjects : list of str, optional
        Subjects to process. If None, every subject except ``emptyroom``.
    output_dir : Path, optional
        Where to write trans files (default: ``bids_root/derivatives/trans``).

    Returns
    -------
    dict
        Keys: 'completed', 'failed', 'summary'.
    """
    bids_root = Path(bids_root)
    output_dir = Path(output_dir) if output_dir is not None else bids_root / "derivatives" / "trans"

    print("\n" + "=" * 70)
    print("CREATING MEG-TO-MRI TRANSFORMS FROM FIDUCIALS")
    print("=" * 70)

    if subjects is None:
        subjects = sorted(d.name.replace("sub-", "") for d in bids_root.glob("sub-*")
                          if d.name != "sub-emptyroom")
    if not subjects:
        print("⚠ No subjects found")
        return {"completed": [], "failed": [], "summary": "No subjects found"}

    print(f"Processing {len(subjects)} subjects")
    print(f"Output directory: {output_dir}\n")

    results = {"completed": [], "failed": []}
    for subject_id in subjects:
        success, message = create_trans_from_fiducials(subject_id, bids_root, output_dir)
        results["completed" if success else "failed"].append(subject_id)
        print(f"  {'✓' if success else '✗'} {subject_id}: {message}")

    print("\n" + "=" * 70)
    print("TRANSFORM CREATION SUMMARY")
    print("=" * 70)
    print(f"Completed: {len(results['completed'])}/{len(subjects)}")
    if results["completed"]:
        print(f"  ✓ {', '.join(results['completed'])}")
    if results["failed"]:
        print(f"  ✗ Failed: {', '.join(results['failed'])}")
    print("=" * 70 + "\n")

    results["summary"] = f"Completed: {len(results['completed'])}, Failed: {len(results['failed'])}"
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Create MEG-to-MRI transformation files from fiducials"
    )
    parser.add_argument("--bids-root", type=Path, required=True, help="BIDS root directory")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory for trans files (default: bids_root/derivatives/trans)",
    )
    parser.add_argument(
        "--subjects", nargs="+",
        help="Specific subjects to process (e.g., 100307 102816)",
    )
    parser.add_argument(
        "--create-template", action="store_true",
        help="Write an example fiducial template to bids_root/FIDUCIALS_TEMPLATE.json and exit",
    )
    args = parser.parse_args()

    if args.create_template:
        create_fiducial_template(args.bids_root / "FIDUCIALS_TEMPLATE.json")
    else:
        create_all_transforms(args.bids_root, args.subjects, args.output_dir)


if __name__ == "__main__":
    main()
