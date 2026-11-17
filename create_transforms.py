"""
Create MEG head -> MRI transforms (trans.fif) for HCP subjects.

Unprocessed HCP MEG has no digitized head shape, so the transform can't be
fitted from the MEG data. Three methods are available:

``--method headmodel`` (default)
    HCP's head model surface (``<id>_MEG_anatomy_headmodel.mat``, a brain
    envelope in 4D head coordinates) is fitted onto the subject's FreeSurfer
    watershed brain surface with rigid ICP. Uses only the head model file
    from HCP's MEG anatomy package; the FreeSurfer subject is made from the
    raw T1w (run_freesurfer.py). Run make_head_model.py first, for
    ``bem/brain.surf``.

``--method transform``
    HCP's own coregistration: ``bti2spm`` in ``<id>_MEG_anatomy_transform.txt``
    maps the 4D head frame (mm) to the scanner coordinates of HCP's processed
    ``T1w_acpc_dc_restore.nii.gz`` (mm). The chain is::

        head (this project's RAS head frame, see common.BTI_HEAD_TO_HEAD)
          -> 4D head                              inverse of BTI_HEAD_TO_HEAD
          -> T1w_acpc_dc_restore scanner RAS      HCP's bti2spm
          -> FreeSurfer surface RAS ("mri")       from the subject's orig.mgz

    Only valid if the FreeSurfer subject was reconstructed from
    T1w_acpc_dc_restore, which the default pipeline doesn't do. See
    docs/COREGISTRATION_GUIDE.md.

``--method fiducials``
    The 4D head frame is defined by the nasion and the two preauricular
    points (origin midway between the ears, x to the nasion, y to the left
    ear). Marking those points on the MRI therefore gives the transform. Uses
    ``<SUBJECTS_DIR>/sub-<id>/bem/sub-<id>-fiducials.fif`` (e.g. saved from
    ``mne coreg``), or MNE's MNI-based estimate if that file doesn't exist.
    Least accurate; for subjects with no HCP MEG anatomy at all.

Output, following the Appleseed dataset:
    <bids_root>/derivatives/trans/sub-<id>_meg_trans.fif
    <bids_root>/derivatives/trans/sub-<id>_meg_trans.json   (method and details)

Usage:
    python create_transforms.py <anatomy_root> <bids_root>
        [--method headmodel|transform|fiducials]
        [--subjects-dir <SUBJECTS_DIR>] [--subjects ID ...]
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import nibabel as nib
import numpy as np
import scipy.io
from mne.io.constants import FIFF
from mne.transforms import apply_trans
from scipy.spatial import cKDTree

from common import BTI_HEAD_TO_HEAD, find_subjects, get_subjects_dir


# ---------------------------------------------------------------------------
# HCP MEG anatomy files
# ---------------------------------------------------------------------------

def hcp_anatomy_dir(anatomy_root: Path, subject_id: str) -> Path:
    """Folder of a subject's HCP MEG anatomy package."""
    return anatomy_root / subject_id / "MEG" / "anatomy"


def read_hcp_transforms(fname: Path) -> Dict[str, np.ndarray]:
    """
    Read every 4x4 matrix in an HCP ``*_MEG_anatomy_transform.txt`` file.

    The file is MATLAB code with entries like ``transform.bti2spm = [ ... ];``.
    Translations are in mm.

    Returns
    -------
    dict
        ``{"bti2spm": array (4, 4), ...}``
    """
    matrices = {}
    for name, body in re.findall(r"transform\.(\w+)\s*=\s*\[(.*?)\];", fname.read_text(), re.S):
        matrices[name] = np.array(body.split(), dtype=float).reshape(4, 4)
    return matrices


def read_hcp_headmodel_bti(anatomy_root: Path, subject_id: str) -> np.ndarray:
    """Vertices of HCP's single-shell head model, in 4D head coordinates (m)."""
    fname = hcp_anatomy_dir(anatomy_root, subject_id) / f"{subject_id}_MEG_anatomy_headmodel.mat"
    headmodel = scipy.io.loadmat(fname, squeeze_me=True, struct_as_record=False)["headmodel"]
    return headmodel.bnd.pnt * 1e-3


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def scanner_to_surface_ras(subjects_dir: Path, fs_subject: str) -> np.ndarray:
    """
    Transform (mm) from scanner RAS to FreeSurfer surface RAS for a subject.

    Read from ``mri/orig.mgz``. FreeSurfer keeps the scanner coordinates of
    the input image.
    """
    header = nib.load(subjects_dir / fs_subject / "mri" / "orig.mgz").header
    return header.get_vox2ras_tkr() @ np.linalg.inv(header.get_vox2ras())


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares rotation + translation mapping ``source`` onto ``target`` (Kabsch)."""
    src_center, tgt_center = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source - src_center).T @ (target - tgt_center))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])  # no reflections
    R = Vt.T @ D @ U.T
    trans = np.eye(4)
    trans[:3, :3] = R
    trans[:3, 3] = tgt_center - R @ src_center
    return trans


def fit_surface_icp(source: np.ndarray, target: np.ndarray, init: np.ndarray,
                    n_iter: int = 100, keep: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rigidly fit ``source`` points onto ``target`` points with trimmed ICP.

    Each iteration pairs every source point with its nearest target point,
    drops the worst ``1 - keep`` of the pairs (where the two surfaces
    genuinely differ, e.g. the flattened base of HCP's head model), and
    refits.

    Returns
    -------
    trans : array (4, 4)
        source -> target transform.
    distances : array
        Final nearest-point distances (same units as the input).
    """
    tree = cKDTree(target)
    trans = init.copy()
    for _ in range(n_iter):
        dist, idx = tree.query(apply_trans(trans, source))
        pairs = dist <= np.quantile(dist, keep)
        new = _rigid_fit(source[pairs], target[idx[pairs]])
        converged = np.abs(new - trans).max() < 1e-7
        trans = new
        if converged:
            break
    return trans, tree.query(apply_trans(trans, source))[0]


def fit_headmodel(hcp_brain: np.ndarray, our_brain: np.ndarray,
                  keep: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit HCP's head model (4D head frame, m) onto our brain surface (MRI, m).

    Starts with the head upright (the 4D -> RAS rotation) and the centres
    matched, then runs ``fit_surface_icp``.

    Returns
    -------
    bti_to_mri : array (4, 4)
        4D head -> MRI transform.
    distances : array
        Final nearest-point distances, in m.
    """
    init = BTI_HEAD_TO_HEAD.copy()
    init[:3, 3] = our_brain.mean(axis=0) - init[:3, :3] @ hcp_brain.mean(axis=0)
    return fit_surface_icp(hcp_brain, our_brain, init, keep=keep)


def bti_frame_from_fiducials(nasion: np.ndarray, lpa: np.ndarray, rpa: np.ndarray) -> np.ndarray:
    """
    Transform from the coordinates the fiducials are given in to the 4D head frame.

    4D/BTi convention: origin midway between the preauricular points, x
    toward the nasion, y toward the left ear (in the fiducial plane), z up.
    """
    origin = (lpa + rpa) / 2
    ex = (nasion - origin) / np.linalg.norm(nasion - origin)
    ez = np.cross(ex, lpa - origin)
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    R = np.array([ex, ey, ez])
    trans = np.eye(4)
    trans[:3, :3] = R
    trans[:3, 3] = -R @ origin
    return trans


# ---------------------------------------------------------------------------
# The three methods. Each returns (head -> MRI transform in m, details dict).
# ---------------------------------------------------------------------------

def trans_from_hcp_transform(subject_id: str, anatomy_root: Path,
                             subjects_dir: Path) -> Tuple[np.ndarray, dict]:
    """Head -> MRI from HCP's ``bti2spm`` (FreeSurfer made from T1w_acpc_dc_restore)."""
    transform_file = (hcp_anatomy_dir(anatomy_root, subject_id)
                      / f"{subject_id}_MEG_anatomy_transform.txt")
    if not transform_file.exists():
        raise FileNotFoundError(f"HCP transform file not found: {transform_file}")

    bti2spm = read_hcp_transforms(transform_file)["bti2spm"]
    trans = scanner_to_surface_ras(subjects_dir, f"sub-{subject_id}") @ bti2spm
    trans[:3, 3] *= 1e-3  # mm -> m
    trans = trans @ np.linalg.inv(BTI_HEAD_TO_HEAD)

    details = {
        "HCPTransform": str(transform_file),
        # bti2spm isn't exactly rigid; record how far from 1 its scaling is.
        "HCPTransformScaleFactors": np.linalg.svd(bti2spm[:3, :3], compute_uv=False).round(5).tolist(),
    }
    return trans, details


def trans_from_hcp_headmodel(subject_id: str, anatomy_root: Path,
                             subjects_dir: Path) -> Tuple[np.ndarray, dict]:
    """Head -> MRI by fitting HCP's head model onto our watershed brain surface."""
    fs_subject = f"sub-{subject_id}"
    brain_surf = subjects_dir / fs_subject / "bem" / "brain.surf"
    if not brain_surf.exists():
        raise FileNotFoundError(f"No watershed brain surface: {brain_surf} "
                                "(run make_head_model.py first)")

    hcp_brain = read_hcp_headmodel_bti(anatomy_root, subject_id)
    our_brain = mne.read_surface(brain_surf)[0] * 1e-3

    bti_to_mri, dist = fit_headmodel(hcp_brain, our_brain)

    trans = bti_to_mri @ np.linalg.inv(BTI_HEAD_TO_HEAD)
    details = {
        "HCPHeadModel": str(hcp_anatomy_dir(anatomy_root, subject_id)),
        "FittedSurface": str(brain_surf),
        "FitDistanceMedianMM": round(float(np.median(dist) * 1e3), 2),
        "FitDistance95thPercentileMM": round(float(np.percentile(dist, 95) * 1e3), 2),
    }
    return trans, details


def trans_from_fiducials(subject_id: str, anatomy_root: Path,
                         subjects_dir: Path) -> Tuple[np.ndarray, dict]:
    """Head -> MRI from nasion/LPA/RPA in MRI space, via the 4D head frame definition."""
    fs_subject = f"sub-{subject_id}"
    fid_file = subjects_dir / fs_subject / "bem" / f"{fs_subject}-fiducials.fif"
    if fid_file.exists():
        dig, _ = mne.io.read_fiducials(fid_file)
        source = str(fid_file)
    else:
        dig = mne.coreg.get_mni_fiducials(fs_subject, subjects_dir=subjects_dir, verbose="error")
        source = "MNI-based estimate (mne.coreg.get_mni_fiducials), not hand-placed"

    points = {d["ident"]: d["r"] for d in dig if d["kind"] == FIFF.FIFFV_POINT_CARDINAL}
    mri_to_bti = bti_frame_from_fiducials(points[FIFF.FIFFV_POINT_NASION],
                                          points[FIFF.FIFFV_POINT_LPA],
                                          points[FIFF.FIFFV_POINT_RPA])
    trans = np.linalg.inv(mri_to_bti) @ np.linalg.inv(BTI_HEAD_TO_HEAD)
    return trans, {"Fiducials": source}


# method name -> (function, description written to the JSON sidecar)
METHODS = {
    "headmodel": (trans_from_hcp_headmodel,
                  "Rigid trimmed ICP of HCP's single-shell head model onto the "
                  "FreeSurfer watershed brain surface"),
    "transform": (trans_from_hcp_transform,
                  "HCP MEG anatomy bti2spm, combined with the 4D->RAS head rotation "
                  "and FreeSurfer's scanner->surface RAS transform"),
    "fiducials": (trans_from_fiducials,
                  "4D head frame (origin between the ears, x to nasion) built from "
                  "fiducials in MRI space"),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def create_trans(subject_id: str, anatomy_root: Path, subjects_dir: Path,
                 output_dir: Path, method: str = "headmodel") -> Tuple[bool, str]:
    """
    Write ``sub-<id>_meg_trans.fif`` (and a JSON sidecar) for one subject.

    Returns
    -------
    tuple
        (success, message)
    """
    fs_subject = f"sub-{subject_id}"
    if not (subjects_dir / fs_subject / "mri" / "orig.mgz").exists():
        return False, f"No FreeSurfer orig.mgz for {fs_subject} in {subjects_dir}"
    try:
        trans, details = METHODS[method][0](subject_id, anatomy_root, subjects_dir)
    except (FileNotFoundError, KeyError) as e:
        return False, str(e)

    output_dir.mkdir(parents=True, exist_ok=True)
    trans_path = output_dir / f"{fs_subject}_meg_trans.fif"
    mne.write_trans(trans_path, mne.transforms.Transform("head", "mri", trans), overwrite=True)

    meta = {
        "Description": "Transform from MEG head to MRI (FreeSurfer surface RAS) coordinates",
        "Subject": fs_subject,
        "Method": method,
        "MethodDescription": METHODS[method][1],
        "FreeSurferSubject": str(subjects_dir / fs_subject),
        **details,
        "TransformationMatrix": trans.tolist(),
        "TransformationUnits": "m",
    }
    trans_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return True, f"Wrote {trans_path.name} ({method})"


def create_all_transforms(anatomy_root: Path, bids_root: Path,
                          subjects_dir: Optional[Path] = None,
                          subjects: Optional[List[str]] = None,
                          method: str = "headmodel") -> dict:
    """
    Create transform files for several subjects.

    Parameters
    ----------
    anatomy_root : Path
        HCP MEG anatomy root, containing ``<id>/MEG/anatomy/``.
    bids_root : Path
        BIDS root. Transforms go in ``derivatives/trans``.
    subjects_dir : Path, optional
        FreeSurfer SUBJECTS_DIR (default: ``bids_root/derivatives/freesurfer``).
    subjects : list of str, optional
        Subject IDs. If None, every BIDS subject except ``sub-emptyroom``.
    method : {"headmodel", "transform", "fiducials"}
        How to get the transform; see the module docstring.

    Returns
    -------
    dict
        ``{"completed": [ids], "failed": [ids]}``
    """
    anatomy_root, bids_root = Path(anatomy_root), Path(bids_root)
    subjects_dir = get_subjects_dir(bids_root, subjects_dir)
    output_dir = bids_root / "derivatives" / "trans"
    subjects = subjects or find_subjects(bids_root)
    print(f"MEG -> MRI transforms (method: {method}) for {len(subjects)} subjects")

    results = {"completed": [], "failed": []}
    for subject_id in subjects:
        success, message = create_trans(subject_id, anatomy_root, subjects_dir, output_dir, method)
        results["completed" if success else "failed"].append(subject_id)
        print(f"  {'✓' if success else '✗'} {subject_id}: {message}")
    print(f"Completed {len(results['completed'])}/{len(subjects)}. Output: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Create MEG head -> MRI transforms for HCP subjects")
    parser.add_argument(
        "anatomy_root", type=Path,
        help="HCP MEG anatomy root containing <id>/MEG/anatomy/",
    )
    parser.add_argument("bids_root", type=Path, help="BIDS root directory")
    parser.add_argument(
        "--method", choices=METHODS, default="headmodel",
        help="headmodel (default): fit HCP's head model to our brain surface; transform: "
             "HCP's bti2spm, only with a FreeSurfer recon of T1w_acpc_dc_restore; "
             "fiducials: nasion/ear points on the MRI",
    )
    parser.add_argument(
        "--subjects-dir", type=Path, default=None,
        help="FreeSurfer SUBJECTS_DIR (default: <bids_root>/derivatives/freesurfer)",
    )
    parser.add_argument("--subjects", nargs="+", help="Subject IDs (e.g., 100307 102816)")
    args = parser.parse_args()

    create_all_transforms(args.anatomy_root, args.bids_root, args.subjects_dir,
                          args.subjects, args.method)


if __name__ == "__main__":
    main()
