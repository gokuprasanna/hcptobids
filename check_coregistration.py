"""
Check the MEG <-> MRI alignment of a subject, with numbers and pictures.

For each task run in the BIDS dataset:

1. Sensor-to-scalp distance. MEG sensors are put in MRI space through the
   run's dev_head_t and the subject's trans file, and their distance to the
   FreeSurfer scalp surface is measured. All sensors should lie outside the
   scalp, the nearest ones roughly 1-3 cm away. This tests the whole chain.
2. HCP's head model vs ours (once per subject). HCP's single-shell head
   model, a brain envelope, is put in MRI space through the same transforms
   and compared with our watershed brain surface. Their centres should agree
   within ~1-2 mm, and the surfaces within a few mm. This tests the MRI side
   (but not for --method headmodel, which was fitted on exactly this).
3. A picture (front, side, top) of helmet, sensors, scalp and brain,
   saved to ``derivatives/trans/figures/``.

Usage:
    python check_coregistration.py <anatomy_root> <bids_root>
        [--subjects-dir <SUBJECTS_DIR>] [--subjects ID ...]
"""

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pyvista
from mne.transforms import apply_trans
from scipy.spatial import cKDTree

from common import BTI_HEAD_TO_HEAD, find_subjects, get_subjects_dir
from create_transforms import read_hcp_headmodel_bti

# MNE's own 3D plotting needs a working Qt display; plain pyvista can render
# without one.
pyvista.OFF_SCREEN = True

VIEWS = ("front", "left", "top")


def meg_sensor_positions(info: mne.Info) -> np.ndarray:
    """Positions of the MEG sensors in the device frame, in m."""
    return np.array([info["chs"][i]["loc"][:3] for i in mne.pick_types(info, meg=True)])


def _mesh(rr: np.ndarray, tris: np.ndarray) -> pyvista.PolyData:
    """A pyvista triangle mesh, with vertices converted from m to mm."""
    faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
    return pyvista.PolyData(rr * 1e3, faces)


def save_alignment_figure(info: mne.Info, trans: mne.transforms.Transform,
                          scalp: dict, brain: dict, sensors_mri: np.ndarray,
                          fname: Path, title: str):
    """
    Save front/left/top views of helmet, sensors, scalp and brain, in MRI space.

    ``scalp`` and ``brain`` are dicts with ``rr`` (m) and ``tris``.
    """
    helmet = mne.surface.get_meg_helmet_surf(info, trans=trans, verbose="error")

    plotter = pyvista.Plotter(off_screen=True, shape=(1, len(VIEWS)), window_size=(1500, 520))
    for col, view in enumerate(VIEWS):
        plotter.subplot(0, col)
        plotter.add_text(f"{title}  ({view})" if col == 0 else view, font_size=9)
        plotter.add_mesh(_mesh(helmet["rr"], helmet["tris"]), color="gray", opacity=0.15)
        plotter.add_mesh(_mesh(scalp["rr"], scalp["tris"]), color="peachpuff", opacity=0.5)
        plotter.add_mesh(_mesh(brain["rr"], brain["tris"]), color="lightsteelblue")
        plotter.add_points(sensors_mri * 1e3, color="black", point_size=6,
                           render_points_as_spheres=True)
        if view == "front":
            plotter.view_xz(negative=True)   # look from the front (+y) toward -y
        elif view == "left":
            plotter.view_yz(negative=True)   # look from the left (-x)
        else:
            plotter.view_xy()                # look down from above
    fname.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(fname)
    plotter.close()
    print(f"  Saved {fname}")


def check_subject(subject_id: str, anatomy_root: Path, bids_root: Path, subjects_dir: Path):
    """Print alignment checks and save figures for one subject."""
    fs_subject = f"sub-{subject_id}"
    trans_dir = bids_root / "derivatives" / "trans"
    trans = mne.read_trans(trans_dir / f"{fs_subject}_meg_trans.fif")
    method = json.loads((trans_dir / f"{fs_subject}_meg_trans.json").read_text()).get("Method", "?")
    print(f"\n=== Coregistration check for {fs_subject} ===")

    # Our surfaces, MRI frame, m.
    bem_dir = subjects_dir / fs_subject / "bem"
    scalp_surf = mne.read_bem_surfaces(bem_dir / f"{fs_subject}-head.fif", verbose="error")[0]
    brain_rr, brain_tris = mne.read_surface(bem_dir / "brain.surf")
    brain_surf = {"rr": brain_rr * 1e-3, "tris": brain_tris}
    scalp, brain = scalp_surf["rr"], brain_surf["rr"]
    scalp_tree, scalp_center = cKDTree(scalp), scalp.mean(axis=0)

    hcp_brain_head = apply_trans(BTI_HEAD_TO_HEAD, read_hcp_headmodel_bti(anatomy_root, subject_id))
    hcp_brain_mri = apply_trans(trans, hcp_brain_head)
    shift = (hcp_brain_mri.mean(axis=0) - brain.mean(axis=0)) * 1e3
    d = cKDTree(brain).query(hcp_brain_mri)[0] * 1e3
    note = "  (fit residual: trans was fitted to this surface)" if method == "headmodel" else ""
    print(f" Transform method: {method}")
    print(f" HCP head model vs our brain surface: centre shift {np.linalg.norm(shift):.1f} mm; "
          f"distance median {np.median(d):.1f}, 95th percentile {np.percentile(d, 95):.1f} mm{note}")

    for meg_file in sorted((bids_root / fs_subject / "meg").glob("*_meg.fif")):
        info = mne.io.read_info(meg_file, verbose="error")
        dev_mri = trans["trans"] @ info["dev_head_t"]["trans"]
        sensors_mri = apply_trans(dev_mri, meg_sensor_positions(info))
        d, nearest = scalp_tree.query(sensors_mri)
        d *= 1e3
        # A sensor is inside the head if it's closer to the scalp's centre than
        # the scalp vertex nearest to it is.
        n_inside = int(np.sum(np.linalg.norm(sensors_mri - scalp_center, axis=1)
                              < np.linalg.norm(scalp[nearest] - scalp_center, axis=1)))

        run_name = meg_file.name.replace("_meg.fif", "")
        print(f" {run_name}: sensor-to-scalp (mm) min {d.min():.1f}, median {np.median(d):.1f}, "
              f"max {d.max():.1f}; sensors inside scalp: {n_inside}")
        save_alignment_figure(info, trans, scalp_surf, brain_surf, sensors_mri,
                              trans_dir / "figures" / f"{run_name}_alignment.png", run_name)


def main():
    parser = argparse.ArgumentParser(description="Check MEG <-> MRI alignment of HCP subjects")
    parser.add_argument("anatomy_root", type=Path,
                        help="HCP MEG anatomy root containing <id>/MEG/anatomy/")
    parser.add_argument("bids_root", type=Path, help="BIDS root directory")
    parser.add_argument("--subjects-dir", type=Path, default=None,
                        help="FreeSurfer SUBJECTS_DIR (default: <bids_root>/derivatives/freesurfer)")
    parser.add_argument("--subjects", nargs="+",
                        help="Subject IDs (default: every subject with a trans file)")
    args = parser.parse_args()

    subjects_dir = get_subjects_dir(args.bids_root, args.subjects_dir)
    trans_dir = args.bids_root / "derivatives" / "trans"
    subjects = args.subjects or [s for s in find_subjects(args.bids_root)
                                 if (trans_dir / f"sub-{s}_meg_trans.fif").exists()]
    for subject_id in subjects:
        check_subject(subject_id, args.anatomy_root, args.bids_root, subjects_dir)


if __name__ == "__main__":
    main()
