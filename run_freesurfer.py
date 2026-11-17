"""
Run FreeSurfer recon-all on the raw T1w images of the BIDS dataset.

The input is ``<bids_root>/sub-<id>/anat/sub-<id>[_run-<N>]_T1w.nii.gz``,
written by hcp2bids.py from HCP's unprocessed T1w. If a subject has several
T1w runs, recon-all averages them. The reconstruction therefore shares the
scanner space of the BIDS ``anat/`` images.

Subjects are named ``sub-<id>`` in SUBJECTS_DIR, like the Appleseed dataset.
Each subject's progress and errors are in
``<SUBJECTS_DIR>/sub-<id>/scripts/recon-all-status.log`` and ``recon-all.log``.

Usage:
    python run_freesurfer.py <bids_root> [--subjects-dir <SUBJECTS_DIR>]
        [--subjects ID ...] [--n-jobs 1] [--n-threads 8]
        [--freesurfer-home <FREESURFER_HOME>]

SUBJECTS_DIR defaults to <bids_root>/derivatives/freesurfer. FreeSurfer is
found through --freesurfer-home or the FREESURFER_HOME environment variable.
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from common import find_subjects, get_subjects_dir, setup_freesurfer


def bids_t1w_paths(bids_root: Path, subject_id: str) -> List[Path]:
    """The subject's raw T1w image(s) in BIDS ``anat/``, sorted by run."""
    return sorted((Path(bids_root) / f"sub-{subject_id}" / "anat").glob("*_T1w.nii.gz"))


def run_freesurfer_command(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Run ``cmd`` in bash after sourcing ``$FREESURFER_HOME/SetUpFreeSurfer.sh``.

    SUBJECTS_DIR and FREESURFER_HOME are taken from the environment (see
    common.setup_freesurfer). Keyword arguments go to ``subprocess.run``.
    """
    script = 'source "$FREESURFER_HOME/SetUpFreeSurfer.sh" > /dev/null\n' + " ".join(cmd)
    return subprocess.run(["bash", "-c", script], **kwargs)


def recon_all(subject_id: str, bids_root: Path, subjects_dir: Path,
              n_threads: int = 1) -> bool:
    """
    Run ``recon-all -all`` for one subject and report the result.

    On the first run the BIDS T1w run(s) are imported with ``-i``. If the
    subject folder already exists, recon-all is rerun on what is there.

    Returns True on success.
    """
    fs_subject = f"sub-{subject_id}"
    cmd = ["recon-all", "-s", fs_subject, "-all"]
    if not (subjects_dir / fs_subject).exists():
        t1w_files = bids_t1w_paths(bids_root, subject_id)
        if not t1w_files:
            print(f"  ✗ {subject_id}: no T1w in {bids_root}/{fs_subject}/anat")
            return False
        cmd += [arg for t1w in t1w_files for arg in ("-i", str(t1w))]
    if n_threads > 1:
        cmd += ["-parallel", "-openmp", str(n_threads)]

    print(f"  Started {subject_id}")
    # recon-all writes its own logs, so its output isn't kept here.
    result = run_freesurfer_command(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = result.returncode == 0
    log = subjects_dir / fs_subject / "scripts" / "recon-all.log"
    print(f"  {'✓' if ok else '✗'} {subject_id}: "
          f"{'finished' if ok else f'recon-all failed, see {log}'}")
    return ok


def run_freesurfer_recon_all(bids_root: Path, subjects_dir: Optional[Path] = None,
                             subjects: Optional[List[str]] = None, n_jobs: int = 1,
                             n_threads: int = 8, freesurfer_home: Optional[str] = None) -> dict:
    """
    Run recon-all for several subjects, ``n_jobs`` at a time.

    Parameters
    ----------
    bids_root : Path
        BIDS root with ``sub-<id>/anat/*_T1w.nii.gz``.
    subjects_dir : Path, optional
        FreeSurfer SUBJECTS_DIR (default: ``bids_root/derivatives/freesurfer``).
    subjects : list of str, optional
        Subject IDs. If None, every BIDS subject with a T1w.
    n_jobs : int
        Number of subjects processed at once.
    n_threads : int
        OpenMP threads for each recon-all run.
    freesurfer_home : str, optional
        FreeSurfer install folder (default: ``$FREESURFER_HOME``).

    Returns
    -------
    dict
        ``{"completed": [ids], "failed": [ids]}``
    """
    subjects_dir = get_subjects_dir(bids_root, subjects_dir)
    subjects_dir.mkdir(parents=True, exist_ok=True)
    setup_freesurfer(freesurfer_home, subjects_dir)
    if subjects is None:
        subjects = [s for s in find_subjects(bids_root) if bids_t1w_paths(bids_root, s)]

    print(f"recon-all for {len(subjects)} subjects, {n_jobs} at a time, {n_threads} threads "
          f"each\nSUBJECTS_DIR: {subjects_dir}")
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        ok = list(executor.map(lambda s: recon_all(s, bids_root, subjects_dir, n_threads),
                               subjects))
    results = {"completed": [s for s, k in zip(subjects, ok) if k],
               "failed": [s for s, k in zip(subjects, ok) if not k]}
    print(f"Completed {len(results['completed'])}/{len(subjects)}"
          + (f"; failed: {', '.join(results['failed'])}" if results["failed"] else ""))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run FreeSurfer recon-all on the raw T1w images of a BIDS dataset"
    )
    parser.add_argument("bids_root", type=Path, help="BIDS root with sub-<id>/anat/*_T1w.nii.gz")
    parser.add_argument("--subjects-dir", type=Path, default=None,
                        help="FreeSurfer SUBJECTS_DIR (default: <bids_root>/derivatives/freesurfer)")
    parser.add_argument("--subjects", nargs="+",
                        help="Subject IDs (default: every BIDS subject with a T1w)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of subjects processed at once (default: 1)")
    parser.add_argument("--n-threads", type=int, default=8,
                        help="OpenMP threads for each recon-all run (default: 8)")
    parser.add_argument("--freesurfer-home", default=None,
                        help="FreeSurfer install folder (default: $FREESURFER_HOME)")
    args = parser.parse_args()

    try:
        run_freesurfer_recon_all(args.bids_root, args.subjects_dir, args.subjects,
                                 args.n_jobs, args.n_threads, args.freesurfer_home)
    except RuntimeError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
