"""
Run FreeSurfer recon-all on the T1w images of a BIDS dataset produced by
hcp2bids.py.

Usage:
    python run_freesurfer.py <bids_root> <subjects_dir>
        [--subjects ID ...] [--n-jobs 2] [--env-name ncrf-hcp]
"""

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple


FREESURFER_HOME = "/usr/local/freesurfer/7-dev"
DEFAULT_ENV_NAME = "ncrf-hcp"


def get_bids_subjects(bids_root: Path) -> List[str]:
    """Return the IDs (without ``sub-``) of BIDS subjects that have a T1w image."""
    return [
        subject_dir.name.replace("sub-", "")
        for subject_dir in sorted(bids_root.glob("sub-*"))
        if any((subject_dir / "anat").glob("*_T1w.nii.gz"))
    ]


def run_freesurfer_command(cmd: List[str], env_name: str = DEFAULT_ENV_NAME,
                           timeout: Optional[int] = 3600,
                           subjects_dir: Optional[Path] = None) -> subprocess.CompletedProcess:
    """
    Run a command in bash with the conda env activated and FreeSurfer sourced.

    Parameters
    ----------
    cmd : list of str
        Command to run.
    env_name : str
        Conda environment to activate first.
    timeout : int or None
        Timeout in seconds (None for no timeout).
    subjects_dir : Path, optional
        Exported as SUBJECTS_DIR if given.

    Returns
    -------
    subprocess.CompletedProcess
        Result of the command, with stdout/stderr captured as text.
    """
    script = "\n".join([
        'eval "$(conda shell.bash hook)"',
        f"conda activate {env_name}",
        f"export FREESURFER_HOME={FREESURFER_HOME}",
        "source $FREESURFER_HOME/SetUpFreeSurfer.sh",
        f"export SUBJECTS_DIR={subjects_dir}" if subjects_dir else "",
        " ".join(cmd),
    ])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout)


def check_freesurfer(env_name: str = DEFAULT_ENV_NAME):
    """Raise RuntimeError if FreeSurfer cannot be sourced in ``env_name``."""
    result = run_freesurfer_command(["which", "recon-all"], env_name, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"FreeSurfer not usable from conda env '{env_name}' with "
            f"FREESURFER_HOME={FREESURFER_HOME}:\n{result.stderr}\n"
            "Install from https://surfer.nmr.mgh.harvard.edu/fswiki/Download "
            "and get a license at https://surfer.nmr.mgh.harvard.edu/registration.html"
        )
    print("✓ FreeSurfer environment setup successful")


def prepare_subject_dir(subject_id: str, bids_root: Path, subjects_dir: Path, env_name: str):
    """
    Convert the subject's first BIDS T1w to ``SUBJECTS_DIR/<id>/mri/orig/001.mgz``.

    Does nothing if ``001.mgz`` already exists or the subject has no T1w.
    """
    orig_dir = subjects_dir / subject_id / "mri" / "orig"
    orig_dir.mkdir(parents=True, exist_ok=True)
    target = orig_dir / "001.mgz"

    t1w_files = sorted((bids_root / f"sub-{subject_id}" / "anat").glob("*_T1w.nii.gz"))
    if not t1w_files or target.exists():
        return

    print(f"  Converting {subject_id}: {t1w_files[0].name}")
    try:
        result = run_freesurfer_command(
            ["mri_convert", str(t1w_files[0]), str(target)],
            env_name, timeout=300, subjects_dir=subjects_dir,
        )
        result.check_returncode()
        print(f"    ✓ Converted to {target}")
    except Exception as e:
        print(f"    ✗ Conversion failed: {e}")


def run_recon_all_subject(subject_id: str, subjects_dir: Path,
                          env_name: str = DEFAULT_ENV_NAME) -> Tuple[str, bool, str]:
    """
    Run recon-all for one subject.

    The T1w must already be at ``SUBJECTS_DIR/<subject_id>/mri/orig/001.mgz``.

    Returns
    -------
    tuple
        (subject_id, success, message)
    """
    t1_file = subjects_dir / subject_id / "mri" / "orig" / "001.mgz"
    if not t1_file.exists():
        return subject_id, False, f"T1w file not found: {t1_file}"

    try:
        # recon-all takes several hours, so no timeout.
        result = run_freesurfer_command(
            ["recon-all", "-subject", subject_id, "-all"],
            env_name, timeout=None, subjects_dir=subjects_dir,
        )
    except Exception as e:
        return subject_id, False, f"Error: {e}"

    if result.returncode == 0:
        return subject_id, True, "recon-all completed successfully"
    error_msg = result.stderr or result.stdout
    return subject_id, False, f"recon-all failed: {error_msg[-200:]}"


def print_summary(results: dict, n_subjects: int, subjects_dir: Path):
    """Print which subjects completed and which failed."""
    print("\n" + "=" * 70)
    print("RECON-ALL SUMMARY")
    print("=" * 70)
    print(f"Completed: {len(results['completed'])}/{n_subjects}")
    if results["completed"]:
        print(f"  ✓ {', '.join(results['completed'])}")
    if results["failed"]:
        print(f"  ✗ Failed: {', '.join(results['failed'])}")
    print(f"SUBJECTS_DIR: {subjects_dir}")
    print("=" * 70 + "\n")


def run_freesurfer_recon_all(bids_root: Path, subjects_dir: Path,
                             env_name: str = DEFAULT_ENV_NAME, n_jobs: int = 2,
                             subjects: Optional[List[str]] = None) -> dict:
    """
    Run recon-all for several subjects in parallel.

    Parameters
    ----------
    bids_root : Path
        BIDS root directory.
    subjects_dir : Path
        FreeSurfer SUBJECTS_DIR to write to.
    env_name : str
        Conda environment to activate before sourcing FreeSurfer.
    n_jobs : int
        Number of recon-all runs at once.
    subjects : list of str, optional
        Subjects to process. If None, every BIDS subject with a T1w.

    Returns
    -------
    dict
        Keys: 'completed', 'failed', 'summary', and 'subjects_dir' if recon-all ran.
    """
    bids_root = Path(bids_root)
    subjects_dir = Path(subjects_dir)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("FREESURFER ANATOMICAL PROCESSING (recon-all)")
    print("=" * 70)

    if subjects is None:
        subjects = get_bids_subjects(bids_root)
    if not subjects:
        print("⚠ No subjects found in BIDS directory")
        return {"completed": [], "failed": [], "summary": "No subjects found"}

    print(f"Processing {len(subjects)} subjects with {n_jobs} parallel jobs")

    try:
        check_freesurfer(env_name)
    except RuntimeError as e:
        print(f"✗ {e}")
        return {"completed": [], "failed": subjects, "summary": str(e)}

    print(f"\nPreparing subject directories in: {subjects_dir}")
    for subject_id in subjects:
        prepare_subject_dir(subject_id, bids_root, subjects_dir, env_name)

    print(f"\nRunning recon-all for {len(subjects)} subjects...")
    results = {"completed": [], "failed": []}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(run_recon_all_subject, subject_id, subjects_dir, env_name)
            for subject_id in subjects
        ]
        for future in as_completed(futures):
            subject_id, success, message = future.result()
            results["completed" if success else "failed"].append(subject_id)
            print(f"  {'✓' if success else '✗'} {subject_id}: {message}")

    print_summary(results, len(subjects), subjects_dir)
    results["summary"] = f"Completed: {len(results['completed'])}, Failed: {len(results['failed'])}"
    results["subjects_dir"] = str(subjects_dir)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run FreeSurfer recon-all on the T1w images of a BIDS dataset"
    )
    parser.add_argument("bids_root", type=Path, help="BIDS root directory")
    parser.add_argument("subjects_dir", type=Path, help="FreeSurfer SUBJECTS_DIR to write to")
    parser.add_argument(
        "--subjects", nargs="+",
        help="Specific subjects to process (e.g., 100307 102816)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=2,
        help="Number of parallel recon-all jobs (default: 2)",
    )
    parser.add_argument(
        "--env-name", default=DEFAULT_ENV_NAME,
        help=f"Conda environment to activate before sourcing FreeSurfer (default: {DEFAULT_ENV_NAME})",
    )
    args = parser.parse_args()

    run_freesurfer_recon_all(
        bids_root=args.bids_root,
        subjects_dir=args.subjects_dir,
        env_name=args.env_name,
        n_jobs=args.n_jobs,
        subjects=args.subjects,
    )


if __name__ == "__main__":
    main()
