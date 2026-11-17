# Installation

The workflow needs two things: a Python environment with MNE-Python and MNE-BIDS, and FreeSurfer. It runs on
Linux and macOS. On Windows, use WSL2 (Ubuntu), because FreeSurfer doesn't run natively on Windows.

Plan for about 16 GB of RAM and, per subject, about 3 GB of disk for the BIDS data and 0.5 GB for the FreeSurfer
reconstruction.

## 1. Python environment

With conda or mamba ([Miniforge](https://github.com/conda-forge/miniforge) is recommended):

```bash
cd /path/to/hcptobids
mamba env create -f environment.yml
mamba activate hcptobids
```

Or with pip, in a virtual environment (Python 3.10 or newer):

```bash
python -m venv ~/venvs/hcptobids
source ~/venvs/hcptobids/bin/activate
pip install -r requirements.txt
```

Check it:

```bash
python -c "import mne, mne_bids, pyvista; print(mne.__version__, mne_bids.__version__)"
```

## 2. FreeSurfer

FreeSurfer reconstructs the brain surfaces from the T1w scan (`run_freesurfer.py`) and makes the watershed
head surfaces (`make_head_model.py`). The workflow was developed with FreeSurfer 7.

### Download

Get the installer for your system from the FreeSurfer download page:
<https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>.

- **Ubuntu/Debian:** the `.deb` package.
- **Other Linux:** the `.rpm` package or the `.tar.gz` archive.
- **macOS:** the `.pkg` installer (Intel or Apple silicon).

### Install

```bash
# Ubuntu/Debian
sudo apt-get install ./freesurfer_*.deb

# .tar.gz archive: unpack into a folder of your choice
sudo tar -xzf freesurfer-linux-*.tar.gz -C /usr/local
```

On Linux, `recon-all` also needs the `tcsh` shell: `sudo apt-get install tcsh` (Ubuntu/Debian).

The installation folder (for example `/usr/local/freesurfer` or `/usr/local/freesurfer/7.4.1`) is
`FREESURFER_HOME`. It contains `SetUpFreeSurfer.sh`.

### License

FreeSurfer is free but needs a license file.

1. Register at <https://surfer.nmr.mgh.harvard.edu/registration.html>. The license arrives by email.
2. Save it as `$FREESURFER_HOME/license.txt`, or save it anywhere and point the `FS_LICENSE` variable at it:

   ```bash
   export FS_LICENSE=$HOME/freesurfer-license.txt
   ```

### Set up your shell

Add these lines to `~/.bashrc` (or `~/.zshrc` on macOS), then open a new terminal:

```bash
export FREESURFER_HOME=/usr/local/freesurfer      # your installation folder
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

The scripts only need `FREESURFER_HOME`, and source `SetUpFreeSurfer.sh` themselves. Instead of setting the
variable, you can pass `--freesurfer-home /path/to/freesurfer` to `run_freesurfer.py` and
`make_head_model.py`.

### Check it

```bash
recon-all -version
mri_convert --version
ls $FREESURFER_HOME/subjects/fsaverage     # the template brain, linked into the dataset later
```

### Using FreeSurfer

You don't need to run FreeSurfer commands by hand. `run_freesurfer.py` runs `recon-all`, and
`make_head_model.py` runs the watershed algorithm through MNE. Two things are useful to know:

- **Progress and errors** of a reconstruction are in `<SUBJECTS_DIR>/sub-<id>/scripts/recon-all-status.log`
  (one line per stage) and `recon-all.log` (everything). `SUBJECTS_DIR` is
  `$BIDS_ROOT/derivatives/freesurfer` by default.
- **Looking at the results** in Freeview, FreeSurfer's viewer, is the best way to spot a failed
  reconstruction:

  ```bash
  cd $BIDS_ROOT/derivatives/freesurfer/sub-<id>
  freeview -v mri/T1.mgz mri/brainmask.mgz:colormap=heat:opacity=0.3 \
      -f surf/lh.white:edgecolor=blue surf/lh.pial:edgecolor=red \
         surf/rh.white:edgecolor=blue surf/rh.pial:edgecolor=red \
         bem/watershed/sub-<id>_brain_surface:edgecolor=yellow
  ```

  The white (blue) and pial (red) surfaces should follow the grey/white boundary and the outer edge of the
  cortex. The watershed brain surface (yellow, after `make_head_model.py`) should wrap the brain closely.

FreeSurfer's own documentation and tutorials are at <https://surfer.nmr.mgh.harvard.edu/fswiki>.
