# HCP MEG/MRI to BIDS: Quick Start

Converts unprocessed HCP data (StoryM task MEG, Rnoise/Pnoise empty room MEG, T1w/T2w MRI) into a BIDS dataset.
FreeSurfer reconstruction and MEG-to-MRI transforms are optional follow-up steps.

| Script | Purpose |
|---|---|
| `hcp2bids.py` | Main conversion: task MEG, empty room MEG, T1w/T2w → BIDS |
| `convert_empty_room_to_bids.py` | Convert only the empty room recordings (same code as `hcp2bids.py`) |
| `run_freesurfer.py` | Run `recon-all` on the BIDS T1w images |
| `create_transforms.py` | Fiducial-based MEG-to-MRI `trans.fif` (see caveat in step 4) |

## Prerequisites

```bash
mamba create -n ncrf-hcp -c conda-forge mne mne-bids nibabel numpy pandas scipy
mamba activate ncrf-hcp
```

FreeSurfer (steps 2–3) is expected at `/usr/local/freesurfer/7-dev` (`FREESURFER_HOME` in `run_freesurfer.py`) with a valid license.

## Expected input layout

```
/home/goku/Data/MNE-HCP/
├── MEG-StoryMath/<subject>/unprocessed/MEG/{8,9}-StoryM/{4D,EPRIME}/
├── emptyroom/<subject>/unprocessed/MEG/{1-Rnoise,2-Pnoise}/4D/
└── Structural/<subject>/unprocessed/3T/{T1w_MPR1,T2w_SPC1}/
```

## 1. Convert to BIDS

```bash
cd /home/goku/Projects/hcptobids

python hcp2bids.py \
    /home/goku/Data/MNE-HCP/MEG-StoryMath/ \
    /home/goku/Data/MNE-HCP/Structural/ \
    /home/goku/Data/MNE-HCP/hcp-bids/ \
    --noise-root /home/goku/Data/MNE-HCP/emptyroom/
```

Output:
- `sub-<id>/meg/sub-<id>_task-storymath_run-<N>_meg.fif` plus events, channels, and sidecars
- `sub-emptyroom/ses-<YYYYMMDD>/meg/sub-emptyroom_ses-<date>_task-noise_acq-{rnoise,pnoise}_meg.fif`
- `sub-<id>/anat/sub-<id>[_run-<N>]_{T1w,T2w}.nii.gz`

Without `--noise-root`, empty room recordings are skipped. To convert only those:

```bash
python convert_empty_room_to_bids.py /home/goku/Data/MNE-HCP/emptyroom/ /home/goku/Data/MNE-HCP/hcp-bids/
```

## 2. FreeSurfer recon-all (optional, ~6–8 h per subject)

```bash
python run_freesurfer.py \
    /home/goku/Data/MNE-HCP/hcp-bids/ \
    /home/goku/Data/MNE-HCP/subjects/ \
    --n-jobs 2 --env-name ncrf-hcp

# Monitor
tail -f /home/goku/Data/MNE-HCP/subjects/100307/scripts/recon-all.log
```

Then create BEM surfaces per subject:

```bash
mne watershed_bem --subject 100307 --subjects-dir /home/goku/Data/MNE-HCP/subjects --overwrite
```

## 3. MRI fiducials (optional)

`hcp2bids.py --subjects-dir` attaches nasion/LPA/RPA to the T1w sidecar if
`$SUBJECTS_DIR/<id>/bem/<id>-fiducials.fif` exists. This file is not produced by
`recon-all` or `watershed_bem`; create it with either:

- the `mne coreg` GUI (place the fiducials, then "Save" — it writes to the path above), or
- an automatic MNI-based estimate:
  ```python
  import mne
  subjects_dir = "/home/goku/Data/MNE-HCP/subjects"
  fids = mne.coreg.get_mni_fiducials("100307", subjects_dir=subjects_dir)
  mne.io.write_fiducials(f"{subjects_dir}/100307/bem/100307-fiducials.fif", fids, coord_frame="mri")
  ```

Then rerun step 1 with `--subjects-dir /home/goku/Data/MNE-HCP/subjects`.

## 4. MEG-to-MRI transforms (optional)

For each subject, save MRI fiducials as `hcp-bids/sub-<id>/sub-<id>_fiducials.json`
(`python create_transforms.py --bids-root ... --create-template` writes an example).
Coordinates are in mm, FreeSurfer surface RAS — in Freeview
(`freeview -v $SUBJECTS_DIR/<id>/mri/T1.mgz`) use the **TkReg** coordinates, not scanner RAS.

```bash
python create_transforms.py --bids-root /home/goku/Data/MNE-HCP/hcp-bids
```

Writes `derivatives/trans/sub-<id>_from-meg_to-mri_trans.fif` (readable with `mne.read_trans`) plus a JSON summary.

**Caveat:** MEG-side fiducials are read from `raw.info['dig']`, which unprocessed HCP MEG does not contain,
so this step currently fails for HCP data. Use `mne coreg` interactively instead until MEG fiducials are available.

## Verification

```bash
pip install bids-validator
bids-validator /home/goku/Data/MNE-HCP/hcp-bids/
```

```python
from mne_bids import read_raw_bids, BIDSPath

bids_path = BIDSPath(subject="100307", task="storymath", run=8, datatype="meg",
                     root="/home/goku/Data/MNE-HCP/hcp-bids")
raw = read_raw_bids(bids_path)
print(raw)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'mne_bids'` | Activate the env with mne-bids installed |
| `No Rnoise/Pnoise 4D data found` | `--noise-root` must point at the `emptyroom/` folder with `unprocessed/MEG` inside each subject, not the QC `datacheck` package |
| FreeSurfer license / not found | Check `FREESURFER_HOME` and license file; `run_freesurfer.py` sources `SetUpFreeSurfer.sh` |
| `Only 0 fiducials found` in `create_transforms.py` | Expected for HCP MEG — see the caveat in step 4 |
