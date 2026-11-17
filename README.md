# hcptobids: HCP MEG/MRI to BIDS

Converts unprocessed HCP data (StoryM task MEG, Rnoise/Pnoise noise MEG, T1w/T2w MRI) into a BIDS dataset
with a source-space-ready head model built from each subject's own MRI.

| Script | Purpose |
|---|---|
| `hcp2bids.py` | Raw task MEG (with head position), noise MEG, T1w/T2w → BIDS |
| `convert_empty_room_to_bids.py` | Convert only the noise recordings (same code as `hcp2bids.py`) |
| `run_freesurfer.py` | `recon-all` on the raw BIDS T1w |
| `make_head_model.py` | Watershed BEM, single-layer BEM + solution, ico-4 source space, fsaverage link |
| `create_transforms.py` | MEG head → MRI `trans.fif`, by fitting HCP's head model to our brain surface |
| `check_coregistration.py` | Numeric checks and pictures of the MEG ↔ MRI alignment |
| `common.py` | Shared constants (coordinate frames) and helpers |

Documentation: [docs/INSTALL.md](docs/INSTALL.md) (Python environment, FreeSurfer),
[docs/TUTORIAL.md](docs/TUTORIAL.md) (one subject end to end, with explanations),
[docs/COREGISTRATION_GUIDE.md](docs/COREGISTRATION_GUIDE.md) (how coregistration works, other methods).

## Prerequisites

A Python environment from `environment.yml` or `requirements.txt`, and FreeSurfer 7 with a license. See
[docs/INSTALL.md](docs/INSTALL.md).

## Input layout

```
$HCP_DIR/
├── MEG-StoryMath/<id>/unprocessed/MEG/{8,9}-StoryM/{4D,EPRIME}/
├── emptyroom/<id>/unprocessed/MEG/{1-Rnoise,2-Pnoise}/4D/
├── Structural/<id>/unprocessed/3T/{T1w_MPR1,T2w_SPC1}/
└── MEG-anatomy/<id>/MEG/anatomy/          ← from HCP's "MEG anatomy" package
    └── <id>_MEG_anatomy_headmodel.mat     the only file the pipeline uses
```

Why the head model: HCP withholds the digitized head shape from the raw MEG, so the MEG ↔ MRI transform can't
be fitted from the raw data. HCP's head model is a brain-envelope surface in MEG head coordinates, made with
that head shape. Fitting it onto the brain surface of our own FreeSurfer reconstruction (from the raw T1w) gives
the transform. No other processed HCP files are used. `transform.txt` isn't needed, because all its matrices
point into HCP's processed MRI spaces.

## Output layout

```
hcp-bids/
├── sub-<id>/
│   ├── anat/  sub-<id>_T1w.nii.gz, _T2w.nii.gz          raw scanner-space scans
│   └── meg/   sub-<id>_task-storymath_run-<N>_meg.fif + sidecars (head position, AssociatedEmptyRoom)
├── sub-emptyroom/ses-<YYYYMMDD>/meg/  ..._task-noise_acq-{rnoise,pnoise}_meg.fif
└── derivatives/
    ├── freesurfer/   SUBJECTS_DIR: sub-<id>/ (recon of the raw T1w in anat/), fsaverage
    │                 sub-<id>/bem/sub-<id>-{head,inner_skull-bem,inner_skull-bem-sol,ico-4-src}.fif
    └── trans/        sub-<id>_meg_trans.fif (+ .json), figures/
```

## Workflow

Set these once:

```bash
export HCP_DIR=/path/to/hcp-data              # holds MEG-StoryMath/, emptyroom/, Structural/, MEG-anatomy/
export BIDS_ROOT=$HCP_DIR/hcp-bids            # output
export FREESURFER_HOME=/path/to/freesurfer    # contains SetUpFreeSurfer.sh
cd /path/to/hcptobids
```

Each step takes `--subjects 100307 ...`. Without it, each step processes every subject that is ready for it.
The order matters: FreeSurfer reads the T1w from step 1, and the transform is fitted to a surface from step 3.

**1. Raw data to BIDS** (minutes per subject)

```bash
python hcp2bids.py $HCP_DIR/MEG-StoryMath $HCP_DIR/Structural $BIDS_ROOT --noise-root $HCP_DIR/emptyroom --subjects 100307
```

**2. FreeSurfer** (about 30 min per subject with 8 threads; `--n-jobs` subjects at once, `--n-threads` each)

```bash
python run_freesurfer.py $BIDS_ROOT --subjects 100307 --n-threads 8
tail -f $BIDS_ROOT/derivatives/freesurfer/sub-100307/scripts/recon-all-status.log   # progress
```

**3. Head model** (after recon-all has finished; also makes `bem/brain.surf`, which step 4 fits to)

```bash
python make_head_model.py $BIDS_ROOT --subjects 100307
```

**4. MEG → MRI transform** (seconds)

```bash
python create_transforms.py $HCP_DIR/MEG-anatomy $BIDS_ROOT --subjects 100307
```

The default is `--method headmodel`. For subjects without a head model file, the last resort is
`--method fiducials`. See [docs/COREGISTRATION_GUIDE.md](docs/COREGISTRATION_GUIDE.md) for the methods
and their measured accuracy.

**5. Check the alignment**

```bash
python check_coregistration.py $HCP_DIR/MEG-anatomy $BIDS_ROOT --subjects 100307
```

What to expect:
- **no sensors inside the scalp**, with the nearest sensors a few centimetres away;
- the head-model line is the ICP fit residual, normally a few millimetres. It isn't an independent check,
  because the transform was fitted to that surface;
- `derivatives/trans/figures/*_alignment.png`: head centred in the helmet, sensors all around it.

HCP's T1w is defaced, so the scalp surface has no face. That doesn't affect a single-layer MEG BEM.

## Using the result in MNE

```python
import os, mne
from mne_bids import BIDSPath, read_raw_bids

bids_root = os.environ["BIDS_ROOT"]
subjects_dir = f"{bids_root}/derivatives/freesurfer"
subject = "sub-100307"

raw = read_raw_bids(BIDSPath(subject="100307", task="storymath", run=8, datatype="meg", root=bids_root))
trans = mne.read_trans(f"{bids_root}/derivatives/trans/{subject}_meg_trans.fif")
src = mne.read_source_spaces(f"{subjects_dir}/{subject}/bem/{subject}-ico-4-src.fif")
bem = mne.read_bem_solution(f"{subjects_dir}/{subject}/bem/{subject}-inner_skull-bem-sol.fif")
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=False)
```

## Verification

Check the dataset with the BIDS Validator, either in a browser at
<https://bids-standard.github.io/bids-validator/> or, with [Deno](https://deno.com) installed:

```bash
deno run -ERWN jsr:@bids/validator $BIDS_ROOT
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the Python environment from [docs/INSTALL.md](docs/INSTALL.md) |
| `KeyError: 'HCP_DIR'` in Python | Python was started from a terminal without the variables. Export them and restart Python |
| `FreeSurfer not found` | Set `FREESURFER_HOME` or pass `--freesurfer-home` |
| `No Rnoise/Pnoise 4D data found` | `--noise-root` must point at the `emptyroom/` folder with `unprocessed/MEG` inside each subject, not the QC `datacheck` package |
| `..._MEG_anatomy_headmodel.mat` not found | Put the subject's head model file in `MEG-anatomy/<id>/MEG/anatomy/` |
| `No FreeSurfer orig.mgz` | Run step 2 first, or check `recon-all.log` in the subject's `scripts/` folder |
| `No watershed brain surface` | Run step 3 before step 4 |
| FreeSurfer license error | Put `license.txt` in `$FREESURFER_HOME`, or set `FS_LICENSE` to its path |
| Sensors inside the scalp, or a large fit residual | Look at the alignment picture and at the watershed surfaces in Freeview ([docs/INSTALL.md](docs/INSTALL.md#using-freesurfer)); the watershed may have failed |
