# HCP MEG → BIDS with MEG–MRI coregistration: how it works

This guide explains how the `hcptobids` workflow is built and why each step is the way it is. It also covers
alternative methods for subjects with different HCP files. For the commands alone, see
[README.md](../README.md); for a hands-on run through one subject, [TUTORIAL.md](TUTORIAL.md);
for installation, [INSTALL.md](INSTALL.md).

All numbers come from subject **100307**.

---

## 1. The goal

Turn unprocessed HCP StoryM MEG data into a BIDS dataset that is ready for source-space analysis with each
subject's **own** MRI. The input and output folders are listed in [README.md](../README.md#input-data).

Source analysis needs three things to agree:

1. where the head was inside the MEG helmet (**device → head**, stored per run in the MEG file);
2. where the head is in the MRI (**head → MRI**, the `trans` file);
3. a head model (BEM) and source space built from **the same MRI** the `trans` points into.

The pipeline uses HCP's **unprocessed** MEG and MRI. From HCP's processed data it uses exactly one file per
subject: the MEG head model, `<id>_MEG_anatomy_headmodel.mat`, from the "MEG anatomy" package.

## 2. The core problem

MEG–MRI coregistration normally works like this. The fiducials (nasion, left and right ear points) and a few
hundred points on the scalp are digitized during the MEG session. That head shape is then fitted to the scalp
surface from the MRI.

**HCP removed the digitized head shape (`hs_file`) from the public raw MEG**, because it contains the face.
Reading the raw data shows this:

```python
raw = mne.io.read_raw_bti(".../8-StoryM/4D/c,rfDC", ".../4D/config", head_shape_fname=None)
raw.info["dig"]          # None: no fiducials, no head shape
raw.info["dev_head_t"]   # identity: MNE had nothing to build it from
raw.info["dev_ctf_t"]    # NOT identity: the real head position is still in the file
```

The head *position* survives in the raw data, but nothing there links the head to the MRI. HCP made that link
with the head shape before removing it, and two of their processed files carry the result:

- `<id>_MEG_anatomy_transform.txt` holds matrices from the MEG head frame into HCP's **processed** MRI spaces;
- `<id>_MEG_anatomy_headmodel.mat` holds a brain-envelope surface **in the MEG head frame**.

The transform file is only usable together with HCP's processed T1w, because every matrix in it points into
that image's space or into HCP's 1 mm resampled anatomy. The head model needs no other HCP file: it is a
surface in MEG head coordinates, so fitting it onto a brain surface from any MRI of the subject gives the
head → MRI transform. That is the approach used here.

## 3. Coordinate frames

| Frame | Defined by | Units | Where it appears |
|---|---|---|---|
| 4D device | the MEG dewar | m | raw `c,rfDC` |
| 4D head ("bti", "ctf_head") | fiducials: origin midway between the ears, **x → nasion, y → left ear**, z up (ALS) | m in MNE, **mm** in HCP files | `dev_ctf_t`, `headmodel.mat`, `transform.txt` |
| Neuromag-style device ("meg") | MNE's conversion of the 4D device frame: rotate 90° about z, shift by (0, 0.02, 0.11) m | m | channel positions after `read_raw_bti(convert=True)` |
| **head** (this project) | the 4D head frame rotated 90° about z → RAS: x → right ear, y → nasion | m | `dev_head_t`, `trans.fif` |
| FreeSurfer scanner RAS | scanner coordinates of the raw T1w | mm | `orig.mgz` header, BIDS `anat/` |
| FreeSurfer surface RAS ("mri", tkr) | FreeSurfer's centred frame | mm on disk, m in MNE | surfaces, `trans.fif` |

The project's "head" frame is the 4D head frame turned into RAS by a pure rotation (`common.BTI_HEAD_TO_HEAD`).
Strictly, a Neuromag head frame puts the x axis exactly through both ear points, and that can differ from this
frame by a small angle. It doesn't matter here, because both `dev_head_t` and `trans` are defined against the
*same* frame, so the angle cancels out.

---

## 4. The workflow, step by step

### Step 1: Raw data to BIDS (`hcp2bids.py`)

- **Task MEG.** Each StoryM run is written as FIF with events taken from the trigger channel.
- **Noise.** HCP records two noise recordings per session. Rnoise is the empty room. Pnoise is recorded
  with the participant in the MEG but no task. Both go under `sub-emptyroom/ses-<date>`, told apart by `acq-`.
  Rnoise is linked to the task runs as `AssociatedEmptyRoom`.
- **MRI.** The raw `T1w_MPR1` and `T2w_SPC1` go to `anat/`.

### Step 2: Restore the head position (device → head)

MNE's BTi reader builds `dev_head_t` from the head shape file. Without that file it falls back to the identity,
which places the head at the device origin. In MNE's source (`mne/_fiff/_digitization.py`), the transform it
would have built is:

```
dev_head_t = ctf_head_t  @  dev_ctf_t  @  inv(bti_dev_t)
             (4D head →     (4D device →   (Neuromag device →
              head)          4D head)        4D device)
```

`dev_ctf_t` is in the file, and `bti_dev_t` is MNE's fixed device conversion. `ctf_head_t` normally comes
from the digitized fiducials. Here it is replaced by the fixed 4D→RAS rotation, which defines the project's
"head" frame. This is `hcp2bids.restore_head_position()`, applied to every task run during step 1.

**Validation:** HCP's FieldTrip-preprocessed StoryM data stores the sensor positions in 4D head coordinates
(`data.grad.chanpos`). Our sensor positions, mapped through the new `dev_head_t` and back into 4D head
coordinates, matched them to **0.000 mm for all 248 sensors in both runs**.

### Step 3: FreeSurfer on the raw T1w (`run_freesurfer.py`)

`recon-all` runs on the BIDS `anat/` T1w. If there are several T1w runs, recon-all averages them. The result
is written to `derivatives/freesurfer/sub-<id>` and shares its scanner space with the BIDS `anat/` images.
It took 28 minutes with 8 threads.

### Step 4: Head model (`make_head_model.py`)

In `derivatives/freesurfer/sub-<id>/bem/`:

- watershed surfaces, including `brain.surf` (used in step 5) and `sub-<id>-head.fif` (scalp);
- single-layer BEM `sub-<id>-inner_skull-bem.fif` and its solution `-bem-sol.fif` (one layer is enough for
  MEG);
- source space `sub-<id>-ico-4-src.fif`;
- a link to FreeSurfer's `fsaverage`, for morphing.

### Step 5: Fit HCP's head model (`create_transforms.py`, `--method headmodel`)

HCP's `headmodel.mat` is a FieldTrip single-shell model: 5000 vertices in 4D head coordinates, in mm. It is an
envelope of the brain, not the inner skull. It is fitted onto our watershed `brain.surf` with rigid ICP
(`fit_surface_icp`):

1. **Start:** rotate 4D head axes to RAS (head upright) and match the two surfaces' centres.
2. **Iterate:** pair each HCP vertex with its nearest `brain.surf` vertex, drop the worst 10% of pairs, then
   solve for the best rotation and translation (Kabsch/SVD). Repeat until the transform stops changing.
3. **Combine:** `trans = (4D head → MRI from ICP) @ inv(BTI_HEAD_TO_HEAD)`, written as
   `derivatives/mne/sub-<id>/meg/sub-<id>_trans.fif`, where eelbrain reads it. The JSON sidecar records the
   method and fit residuals.

The trimming step matters: the two surfaces differ in some places, especially at the base of the brain, and
those places shouldn't pull the fit. Trimming 10% instead of 0% moves the cortex by 2.1 mm (median) for 100307.

### Step 6: Check (`check_coregistration.py`)

| Check | 100307 | What it tests |
|---|---|---|
| ICP fit residual (sidecar `FitDistanceMedianMM`) | 2.5 mm median, 7.5 mm 95th percentile | how well the two surfaces match after fitting |
| Sensor-to-scalp distance | 23–44 mm, **0 sensors inside the scalp** | the whole chain: head position, trans, our surfaces |
| Pictures `derivatives/mne/sub-<id>/meg/*_coreg.png` | head centred in the helmet | gross errors |

These are sanity checks, not proof of accuracy, so the method was also validated once against HCP's own
transform for 100307. That transform is valid for a FreeSurfer reconstruction of HCP's processed T1w; it was
carried into the raw-T1w space with `mri_robust_register`, which leaves the two brain surfaces within 1.3 mm
(median). Compared with it, the head-model transform moves:

| | Median | Max |
|---|---|---|
| cortical source points | 1.4 mm | 2.6 mm |
| MEG sensors | 2.2 mm | 4.0 mm |

A functional test agreed. The evoked response to the most frequent StoryM trigger (code 88, 105 trials, peak
around 210 ms) was localized with dSPM using either transform. Both put transverse temporal cortex (primary
auditory) at or near the top of the regions in both hemispheres, with nearly the same region ranking.

**Pitfalls:**

- HCP's head model is a brain envelope. Compared with the watershed *inner skull* it looks about 5 mm off;
  compared with the watershed *brain* surface it matches. Always fit it to `brain.surf`.
- `check_coregistration.py`'s head-model comparison is the fit residual for this method, not an independent
  check, because the transform was fitted to that surface.
- HCP's MRIs are defaced, so the scalp surface has no face. This doesn't matter for a single-layer MEG BEM.
- MNE's 3D plotting needs a working Qt display, so `check_coregistration.py` renders with plain pyvista
  offscreen.

---

## 5. Other methods, for subjects with different files

`create_transforms.py --method` chooses how to get the transform:

```
Is <id>_MEG_anatomy_headmodel.mat available?
├── yes ──────────────────────────────────────────────▶ headmodel (standard)
└── no ── Are transform.txt AND HCP's processed T1w available?
          ├── yes ────────────────────────────────────▶ transform
          └── no ─────────────────────────────────────▶ fiducials (last resort)
```

Accuracy on 100307, compared with HCP's transform:

| Method | Cortex, median (max) | Sensors, median (max) |
|---|---|---|
| `headmodel` | 1.4 (2.6) mm | 2.2 (4.0) mm |
| `transform` | reference | reference |
| `fiducials`, MNI-estimated points | 15.0 (18.9) mm | not measured |

### `--method headmodel` (standard)

Section 4. It needs only `headmodel.mat`, and it works with a FreeSurfer reconstruction from any T1w of the
subject. Run `make_head_model.py` before `create_transforms.py`, so that `bem/brain.surf` exists.

Fitted to a reconstruction of HCP's processed T1w instead of the raw one, it differed from HCP's transform by
about the same amount (cortex 1.5 mm median). Most of the difference is therefore in the fit itself, not in the
registration between the raw and processed images that the comparison above needs.

### `--method transform`

`transform.txt` holds HCP's coregistration as 4×4 matrices between four spaces:

| Name | Space |
|---|---|
| `bti` | 4D head frame, mm |
| `spm` | scanner RAS of HCP's processed `T1w_acpc_dc_restore.nii.gz`, mm |
| `vox07mm` | voxels of `T1w_acpc_dc_restore`, **1-based** (MATLAB) |
| `vox` | voxels of `<id>_MEG_anatomy_anatomical.nii`, HCP's 1 mm resampled anatomy, 1-based |

`bti2spm` gives head → MRI only if FreeSurfer was run on `T1w_acpc_dc_restore`. The chain is
`head → 4D head → spm (bti2spm) → surface RAS (from orig.mgz)`. It can't be combined with the raw T1w: the
processed image is AC-PC aligned and distortion corrected, which for 100307 is a rotation of about 11° and a
shift of about 44 mm. Bridging the two would need a registration to the processed image, so that image is
needed either way.

To use a different image named in the file, the general rule is:

```
head → 4D head  (inv(BTI_HEAD_TO_HEAD))
     → 1-based voxels of image X        (bti2<X-voxels>, e.g. bti2vox07mm)
     → 0-based voxels                   (subtract 1 from each index)
     → scanner RAS of image X           (image X's NIfTI affine)
     → FreeSurfer surface RAS           (orig.mgz of a recon-all run on image X)
```

For 100307, going through `bti2vox07mm`, the −1 shift and the NIfTI affine of `T1w_acpc_dc_restore` lands
within 0.001 mm of `bti2spm`, which confirms the rule. It has not been tried with the 1 mm `vox` image.

Notes on HCP transform files, from checking 57 subjects:

- all of them point to `T1w_acpc_dc_restore` and are internally consistent (`spm2bti == vox2bti @ spm2vox`);
- some have an extra `vox2bti_registered` matrix, which was identical to `vox2bti` in every case checked.

### `--method fiducials` (last resort)

For subjects with no HCP MEG anatomy at all. The 4D head frame is defined by the nasion and ear points (origin
midway between the ears, x through the nasion, y toward the left ear). Marking those three points on the MRI
therefore defines the transform.

- It uses `derivatives/freesurfer/sub-<id>/bem/sub-<id>-fiducials.fif`. Make it in `mne coreg`: load the
  subject, place the three points on the scalp, then save. Step by step: [MANUAL_COREGISTRATION.md](MANUAL_COREGISTRATION.md).
- Without that file it falls back to `mne.coreg.get_mni_fiducials`, which was about **15 mm** off for 100307.
  That's fine for checking that the pipeline runs, not for analysis.
- Hand-placed fiducials did somewhat better on 100307: about 11 mm (median) in the best of three attempts.
  Accuracy depends on placing the points where HCP's technicians placed their coils, and there is no head shape
  to refine the result. The ear points were the main error, about 1 cm lower than HCP's; see
  [MANUAL_COREGISTRATION.md](MANUAL_COREGISTRATION.md#how-accurate-is-it). Treat results as approximate and
  say so in any publication.

---

## 6. Known limitations

- Empty-room sessions are named by date, so two subjects recorded on the same day overwrite each other.
- Pnoise (noise with the participant present) is filed under `sub-emptyroom`, although it isn't an empty-room
  recording.
- `coordsystem.json` has no landmark coordinates, because there is no digitization.
- The raw T1w has no gradient-distortion correction. This is part of the 1.4 mm difference from HCP's
  transform.
- The ICP fit is rigid and uses one surface. A poor watershed `brain.surf` gives a poor fit, so check the
  pictures and the fit residual for every subject.
