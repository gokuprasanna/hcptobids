# Tutorial: one HCP subject, from raw data to source-ready BIDS

This tutorial runs the whole `hcptobids` workflow on one Human Connectome Project (HCP) subject, **100307**,
and explains each step along the way. You convert the raw MEG and MRI to BIDS, reconstruct the anatomy, build
the head model, line up the MEG with the MRI, and check the alignment.

It takes about 1.5 hours, about 30 minutes of which is FreeSurfer running. Use that time for the inspection in
step 4.

By the end you will know:

- why HCP data needs special handling for MEG–MRI coregistration;
- what `dev_head_t` and `trans` are, and how to check them;
- how fitting HCP's head model gives the MEG–MRI transform, and what limits its accuracy.

The commands alone are in [README.md](../README.md). The full reasoning, coordinate frames and other
coregistration methods are in [COREGISTRATION_GUIDE.md](COREGISTRATION_GUIDE.md).

---

## 0. Background

To estimate where in the brain an MEG signal comes from, you need to know where the sensors were relative to
the brain. That takes two transforms:

```
MEG sensors ──dev_head_t──▶ head coordinates ──trans──▶ MRI coordinates ──▶ brain surfaces, head model
             (per run,                         (per subject,
              in the MEG file)                  derivatives/trans/)
```

- **`dev_head_t`** says where the head was inside the helmet during a run. The MEG system measures it at the
  start of each run.
- **`trans`** says where the head is in the MRI. Normally it comes from fitting a digitized head shape onto the
  MRI scalp.

**What is different about HCP:** HCP removed the digitized head shape from the public data, because it reveals
the face, so `trans` can't be fitted from the raw data. But HCP shares a **head model**: a surface around the
brain, already in MEG head coordinates. We reconstruct the brain from the raw MRI ourselves and fit HCP's
surface onto ours. The fit gives `trans`.

---

## 1. Setup

Install the Python environment and FreeSurfer as described in [INSTALL.md](INSTALL.md). Then get subject
100307's data from HCP into one folder, laid out as in
[README.md, Input data](../README.md#input-data).

Open a terminal, activate the Python environment, and set these variables. Replace the `/path/to/...` parts
with the real locations. Write to a new output folder, so you don't overwrite an existing dataset.

```bash
export HCP_DIR=/path/to/hcp-data                 # holds MEG-StoryMath/, emptyroom/, Structural/, MEG-anatomy/
export BIDS_ROOT=/path/to/tutorial-bids          # your BIDS dataset (created in step 2)
export FREESURFER_HOME=/path/to/freesurfer       # contains SetUpFreeSurfer.sh
cd /path/to/hcptobids                            # run everything from here
```

Every command and Python snippet below uses these variables. Start Python **from this terminal**, so it sees
them.

Look at the inputs:

```bash
ls $HCP_DIR/MEG-StoryMath/100307/unprocessed/MEG/    # 8-StoryM, 9-StoryM: the task runs
ls $HCP_DIR/emptyroom/100307/unprocessed/MEG/        # 1-Rnoise (empty room), 2-Pnoise (participant present)
ls $HCP_DIR/Structural/100307/unprocessed/3T/        # raw T1w and T2w
ls $HCP_DIR/MEG-anatomy/100307/MEG/anatomy/          # HCP's head model
```

> **Checkpoint 1.** All four folders exist, and the last one contains `100307_MEG_anatomy_headmodel.mat`.

---

## 2. Look at the raw MEG, then convert everything to BIDS

### 2a. What's in the raw file?

```python
import os, mne
run = f"{os.environ['HCP_DIR']}/MEG-StoryMath/100307/unprocessed/MEG/8-StoryM/4D/"
raw = mne.io.read_raw_bti(run + "c,rfDC", run + "config", head_shape_fname=None, preload=False)

print(raw.info["dig"])          # digitized points
print(raw.info["dev_head_t"])   # device -> head as MNE built it
print(raw.info["dev_ctf_t"])    # device -> head as the 4D system stored it
```

> **Checkpoint 2a.** `dig` is `None` and `dev_head_t` is the identity matrix. `dev_ctf_t` is **not** the
> identity: it holds the real head position.

**Question 1.** What would happen to a source analysis if `dev_head_t` stayed the identity?
*(Hint: where would MNE think the head is?)*

### 2b. Convert

```bash
python scripts/hcp2bids.py $HCP_DIR/MEG-StoryMath $HCP_DIR/Structural $BIDS_ROOT \
    --noise-root $HCP_DIR/emptyroom --subjects 100307
```

This writes the task runs, the noise recordings and the raw T1w/T2w. It also rebuilds `dev_head_t` from
`dev_ctf_t` (`restore_head_position` in `hcp2bids.py`) and links each task run to its empty-room recording.

```bash
find $BIDS_ROOT -type f | sort
grep AssociatedEmptyRoom $BIDS_ROOT/sub-100307/meg/*_meg.json
```

Check the head position that was written:

```python
import os, mne, numpy as np
bids_root = os.environ["BIDS_ROOT"]
for run in (8, 9):
    info = mne.io.read_info(f"{bids_root}/sub-100307/meg/sub-100307_task-storymath_run-{run}_meg.fif",
                            verbose="error")
    t = info["dev_head_t"]["trans"]
    angle = np.degrees(np.arccos((np.trace(t[:3, :3]) - 1) / 2))
    print(run, "translation (cm):", (t[:3, 3] * 100).round(2), " rotation (deg):", angle.round(1))
```

> **Checkpoint 2b.** Both runs show a translation of a few centimetres and a rotation of a few degrees: the
> head was not at the device origin. The two runs differ slightly, because the participant moved between them.

---

## 3. Start FreeSurfer (the slow part)

FreeSurfer reconstructs the brain surfaces from the T1w image that step 2 wrote to `anat/`. Start it in a
**second terminal** with the same variables set, and carry on in the first one:

```bash
python scripts/run_freesurfer.py $BIDS_ROOT --subjects 100307 --n-threads 8
```

Follow the progress with:

```bash
tail -f $BIDS_ROOT/derivatives/freesurfer/sub-100307/scripts/recon-all-status.log
```

> **Checkpoint 3.** The log ends with `recon-all -s sub-100307 finished without error`. With 8 threads this
> takes about half an hour.

---

## 4. Look at HCP's head model (while FreeSurfer runs)

```python
import os, numpy as np, scipy.io
fname = f"{os.environ['HCP_DIR']}/MEG-anatomy/100307/MEG/anatomy/100307_MEG_anatomy_headmodel.mat"
hm = scipy.io.loadmat(fname, squeeze_me=True, struct_as_record=False)["headmodel"]

print(hm.type, hm.unit, hm.coordsys)              # model type, units, coordinate system
print(hm.bnd.pnt.shape, hm.bnd.tri.shape)          # vertices, triangles
print("extent (mm):", np.ptp(hm.bnd.pnt, axis=0).round(1))
print("centre (mm):", hm.bnd.pnt.mean(axis=0).round(1))
```

> **Checkpoint 4.** The model is `singleshell`, in `mm`, in the `bti` (4D head) frame: one closed surface of a
> few thousand vertices. Its extent is brain-sized, and its centre is a few centimetres above the origin.

**Question 2.** In the `bti` frame, x points to the nose and y to the left ear. Which axis has the largest
extent, and does that make sense for a brain? Why is the centre *above* the origin?

**Question 3.** This surface is in MEG head coordinates. Why does that make it useful for coregistration,
even though it came from HCP's processing and not from our MRI?

---

## 5. Head model and transform

Wait for checkpoint 3 (FreeSurfer finished), then:

```bash
python scripts/make_head_model.py $BIDS_ROOT --subjects 100307
python scripts/create_transforms.py $HCP_DIR/MEG-anatomy $BIDS_ROOT --subjects 100307
ls $BIDS_ROOT/derivatives/freesurfer/sub-100307/bem $BIDS_ROOT/derivatives/trans
```

The order matters. `make_head_model.py` creates `bem/brain.surf`, and `create_transforms.py` fits HCP's head
model onto that surface.

> **Checkpoint 5.** `bem/` contains `brain.surf`, `sub-100307-inner_skull-bem-sol.fif` and
> `sub-100307-ico-4-src.fif`. `trans/` contains `sub-100307_meg_trans.fif` and a `.json` sidecar whose
> `Method` is `headmodel` and whose `FitDistanceMedianMM` is a few millimetres.

---

## 6. Check the alignment

```bash
python scripts/check_coregistration.py $HCP_DIR/MEG-anatomy $BIDS_ROOT --subjects 100307
```

The output has one line comparing HCP's head model with our brain surface, and one sensor-to-scalp line per
run. Pictures go to `$BIDS_ROOT/derivatives/trans/figures/`.

> **Checkpoint 6.** No sensors are inside the scalp, and the nearest sensors are a few centimetres from it.
> The head-model distances are a few millimetres. In the pictures, the head sits centred in the grey helmet,
> with the sensors (black dots) all around it.

How to read the numbers:

- **Sensors inside the scalp**, or a minimum distance of only a few millimetres, means the alignment is wrong.
- The head-model line is the **fit residual**: how well HCP's surface matches ours after fitting. A few
  millimetres is normal, because the two surfaces were made by different software. A much larger value means
  the fit or our `brain.surf` went wrong.
- The face is missing from the scalp because HCP defaces its MRIs. That's expected.

With a display, you can also look at the alignment interactively:

```python
import os, mne
bids_root = os.environ["BIDS_ROOT"]
info = mne.io.read_info(f"{bids_root}/sub-100307/meg/sub-100307_task-storymath_run-8_meg.fif")
mne.viz.plot_alignment(info, trans=f"{bids_root}/derivatives/trans/sub-100307_meg_trans.fif",
                       subject="sub-100307", subjects_dir=f"{bids_root}/derivatives/freesurfer",
                       surfaces=dict(head=0.4, brain=1.0), meg=("helmet", "sensors"), coord_frame="mri")
```

**Question 4.** The sensor-to-scalp check uses `dev_head_t`, `trans` and our scalp surface. The fit residual
uses only `trans` and our brain surface. Which errors can each one catch?

---

## 7. Exercise: how sensitive is the result?

Run this from the `hcptobids` folder. It refits the transform two other ways and measures how far each moves the
cortex compared with your `trans` file. Your `trans` file isn't changed.

```python
import os, sys, numpy as np, mne
from pathlib import Path
from mne.transforms import apply_trans
sys.path.insert(0, "scripts")    # the hcptobids code
import create_transforms as ct

anat = Path(os.environ["HCP_DIR"]) / "MEG-anatomy"
sd = Path(os.environ["BIDS_ROOT"]) / "derivatives/freesurfer"

trans = mne.read_trans(sd.parent / "trans/sub-100307_meg_trans.fif")["trans"]
src = mne.read_source_spaces(sd / "sub-100307/bem/sub-100307-ico-4-src.fif", verbose="error")
cortex_mri = np.vstack([s["rr"][s["vertno"]] for s in src])
cortex_head = apply_trans(np.linalg.inv(trans), cortex_mri)

def cortex_error(other, name):
    err = np.linalg.norm(apply_trans(other, cortex_head) - cortex_mri, axis=1) * 1000
    print(f"{name:24s} cortex moves: median {np.median(err):.1f} mm, max {err.max():.1f} mm")

# (a) ICP without trimming: use every point pair instead of the best 90%.
hcp = ct.read_hcp_headmodel_bti(anat, "100307")
ours = mne.read_surface(sd / "sub-100307/bem/brain.surf")[0] / 1000
bti_to_mri, _ = ct.fit_headmodel(hcp, ours, keep=1.0)
cortex_error(bti_to_mri @ np.linalg.inv(ct.BTI_HEAD_TO_HEAD), "headmodel, no trimming")

# (b) No head model at all: fiducials (MNI estimate unless you saved your own).
fid_trans, details = ct.trans_from_fiducials("100307", anat, sd)
cortex_error(fid_trans, "fiducials")
print(details)
```

> **Checkpoint 7.** Without trimming, the cortex moves by a couple of millimetres. With MNI-estimated
> fiducials it moves by more than a centimetre, and `details` says the points are an estimate.

**Question 5.** Why does including all point pairs change the fit? Look at the bottom of the brain in both
surfaces.

**Question 6.** Some HCP subjects have no head model. Which method would you use for them, and what would you
write in the methods section of a paper about its accuracy?

**Optional, with a display:** place the fiducials yourself in `mne coreg`. Load subject `sub-100307` from
`$BIDS_ROOT/derivatives/freesurfer`, mark the nasion and both ear points, and save. That writes
`bem/sub-100307-fiducials.fif`. Rerun part (b). How close do you get to the head-model result?

If something fails, see [README.md, Troubleshooting](../README.md#troubleshooting).

---

## Points of discussion

1. MNE would place the head at the device origin, centimetres from where it really was. Every source would be
   mislocated, and differently in each run.
2. x (front to back) is longest, as expected for a brain. The 4D head frame's origin is midway between the ear
   points, which sit low on the head, so most of the brain lies above it.
3. MEG head coordinates are the frame the MEG sensors are expressed in (through `dev_head_t`). Once the same
   surface is located in our MRI, we know where the MEG head frame sits in the MRI, which is `trans`. HCP made
   the surface with the head shape they later withheld, so it carries that information.
4. The sensor-to-scalp check catches errors anywhere in the chain, including `dev_head_t`, but only gross ones,
   since the sensors are centimetres from the scalp. The fit residual catches a bad fit or a bad `brain.surf`
   at the millimetre level, but says nothing about `dev_head_t`.
5. The two surfaces differ most at the base of the brain, which the two programs model differently. Including
   those pairs pulls the fit toward the mismatch. Trimming them leaves the parts where both surfaces describe
   the same anatomy.
6. `fiducials`, ideally with hand-placed points. Report it as approximate, and don't mix those subjects with
   head-model subjects without saying so.
