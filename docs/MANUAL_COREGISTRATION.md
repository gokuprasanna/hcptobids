# Manual coregistration with fiducials

This guide shows how to make the MEG → MRI transform (`trans.fif`) by hand, without any of HCP's processed
files, not even the head model. You open the subject in eelbrain's pipeline GUI, mark three landmarks on the
MRI in its coregistration window (MNE-Python's `mne coreg`), and the pipeline builds the transform from them.

Use it for subjects that have no HCP head model, or to compare a hand-made transform with the head-model one.
It is less accurate than the head-model method: see [How accurate is it?](#how-accurate-is-it) below.

## Why only three points?

Normally, coregistration fits the head shape that was digitized during the MEG session onto the MRI scalp. HCP
removed the digitized head shape and fiducials from its public data, so there is nothing to fit, and the
**Fit Fiducials** and **Fit ICP** buttons in the coregistration window don't work for HCP subjects.

The three landmarks are enough on their own, because they *define* the MEG head coordinate frame of the 4D
system HCP used:

- the origin is midway between the left and right ear points (LPA and RPA);
- the x axis runs from the origin through the nasion;
- the y axis points toward the left ear, and z points up.

So once you mark the nasion, LPA and RPA on the MRI, the position of the MEG head frame in the MRI, which is
the transform, is fixed. The catch is that your points have to match where HCP's technicians placed theirs
during the MEG session.

## What you need

- The steps in the [README](../README.md#workflow) done up to and including step 3 for the subject:
  `hcp2bids.py`, `run_freesurfer.py` and `make_head_model.py`. Step 3 makes the scalp surface you click on.
- The eelbrain environment from [INSTALL.md](INSTALL.md#3-eelbrain-optional), activated. It also runs the
  `hcptobids` scripts.
- A display for the GUI: a desktop session, or a remote desktop. Over SSH with X forwarding it works but can be
  slow.
- `$BIDS_ROOT` and `$HCP_DIR` set as in the README, and the `hcptobids` folder as your working folder.

The commands below use subject 100307. Replace it with your subject's ID.

---

## Step 1. Keep a copy of the existing transform (if there is one)

The subject's transform is `$BIDS_ROOT/derivatives/mne/sub-<id>/meg/sub-<id>_trans.fif`, the file eelbrain
uses. Step 3 overwrites it. If the subject already has one, for example from the head-model method, copy it
first. You can then compare the two later.

```bash
cd $BIDS_ROOT/derivatives/mne/sub-100307/meg
cp sub-100307_trans.fif  sub-100307_desc-headmodel_trans.fif
cp sub-100307_trans.json sub-100307_desc-headmodel_trans.json
cd -
```

Eelbrain tracks its input files. When the transform changes, it recomputes the forward model and any source
results that depend on it the next time they're needed.

## Step 2. Mark the landmarks in eelbrain's coregistration window

1. Start eelbrain's pipeline GUI with the pipeline file in this repository:

   ```bash
   eelbrain eelbrain/pipeline.py
   ```

2. In the task menu at the top, choose **Coregistration**. Each subject has a row, with status `missing` (no
   transform yet) or `ok`.

3. Double-click the subject's row (or right-click it and choose **Open coregistration…**). The coregistration
   window opens with the subject's MRI, its first MEG run and, if there is one, its transform. It reports that
   there are no head shape points: that's expected for HCP data.

   The window starts with three coloured markers at MNI-based estimates of the landmarks, or at your saved
   landmarks if you've done this before. They are only a starting point.

4. In the **MRI Fiducials** panel on the left, untick **Lock fiducials**.

5. Place each landmark: choose **LPA**, **Nasion** or **RPA** with the buttons below the checkbox, then click
   that point on the scalp. The **X/Y/Z** boxes underneath show the selected point's coordinates (in mm), and
   you can type in them to fine-tune.

   | Landmark | Where to click |
   |---|---|
   | **Nasion** | the deepest point of the dip between the forehead and the nose, on the midline |
   | **LPA** (left preauricular point) | just in front of the left ear, at the small notch above the tragus (the flap in front of the ear canal) |
   | **RPA** (right preauricular point) | the same point on the right |

   On subject 100307, ear points placed at that notch came out about 1 cm lower than HCP's; see
   [How accurate is it?](#how-accurate-is-it).

   Tips:
   - Rotate the head (drag), zoom (scroll) and look at each point from the side and from the front.
   - Place LPA and RPA at the same height and depth. A difference between them tilts the whole transform.
   - HCP removes the face from its MRIs, so part of the face is missing from the scalp. The nasion is usually
     still visible. If it's gone for your subject, place it on the midline where the remaining forehead meets
     the cut.

6. Tick **Lock fiducials** again.

7. Click **Save MRI Fid.** This writes
   `$BIDS_ROOT/derivatives/freesurfer/sub-100307/bem/sub-100307-fiducials.fif`.

8. Close the coregistration window. If it asks about saving the *Head<>MRI transform*, choose **Discard**: the
   transform is made in the next step, not in this window. Leave eelbrain's GUI open for step 5.

Without eelbrain, the same window opens with
`mne coreg -s sub-100307 -d $BIDS_ROOT/derivatives/freesurfer`.

## Step 3. Build the transform from your landmarks

```bash
python scripts/create_transforms.py $HCP_DIR/MEG-anatomy $BIDS_ROOT --method fiducials --subjects 100307
```

The first argument isn't used by the `fiducials` method, so any folder works there.

This writes `$BIDS_ROOT/derivatives/mne/sub-100307/meg/sub-100307_trans.fif` and its `.json` sidecar. Open
the sidecar and check its `Fiducials` entry. It should name the file you saved in step 2. If it says
`MNI-based estimate`, the file wasn't found: go back to step 2 and make sure you clicked **Save MRI Fid.**

## Step 4. Check the result

```bash
python scripts/check_coregistration.py $HCP_DIR/MEG-anatomy $BIDS_ROOT --subjects 100307
```

If the subject has no HCP head model, the script says so and skips that comparison. The sensor check and the
pictures don't need it.

Look at:

- **Sensors inside the scalp:** must be 0.
- **Sensor-to-scalp distance:** the nearest sensors should be a few centimetres from the scalp, as for other
  subjects. If the nearest are much closer, the head is probably placed too high or tilted.
- **The pictures** `*_coreg.png`, next to the transform: the head should sit upright and centred in the
  grey helmet, with sensors all around it and none touching the scalp.

If something looks wrong, go back to step 2. The window loads your saved landmarks, so you only need to adjust
them. Then repeat steps 3 and 4.

Passing these checks doesn't prove the transform is accurate. They catch a head that is placed badly wrong, but
a transform that is off by a centimetre or two can still pass. Step 5 is a closer visual check, and step 6
measures the error if the subject also has a head-model transform.

## Step 5. Confirm in the coregistration window

In eelbrain's GUI, click the refresh button (**↺**) in the toolbar: the subject's status changes to `ok`.
Double-click the row again: the coregistration window opens with your new transform.

1. Under **View Options**, tick **Show MEG helmet** and **Show MEG sensors**.
2. Rotate the view and check that:
   - the head is upright and centred in the helmet, not tilted forward, backward or to one side;
   - the gap between scalp and helmet looks even over the top and back of the head;
   - your three landmarks sit where you meant to place them.
3. Close the window. If it asks about saving anything, choose **Discard**.

This step is only for looking. The landmarks alone define the transform, so if something looks wrong, improve
them in step 2 and repeat steps 3 and 4.

Warnings:

- **Don't use Fit Fiducials or Fit ICP.** They need the digitized points HCP removed, so they don't work for HCP
  subjects.
- **Don't adjust the head by hand.** Changing the values under **Translation (t) and Rotation (r)** moves the
  head by eye, with nothing tying it to the anatomy. It only gets the head roughly into place, is hard to
  reproduce, and can't be reported as a method.
- **Don't save a transform from this window** (**Save...** under **HEAD <> MRI Transform**). Its save dialog
  starts in the raw data folder (`sub-<id>/meg/`), where derivatives don't belong, and saving over
  `derivatives/mne/sub-<id>/meg/sub-<id>_trans.fif` would leave its `.json` sidecar describing a different
  method.
- **Leave the scaling options alone.** They turn the subject's MRI into a scaled template brain.

## Step 6 (optional). Compare with the head-model transform

If you kept a head-model transform in step 1, this measures how far apart the two put the cortex. Run it in
Python, started from the same terminal:

```python
import os, numpy as np, mne
from mne.transforms import apply_trans

bids_root = os.environ["BIDS_ROOT"]
sd = f"{bids_root}/derivatives/freesurfer"
meg = f"{bids_root}/derivatives/mne/sub-100307/meg"
manual = mne.read_trans(f"{meg}/sub-100307_trans.fif")["trans"]
reference = mne.read_trans(f"{meg}/sub-100307_desc-headmodel_trans.fif")["trans"]

src = mne.read_source_spaces(f"{sd}/sub-100307/bem/sub-100307-ico-4-src.fif", verbose="error")
cortex = np.vstack([s["rr"][s["vertno"]] for s in src])          # MRI coordinates, m
moved = apply_trans(manual @ np.linalg.inv(reference), cortex)  # where your trans puts the same points
err = np.linalg.norm(moved - cortex, axis=1) * 1000
print(f"cortex difference: median {np.median(err):.1f} mm, max {err.max():.1f} mm")
```

This only measures the difference. The head-model transform isn't used by your manual transform.

To go back to the head-model transform, copy the `desc-headmodel` files back over `sub-100307_trans.fif` and
`sub-100307_trans.json`.

---

## How accurate is it?

- The transform is only as good as the three points. A few millimetres of difference in where you put the ear
  points moves the whole head frame, including the cortex far from the ears.
- The points have to match the ones HCP's technicians used on the day of the MEG session. Ear points are the
  hardest: the tragus, the ear canal and the preauricular notch can be several millimetres apart. Check HCP's
  MEG documentation for how they placed their fiducials, and use the same convention for every subject.
- Measured on subject 100307, against the head-model transform (cortex, median):
  - MNI-based estimates (the starting markers, never moved): 16 mm;
  - three hand-placed attempts: 20, 15 and 11 mm.

  In every attempt the nasion was within a few millimetres of HCP's, and the main error was in the ear points.
  Placed at the notch above the tragus, they were about 1 cm lower than HCP's. Once they were level with each
  other and right front to back, that height difference alone left the cortex about 1 cm off. This is one
  subject; check the convention against HCP's documentation before relying on it.
- Report the method in any publication, and don't mix fiducial-based and head-model subjects without saying so.

For how the transform is computed from the three points, and the other methods, see
[COREGISTRATION_GUIDE.md](COREGISTRATION_GUIDE.md#--method-fiducials-last-resort).
