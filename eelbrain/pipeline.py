"""
Minimal eelbrain pipeline for the converted HCP dataset.

It lets eelbrain's pipeline GUI open the dataset, for example to check the
MEG <-> MRI coregistration of each subject:

    export BIDS_ROOT=/path/to/hcp-bids
    eelbrain eelbrain/pipeline.py

Extend it (preprocessing, epochs, models) for an actual analysis.
"""
import os

from eelbrain.pipeline import *


ROOT = os.environ["BIDS_ROOT"]


class HCPStoryMath(Pipeline):

    defaults = {
        'task': 'storymath',
    }

    raw = {
        'raw': RawSource(),
    }
