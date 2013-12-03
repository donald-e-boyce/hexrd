#! /usr/bin/env python
#
"""Handles series of frames (images)
"""
from string import Template

import h5py
import numpy as np
#
# ---------------------------------------------------CLASS:  FrameSeries
#
class FrameSeries(object):
    """collection of images"""
    FRAMEDIR = 'frames'

    def __init__(self, h5name):
        """Constructor for FrameSeries."""
        #
        self.h5name = h5name
        self.h5file = h5py.File(h5name, "r")
        self.frames = self.h5file[self.FRAMEDIR]
        #
        return

    def __getitem__(self, key):
        # not defined for slices
        dset = self.frames[str(key)]
        return dset[...]

    def __len__(self):
        return self.frames.attrs['nframes']

    # property:  shape

    @property
    def shape(self):
        """(get-only) Shape of images"""
        return self.frames.attrs['shape']

    pass  # end class
#
# -----------------------------------------------END CLASS:  FrameSeries

