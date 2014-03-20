#! /usr/bin/env python
#
"""Handles series of frames (images)

Contains generic FrameSeries class, adapters for particular
data formats and a function for loading.
"""
from string import Template

import h5py
import numpy as np

# Module data

adapter_registry = dict()

# FrameSeries proxy class

class FrameSeries(object):
    """collection of images

    Basic sequence class with additional properties for image shape and
    metadata (possibly None).
    """

    def __init__(self, adapter):
        """Build FrameSeries from adapter instance

        *adapter* - object instance based on abstract Sequence class with
        properties for image shape and, optionally, metadata.
        """
        self.adapter = adapter
        print adapter
        print dir(self.adapter)

        return

    def __getitem__(self, key):
        return self.adapter[key]

    def __len__(self):
        return len(self.adapter)

    # property:  shape

    @property
    def shape(self):
        """(get-only) Shape of individual images"""
        return self.adapter.shape

    # property:  metadata

    @property
    def metadata(self):
        """(get-only) Image sequence metadata, if available"""
        if hasattr(self.adapter, 'metadata'):
            return self.adapter.metadata
        else:
            return None

    pass  # end class

# Metaclass for adapter registry

class _RegisterAdapterClass(type):

    def __init__(cls, name, bases, attrs):
        if cls.__name__ is not 'FrameSeriesAdapter':
            adapter_registry[cls.format] = cls

class FrameSeriesAdapter(object):

    __metaclass__ = _RegisterAdapterClass

# Adapter classes

class H5FrameSeries(FrameSeriesAdapter):
    """collection of images in HDF5 format"""

    format = 'HDF5'

    def __init__(self, fname, **kwargs):
        """Constructor for H5FrameSeries

        *fname* - name of the HDF5 file

        Accepts keyword arguments:

        *fspath* - path to the frame series dataset
        """
        self.h5name = fname
        self.h5file = h5py.File(fname, "r")
        self.dset = self.h5file.get(kwargs['fspath'])
        self.shape3d = self.dset.shape
        print self.shape3d

        return

    def __getitem__(self, key):
        return self.dset[key, :, :]

    def __len__(self):
        return self.shape3d[0]

    @property
    def shape(self):
        """(get-only) Shape of images"""
        return self.shape3d[1:]

    @property
    def metadata(self):
        """(get-only) Image sequence metadata"""
        # also check for dimension scales
        return self.dset.attrs

    pass  # end class
#
# -----------------------------------------------END CLASS:  H5FrameSeries

def load_series(filename, format=None, **kwargs):
    # find the appropriate adapter based on format specified
    adapter = adapter_registry[format](filename, **kwargs)
    return FrameSeries(adapter)
