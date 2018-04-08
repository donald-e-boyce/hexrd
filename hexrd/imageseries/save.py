"""Write imageseries to various formats"""
from __future__ import print_function
import abc
import os

import numpy as np
import h5py
import yaml


def write(ims, fname, fmt, **kwargs):
    """write imageseries to file with options

    *ims* - an imageseries
    *fname* - name of file
    *fmt* - a format string
    *kwargs* - options specific to format
    """
    wcls = _Registry.getwriter(fmt)
    w = wcls(ims, fname, **kwargs)
    w.write()


# Registry
class _RegisterWriter(abc.ABCMeta):

    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        _Registry.register(cls)


class _Registry(object):
    """Registry for imageseries writers"""
    writer_registry = dict()

    @classmethod
    def register(cls, wcls):
        """Register writer class"""
        if wcls.__name__ is not 'Writer':
            cls.writer_registry[wcls.fmt] = wcls

    @classmethod
    def getwriter(cls, name):
        """return instance associated with name"""
        return cls.writer_registry[name]
    #
    pass  # end class


class Writer(object):
    """Base class for writers"""
    __metaclass__ = _RegisterWriter
    fmt = None

    def __init__(self, ims, fname, **kwargs):
        self._ims = ims
        self._shape = ims.shape
        self._dtype = ims.dtype
        self._nframes = len(ims)
        self._meta = ims.metadata
        self._fname = fname
        self._opts = kwargs

        # split filename into components
        tmp = os.path.split(fname)
        self._fname_dir = tmp[0]
        tmp = os.path.splitext(tmp[1])
        self._fname_base = tmp[0]
        self._fname_suff = tmp[1]

    @property
    def options(self):
        """(get-only) write options passed to save function"""
        return self._opts


    pass  # end class


class WriteH5(Writer):
    fmt = 'hdf5'
    dflt_gzip = 4
    dflt_chrows = 0

    def __init__(self, ims, fname, **kwargs):
        """Write imageseries in HDF5 file

           Required Args:
           path - the path in HDF5 file

           Options:
           gzip - 0-9; 0 turns off compression; 4 is default
           chunk_rows - number of rows per chunk; default is all
           """
        Writer.__init__(self, ims, fname, **kwargs)
        self._path = self._opts['path']

    #
    # ======================================== API
    #
    def write(self):
        """Write imageseries to HDF5 file"""
        f = h5py.File(self._fname, "a")
        g = f.create_group(self._path)
        s0, s1 = self._shape

        ds = g.create_dataset('images', (self._nframes, s0, s1), self._dtype,
                              **self.h5opts)

        for i in range(self._nframes):
            ds[i, :, :] = self._ims[i]

        # add metadata
        for k, v in self._meta.items():
            g.attrs[k] = v

    @property
    def h5opts(self):
        d = {}
        # compression
        compress = self._opts.pop('gzip', self.dflt_gzip)
        if compress > 9:
            raise ValueError('gzip compression cannot exceed 9: %s' % compress)
        if compress > 0:
            d['compression'] = 'gzip'
            d['compression_opts'] = compress

        # chunk size
        s0, s1 = self._shape
        chrows = self._opts.pop('chunk_rows', self.dflt_chrows)
        if chrows < 1 or chrows > s0:
            chrows = s0
        d['chunks'] = (1, chrows, s1)

        return d

    pass  # end class


class WriteFrameCache(Writer):
    """info from yml file"""
    fmt = 'frame-cache'

    def __init__(self, ims, fname, **kwargs):
        """write yml file with frame cache info

        kwargs has keys:

        cache_file - name of array cache file
        meta - metadata dictionary
        """
        Writer.__init__(self, ims, fname, **kwargs)
        self._thresh = self.options['threshold']
        cf = kwargs['cache_file']
        if os.path.isabs(cf):
            self._cache = cf
        else:
            cdir = os.path.dirname(fname)
            self._cache = os.path.join(cdir, cf)
        self._cachename = cf

    def _process_meta(self, save_omegas=False):
        d = {}
        for k, v in self._meta.items():
            if isinstance(v, np.ndarray) and save_omegas:
                # Save as a numpy array file
                # if file does not exist (careful about directory)
                #    create new file

                cdir = os.path.dirname(self._cache)
                b = self._fname_base
                fname = os.path.join(cdir, "%s-%s.npy" % (b, k))
                if not os.path.exists(fname):
                    np.save(fname, v)

                # add trigger in yml file
                d[k] = "! load-numpy-array %s" % fname
            else:
                d[k] = v

        return d

    def _write_yml(self):
        datad = {'file': self._cachename, 'dtype': str(self._ims.dtype),
                 'nframes': len(self._ims), 'shape': list(self._ims.shape)}
        info = {'data': datad, 'meta': self._process_meta(save_omegas=True)}
        with open(self._fname, "w") as f:
            yaml.dump(info, f)

    def _write_frames(self):
        """also save shape array as originally done (before yaml)"""
        arrd = dict()
        for i in range(len(self._ims)):
            # RFE: make it so we can use emumerate on self._ims???
            frame = self._ims[i]
            mask = frame > self._thresh
            # FIXME: formalize this a little better???
            if np.sum(mask) / float(frame.shape[0]*frame.shape[1]) > 0.25:
                raise Warning("frame %d is less than 75%% sparse" % i)
            row, col = mask.nonzero()
            arrd['%d_data' % i] = frame[mask]
            arrd['%d_row' % i] = row
            arrd['%d_col' % i] = col
        arrd['shape'] = self._ims.shape
        arrd['nframes'] = len(self._ims)
        arrd['dtype'] = str(self._ims.dtype)
        arrd.update(self._process_meta())
        np.savez_compressed(self._cache, **arrd)

    def write(self):
        """writes frame cache for imageseries

        presumes sparse forms are small enough to contain all frames
        """
        self._write_frames()
        if self.options.get('output_yaml', False):
            self._write_yml()
