import os
import unittest
import tempfile

import numpy as np

from .common import ImageSeriesTest
from .common import make_array_ims, compare, compare_meta

from hexrd import imageseries

class ImageSeriesFormatTest(ImageSeriesTest):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.tmpdir)

class TestFormatH5(ImageSeriesFormatTest):

    def setUp(self):
        self.h5file = os.path.join(self.tmpdir, 'test_ims.h5')
        self.h5path = 'array-data'
        self.fmt = 'hdf5'
        self.is_a = make_array_ims()

    def tearDown(self):
        os.remove(self.h5file)


    def test_fmth5(self):
        """save/load HDF5 format"""
        imageseries.write(self.is_a, self.h5file, self.fmt, path=self.h5path)
        is_h = imageseries.open(self.h5file, self.fmt, path=self.h5path)

        diff = compare(self.is_a, is_h)
        self.assertAlmostEqual(diff, 0., "h5 reconstruction failed")
        self.assertTrue(compare_meta(self.is_a, is_h))

    def test_fmth5_nparray(self):
        """HDF5 format with numpy array metadata"""
        key = 'np-array'
        npa = np.array([0,2.0,1.3])
        self.is_a.metadata[key] = npa
        imageseries.write(self.is_a, self.h5file, self.fmt, path=self.h5path)
        is_h = imageseries.open(self.h5file, self.fmt, path=self.h5path)
        meta = is_h.metadata

        diff = np.linalg.norm(meta[key] - npa)
        self.assertAlmostEqual(diff, 0., "h5 numpy array metadata failed")

    def test_fmth5_nocompress(self):
        """HDF5 options: no compression"""
        imageseries.write(self.is_a, self.h5file, self.fmt,
                          path=self.h5path, gzip=0)
        is_h = imageseries.open(self.h5file, self.fmt, path=self.h5path)

        diff = compare(self.is_a, is_h)
        self.assertAlmostEqual(diff, 0., "h5 reconstruction failed")
        self.assertTrue(compare_meta(self.is_a, is_h))

    def test_fmth5_compress_err(self):
        """HDF5 options: compression level out of range"""
        with self.assertRaises(ValueError):
            imageseries.write(self.is_a, self.h5file, self.fmt,
                              path=self.h5path, gzip=10)

    def test_fmth5_chunk(self):
        """HDF5 options: chunk size"""
        imageseries.write(self.is_a, self.h5file, self.fmt,
                          path=self.h5path, chunk_rows=0)
        is_h = imageseries.open(self.h5file, self.fmt, path=self.h5path)

        diff = compare(self.is_a, is_h)
        self.assertAlmostEqual(diff, 0., "h5 reconstruction failed")
        self.assertTrue(compare_meta(self.is_a, is_h))

class TestFormatFrameCache(ImageSeriesFormatTest):

    def setUp(self):
        self.fcyaml = os.path.join(self.tmpdir,  'frame-cache.yml')
        self.cache_file='frame-cache.npz'
        self.fmt = 'frame-cache'
        self.thresh = 0.5
        self.is_a = make_array_ims()
        self.npys = [] # list of npy files to remove

    def tearDown(self):
        if os.path.exists(self.fcyaml):
            os.remove(self.fcyaml)
        for npy in self.npys:
            os.remove(npy)
        os.remove(os.path.join(self.tmpdir, self.cache_file))


    def test_fmtfc_yaml(self):
        """save/load frame-cache yaml format"""
        imageseries.write(self.is_a, self.fcyaml, self.fmt,
                          threshold=self.thresh,
                          cache_file=self.cache_file,
                          output_yaml=True)
        is_fc = imageseries.open(self.fcyaml, self.fmt, style='yaml')
        diff = compare(self.is_a, is_fc)
        self.assertAlmostEqual(diff, 0., "frame-cache reconstruction failed")
        self.assertTrue(compare_meta(self.is_a, is_fc))

    def test_fmtfc_yaml_nparray(self):
        """frame-cache format with numpy array metadata"""
        key = 'np-array'
        npa = np.array([0,2.0,1.3])
        self.is_a.metadata[key] = npa

        imageseries.write(self.is_a, self.fcyaml, self.fmt,
                          threshold=self.thresh,
                          cache_file=self.cache_file,
                          output_yaml=True)
        self.npys.append(os.path.join(self.tmpdir, 'frame-cache-%s.npy' % key))

        is_fc = imageseries.open(self.fcyaml, self.fmt, style='yaml')
        meta = is_fc.metadata
        diff = np.linalg.norm(meta[key] - npa)
        self.assertAlmostEqual(diff, 0.,
                               "frame-cache numpy array metadata failed")

    def test_fmtfc_npz(self):
        """save/load frame-cache npz format"""
        cache_path = os.path.join(self.tmpdir, self.cache_file)
        imageseries.write(self.is_a, '', self.fmt,
                          threshold=self.thresh,
                          cache_file=cache_path,
                          output_yaml=False)

        is_fc = imageseries.open(cache_path, self.fmt, style='npz')
        diff = compare(self.is_a, is_fc)
        self.assertAlmostEqual(diff, 0., "frame-cache (npz) reconstruction failed")

    def test_fmtfc_npz_nparray(self):
        """save/load frame-cache npz format with numpy array as metadata"""
        key = 'np-array'
        npa = np.array([0,2.0,1.3])
        self.is_a.metadata[key] = npa

        cache_path = os.path.join(self.tmpdir, self.cache_file)
        imageseries.write(self.is_a, '', self.fmt,
                          threshold=self.thresh,
                          cache_file=cache_path,
                          output_yaml=False)

        is_fc = imageseries.open(cache_path, self.fmt, style='npz')
        meta = is_fc.metadata

        diff = np.linalg.norm(meta[key] - npa)
        self.assertAlmostEqual(diff, 0.,
                               "frame-cache numpy array metadata failed")

    def test_fmtfc_npz_meta(self):
        """save/load frame-cache npz format with generic metadata"""
        cache_path = os.path.join(self.tmpdir, self.cache_file)
        imageseries.write(self.is_a, '', self.fmt,
                          threshold=self.thresh,
                          cache_file=cache_path,
                          output_yaml=False)

        is_fc = imageseries.open(cache_path, self.fmt, style='npz')
        self.assertTrue(compare_meta(self.is_a, is_fc))
