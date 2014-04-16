#! /usr/bin/env python
#
"""Make a frame series from a list image files
"""
import sys
import argparse
import logging

# Put this before fabio import and reset level if you
# want to control its import warnings.
logging.basicConfig(level=logging.DEBUG)

import h5py
import numpy as np

import fabio
from fabio import file_series

# Error messages

ERR_NO_FILE = 'Append specified, but could not open file'
ERR_NO_DATA = 'Append specified, but dataset not found in file'
ERR_OVERWRITE = 'Failed to create new dataset. Does it already exist?'
ERR_SHAPE = 'Image shape not consistent with previous images'

DSetPath = lambda f, p: "%s['%s']" % (f, p)

def write_file(a):
    #
    # Get shape and dtype information from files
    #
    shp, dtp = image_info(a)
    #
    # If append option is true, file and target group must exist;
    # otherwise, file may exist but may not already contain the
    # target dataset.
    #
    if a.append:
        try:
            f = h5py.File(a.outfile, "r+")
        except:
            errmsg = '%s: %s' % (ERR_NO_FILE, a.outfile)
            raise RuntimeError(errmsg)

        ds = f.get(a.dset)
        if ds is None:
            errmsg = '%s: %s' % (ERR_NO_DATA, DSetPath(a.outfile, a.dset))
            raise RuntimeError(errmsg)
    else:
        f = h5py.File(a.outfile, "a")
        try:
            ds = f.create_dataset(a.dset, (0, shp[0], shp[1]), dtp,
                                  maxshape=(None, shp[0], shp[1]))
        except:
            errmsg = '%s: %s' % (ERR_OVERWRITE, DSetPath(a.outfile, a.dset))
            raise RuntimeError(errmsg)
    #
    # Now add the images
    # . empty frames only apply to multiframe images
    #
    nframes = ds.shape[0]
    nfiles = len(a.imagefiles)
    for i in range(nfiles):
        logging.debug('processing file %d of %d' % (i, nfiles))
        img_i = fabio.open(a.imagefiles[i])
        nfi = img_i.nframes
        for j in range(nfi):
            logging.debug('... processing image %d of %d' % (j, img_i.nframes))
            if nfi > 1 and j < a.empty:
                logging.debug('...empty frame ... skipping')
                continue
            nframes += 1
            ds.resize(nframes, 0)
            ds[nframes - 1, :, :] = img_i.data
            if (j + 1) < nfi:
                img_i = img_i.next()
        pass

    f.close()
    return

def image_info(a):
    """Return shape and dtype of first image"""
    img_0 = fabio.open(a.imagefiles[0])
    return img_0.data.shape, img_0.data.dtype

def describe_imgs(a):
    print 'image files are: ', a.imagefiles
    im0 = fabio.open(a.imagefiles[0])
    print 'Total number of files: %d' % len(a.imagefiles)
    print 'First file: %s' % a.imagefiles[0]
    print '... fabio class: %s' % im0.__class__
    print '... number of frames: %d' % im0.nframes
    print '... image dimensions: %d X %d' % (im0.dim1, im0.dim2)
    print '... image data type: %s' % im0.data.dtype

    pass

def set_options():
    """Set options for command line"""
    parser = argparse.ArgumentParser(description="frame series builder")

    parser.add_argument("-o", "--outfile", help="name of HDF5 output file",
                        default="frame_series.h5")
    parser.add_argument("-a", "--append",
                        help="append to output file instead of making a new one",
                        action="store_true")

    help_d = "path to HDF5 data set"
    parser.add_argument("-d", "--dset", help=help_d, default="/frame_series")

    parser.add_argument("-i", "--info", help="describe the input files and quit",
                        action="store_true")
    parser.add_argument("--empty", "--blank",
                        help="number of blank frames in beginning of file",
                        metavar="N", type=int, action="store", default=0)
    # these two not yet implemented
    parser.add_argument("--dark-from-empty",
                        help="use empty frames to build dark image",
                        action="store_true")
    parser.add_argument("--dark-file", help="name of file containing dark image")

    parser.add_argument("imagefiles", nargs="+", help="image files")

    return parser

def execute(args):
    """Main execution"""
    p = set_options()
    a = p.parse_args(args)
    logging.info(str(a))

    if a.info:
        describe_imgs(a)
        sys.exit()

    write_file(a)

    return

if __name__ == '__main__':
    #
    #  run
    #
    execute(sys.argv[1:])

