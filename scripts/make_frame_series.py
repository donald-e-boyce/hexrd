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

ERR_SHAPE = 'Image shape not consistent with previous images'

def write_file(a):
    f =h5py.File(a.outfile, "w")
    frame_grp = f.create_group("frames")

    imgnum = 0
    shape = None
    nfiles = len(a.imagefiles)
    for i in range(nfiles):
        logging.debug('file %d of %d' % (i, nfiles))
        img_i = fabio.open(a.imagefiles[i])
        if shape is None:
            shape = img_i.data.shape
            logging.debug('shape is: %d X %d' )
        else:
            if not shape == img_i.data.shape:
                raise ValueError(ERR_SHAPE)
        for j in range(img_i.nframes):
            logging.debug('processing image %d of %d' % (j, img_i.nframes))
            logging.debug('image_number: %d' % imgnum)
            if j < a.empty:
                logging.debug('...empty frame ... skipping')
                continue

            dset = frame_grp.create_dataset(str(imgnum), img_i.data.shape,
                                            img_i.data.dtype, img_i.data)
            if (j + 1) < img_i.nframes:
                img_i = img_i.next()
            imgnum += 1
        pass

    nframes = imgnum
    frame_grp.attrs['nframes'] = nframes
    frame_grp.attrs['shape'] = shape
    logging.debug('number of frames: %d' % nframes)

    return

def describe_imgs(a):
    print 'image files are: ', a.imagefiles
    im0 = fabio.open(a.imagefiles[0])
    print 'First file is of fabio class: %s' % im0.__class__
    print 'Dimensions are: %d X %d' % (im0.dim1, im0.dim2)
    print 'number of frames: %d' % im0.nframes

    pass

def set_options():
    """Set options for command line"""
    parser = argparse.ArgumentParser(description="frame series builder")
    parser.add_argument("-a", "--append",
                        help="append to output file instead of making a new one",
                        action="store_true")
    parser.add_argument("-o", "--outfile", help="name of HDF5 output file",
                        default="frame_series.h5")
    parser.add_argument("-i", "--info", help="describe the input files and quit",
                        action="store_true")
    parser.add_argument("--empty", "--blank",
                        help="number of blank frames in beginning of file",
                        metavar="N", type=int, action="store", default=0)
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

