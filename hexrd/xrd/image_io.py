"""Image reading (mostly) and writing

Classes
-------
Framer2DRC: base class for reader/writers
ReadGeneric:
ReadGE:

ThreadReadFrame: class for using threads to read frames

Functions
---------
getNFramesFromBytes - reader function from original detector module
newGenericReader - returns a reader instance

"""
import copy
import os
import time
import warnings

import numpy as num

from frame_series import load_series

warnings.filterwarnings('always', '', DeprecationWarning)

class OmegaFrameReader(object):
    """Facade for frame_series class, replacing other readers, primarily ReadGE"""

    def __init__(self, fileinfo):
        """Initialize frame reader

        *fileinfo* is a dictionary including keys for "format" and "filename";
                      other keys depend on the format and will be passed on
        """
        if isinstance(fileinfo, dict):
            if "format" not in fileinfo or "filename" not in fileinfo:
                raise RuntimeError('Could not find "name" or "path" in fileinfo.')
        else:
            raise RuntimeError('Old style fileinfo no longer in use.')

        fmt = fileinfo.pop('format')
        fname = fileinfo.pop('filename')

        self.frame_series = load_series(fname, fmt, fileinfo)

        return

    # property:  nframes

    @property
    def nframes(self):
        """(get-only) Number of available data frames"""
        return len(self.frame_series)

    pass

class Framer2DRC(object):
    """Base class for readers.

    You can make an instance of this class and use it for most of the
    things a reader would do, other than actually reading frames
    """
    def __init__(self,
                 ncols, nrows,
                 dtypeDefault='int16', dtypeRead='uint16', dtypeFloat='float64'):
        self.__ncols = ncols
        self.__nrows = nrows
        self.__frame_dtype_dflt  = dtypeDefault
        self.__frame_dtype_read  = dtypeRead
        self.__frame_dtype_float = dtypeFloat

        self.__nbytes_frame  = num.nbytes[dtypeRead]*nrows*ncols

        return

    def get_ncols(self):
        return self.__ncols
    ncols = property(get_ncols, None, None)

    def get_nbytesFrame(self):
        return self.__nbytes_frame
    nbytesFrame = property(get_nbytesFrame, None, None)

    def get_nrows(self):
        return self.__nrows
    nrows = property(get_nrows, None, None)

    def get_dtypeDefault(self):
        return self.__frame_dtype_dflt
    dtypeDefault = property(get_dtypeDefault, None, None)
    def get_dtypeRead(self):
        return self.__frame_dtype_read
    dtypeRead = property(get_dtypeRead, None, None)
    def get_dtypeFloat(self):
        return self.__frame_dtype_float
    dtypeFloat = property(get_dtypeFloat, None, None)

    def getOmegaMinMax(self):
        raise NotImplementedError
    def getDeltaOmega(self):
        'needed in findSpotsOmegaStack'
        raise NotImplementedError
    def getNFrames(self):
        """
        number of total frames with real data, not number remaining
        needed in findSpotsOmegaStack
        """
        raise NotImplementedError
    def read(self, nskip=0, nframes=1, sumImg=False):
        'needed in findSpotsOmegaStack'
        raise NotImplementedError
    def getDark(self):
        'needed in findSpotsOmegaStack'
        raise NotImplementedError
    def getFrameOmega(self, iFrame=None):
        'needed in findSpotsOmegaStack'
        raise NotImplementedError


    @classmethod
    def maxVal(cls, dtypeRead):
        """
        maximum value that can be stored in the image pixel data type;
        redefine as desired
        """
        maxInt = num.iinfo(dtypeRead).max
        return maxInt

    def getEmptyMask(self):
        """convenience method for getting an emtpy mask"""
        # this used to be a class method
        mask = num.zeros([self.nrows, self.ncols], dtype=bool)
        return mask

    def getSize(self):
        retval = (self.nrows, self.ncols)
        return retval

    def frame(self, nframes=None, dtype=None, buffer=None, mask=None):
        if buffer is not None and dtype is None:
            if hasattr(buffer,'dtype'):
                dtype = buffer.dtype
        if dtype is None:
            dtype = self.__frame_dtype_dflt
        if nframes is None:
            shape = (self.nrows, self.ncols)
        else:
            assert mask is None,\
                'not coded: multiframe with mask'
            shape = (nframes, self.rows, self.ncols)
        if buffer is None:
            retval = num.zeros(shape, dtype=dtype)
        else:
            retval = num.array(buffer, dtype=dtype).reshape(shape)
        if mask is not None:
            retval = num.ma.masked_array(retval, mask, hard_mask=True, copy=False)
        return retval



class ReadGeneric(Framer2DRC):
    '''
    may eventually want ReadGE to inherit from this, or pull common things
    off to a base class
    '''
    def __init__(self, filename, ncols, nrows, *args, **kwargs):
        self.filename        = filename
        self.__nbytes_header = kwargs.pop('nbytes_header', 0)
        self.__nempty        = kwargs.pop('nempty', 0)
        doFlip               = kwargs.pop('doFlip', False)
        self.subtractDark    = kwargs.pop('subtractDark', False)

        if doFlip is not False:
            raise NotImplementedError, 'doFlip not False'
        if self.subtractDark is not False:
            raise NotImplementedError, 'subtractDark not False'

        Framer2DRC.__init__(self, ncols, nrows, **kwargs)

        self.dark = None
        self.dead = None
        self.mask = None

        self.omegaStart = None
        self.omegaDelta = None
        self.omegas = None
        #
        if len(args) == 0:
            pass
        elif len(args) == 2:
            self.omegaStart = omegaStart = args[0]
            self.omegaDelta = omegaDelta = args[1]
        else:
            raise RuntimeError, 'do not know what to do with args: '+str(args)
        self.omegas = None
        if self.omegaStart is not None:
            if hasattr(omegaStart, 'getVal'):
                omegaStart = omegaStart.getVal('radians')
            if hasattr(omegaDelta, 'getVal'):
                omegaDelta = omegaDelta.getVal('radians')
            nFramesTot = self.getNFrames()
            self.omegas = \
                num.arange(omegaStart, omegaStart+omegaDelta*(nFramesTot-0.5), omegaDelta) + \
                0.5 * omegaDelta # put omegas at mid-points of omega range for frame
            omegaEnd = omegaStart+omegaDelta*(nFramesTot)
            self.omegaMin = min(omegaStart, omegaEnd)
            self.omegaMax = max(omegaStart, omegaEnd)
            self.omegaDelta = omegaDelta
            self.omegaStart = omegaStart

        if len(kwargs) > 0:
            raise RuntimeError, 'unparsed kwargs : %s' + str(kwargs.keys())

        self.iFrame = -1 # counter for last global frame that was read

        self.img = None
        if self.filename is not None:
            self.img = open(self.filename, mode='rb')
            # skip header for now
            self.img.seek(self.__nbytes_header, 0)
            if self.__nempty > 0:
                self.img.seek(self.nbytesFrame*self.__nempty, 1)

        return

    def getFrameUseMask(self):
        return False
    def __flip(self, thisframe):
        return thisframe

    '''
    def read(self, nskip=0, nframes=1, sumImg=False):

        if not nframes == 1:
            raise NotImplementedError, 'nframes != 1'
        if not sumImg == False:
            raise NotImplementedError, 'sumImg != False'

        data = self.__readNext(nskip=nskip)

        self.iFrame += nskip + 1

        return data
    '''
    def read(self, nskip=0, nframes=1, sumImg=False):
        """
        sumImg can be set to True or to something like numpy.maximum
        """

        if self.img is None:
            raise RuntimeError, 'no image file open'

        'get iFrame ready for how it is used here'
        self.iFrame = num.atleast_1d(self.iFrame)[-1]
        iFrameList = []
        multiframe = nframes > 1

        nFramesInv = 1.0 / nframes
        doDarkSub = self.subtractDark # and self.dark is not None

        if doDarkSub:
            assert self.dark is not None, 'self.dark is None'

        # assign storage array
        if sumImg:
            sumImgCallable = hasattr(sumImg,'__call__')
            imgOut = self.frame(dtype=self.dtypeFloat, mask=self.dead)
        elif multiframe:
            imgOut = self.frame(nframes=nframes, dtype=self.dtypeDflt, mask=self.dead)


        # now read data frames
        for i in range(nframes):

            #data = self.__readNext(nskip=nskip)
            #thisframe = data.reshape(self.__nrows, self.__ncols)
            data = self.__readNext(nskip=nskip) # .reshape(self.__nrows, self.__ncols)
            self.iFrame += nskip + 1
            nskip=0 # all done skipping once have the first frame!
            iFrameList.append(self.iFrame)
            # dark subtraction
            if doDarkSub:
                'used to have self.dtypeFloat here, but self.dtypeDflt does the trick'
                thisframe = self.frame(buffer=data,
                                       dtype=self.dtypeDflt, mask=self.dead) - self.dark
            else:
                thisframe = self.frame(buffer=data,
                                       mask=self.dead)

            # flipping
            thisframe = self.__flip(thisframe)

            # masking (True get zeroed)
            if self.mask is not None:
                if self.getFrameUseMask():
                    thisframe[self.mask] = 0

            # assign output
            if sumImg:
                if sumImgCallable:
                    imgOut = sumImg(imgOut, thisframe)
                else:
                    imgOut = imgOut + thisframe * nFramesInv
            elif multiframe:
                imgOut[i, :, :] = thisframe[:, :]
        'end of loop over nframes'

        if sumImg:
            # imgOut = imgOut / nframes # now taken care of above
            pass
        elif not multiframe:
            imgOut = thisframe

        if multiframe:
            'make iFrame a list so that omega or whatever can be averaged appropriately'
            self.iFrame = iFrameList
        return imgOut

    def getNFrames(self, lessEmpty=True):
        fileBytes = os.stat(self.filename).st_size
        nFrames = getNFramesFromBytes(fileBytes, self.__nbytes_header, self.nbytesFrame)
        if lessEmpty:
            nFrames -= self.__nempty
        return nFrames

    def getOmegaMinMax(self):
        assert self.omegas is not None,\
            """instance does not have omega information"""
        return self.omegaMin, self.omegaMax
    def getDeltaOmega(self, nframes=1):
        assert self.omegas is not None,\
            """instance does not have omega information"""
        return self.omegaDelta * nframes
    def getDark(self):
        'no dark yet supported'
        return 0
    def getFrameOmega(self, iFrame=None):
        """if iFrame is none, use internal counter"""
        assert self.omegas is not None,\
            """instance does not have omega information"""
        if iFrame is None:
            iFrame = self.iFrame
        if hasattr(iFrame, '__len__'):
            'take care of case nframes>1 in last call to read'
            retval = num.mean(self.omegas[iFrame])
        else:
            retval = self.omegas[iFrame]
        return retval

    def __readNext(self, nskip=0):
        if self.img is None:
            raise RuntimeError, 'no image file open'

        if nskip > 0:
            self.img.seek(self.nbytesFrame*nskip, 1)
        data = num.fromfile(self.img,
                            dtype=self.dtypeRead,
                            count=self.nrows*self.ncols)
        return data



    def getWriter(self, filename):
        return None

class ReadGE(object):
    """General reader for omega scans

    Originally, this was for reading GE format images, but this is now
    a general reader accessing the OmegaFrameReader facade class. The main
    functionality to read a sequence of images with associated omega ranges.


    ORIGINAL DOCS
    =============
    Read in raw GE files; this is the class version of the foregoing functions

    NOTES

    *) The flip axis ('v'ertical) was verified on 06 March 2009 by
       JVB and UL.  This should be rechecked if the configuration of the GE
       changes or you are unsure.

    *) BE CAREFUL! nframes should be < 10 or so, or you will run out of
       memory in the namespace on a typical machine.

    *) The header is currently ignored

    *) If a dark is specified, this overrides the use of empty frames as
       background; dark can be a file name or frame

    *) In multiframe images where background subtraction is requested but no
       dark is specified, attempts to use the
       empty frame(s).  An error is returned if there are not any specified.
       If there are multiple empty frames, the average is used.

    """
    def __init__(self, file_info, *args, **kwargs):
        """Initialize the reader

        *file_info* a dictionary providing file and format specifications
        """
        self._ofr = OmegaFrameReader(file_info)  # todo: clarify this

        # initialization
        self.omegas = None
        self.img = None
        self.th  = None
        self.fileInfo      = None
        self.fileInfoR     = None
        self.nFramesRemain = None # remaining in current file
        self.iFrame = -1 # counter for last global frame that was read


        if fileInfo is not None:
            self.__setupRead(fileInfo, self.subtractDark, self.mask, self.omegaStart, self.omegaDelta)

        return


    @classmethod
    def display(cls,
                thisframe,
                roi = None,
                pw  = None,
                **kwargs
                ):
        warnings.warn('display method on readers no longer implemented',
                      ReaderDeprecationWarning)

    def makeNew(self):
        """return a clean instance for the same data files
        useful if want to start reading from the beginning"""
        # Might need this
        inParmDict = {}
        inParmDict.update(self.__inParmDict)
        for key in self.__inParmDict.keys():
            inParmDict[key] = eval("self."+key)
        newSelf = self.__class__(self.fileInfo, **inParmDict)
        return newSelf
    def getRawReader(self, doFlip=False):
        new = self.__class__(self.fileInfo, doFlip=doFlip)
        return new

    def get_nbytes_header(self):
        return self.__nbytes_header
    nbytesHeader = property(get_nbytes_header, None, None)

    def getWriter(self, filename):
        return None

    def __setupRead(self, fileInfo, subtractDark, mask, omegaStart, omegaDelta):

        self.fileInfo = fileInfo
        self.fileListR = self.__convertFileInfo(self.fileInfo)
        self.fileListR.reverse() # so that pop reads in order

        self.subtractDark = subtractDark
        self.mask         = mask

        if self.dead is not None:
            self.deadFlipped = self.__flip(self.dead)

        assert (omegaStart is None) == (omegaDelta is None),\
            'must provide either both or neither of omega start and delta'
        if omegaStart is not None:
            if hasattr(omegaStart, 'getVal'):
                omegaStart = omegaStart.getVal('radians')
            if hasattr(omegaDelta, 'getVal'):
                omegaDelta = omegaDelta.getVal('radians')
            nFramesTot = self.getNFrames()
            self.omegas = \
                num.arange(omegaStart, omegaStart+omegaDelta*(nFramesTot-0.5), omegaDelta) + \
                0.5 * omegaDelta # put omegas at mid-points of omega range for frame
            omegaEnd = omegaStart+omegaDelta*(nFramesTot)
            self.omegaMin = min(omegaStart, omegaEnd)
            self.omegaMax = max(omegaStart, omegaEnd)
            self.omegaDelta = omegaDelta
            self.omegaStart = omegaStart

        self.__nextFile()

        return

    def getNFrames(self):
        """number of total frames with real data, not number remaining"""
        nFramesTot = self.getNFramesFromFileInfo(self.fileInfo)
        return nFramesTot
    def getDeltaOmega(self, nframes=1):
        assert self.omegas is not None,\
            """instance does not have omega information"""
        return self.omegaDelta * nframes
    def getOmegaMinMax(self):
        assert self.omegas is not None,\
            """instance does not have omega information"""
        return self.omegaMin, self.omegaMax
    def frameToOmega(self, frame):
        scalar = num.isscalar(frame)
        frames = num.asarray(frame)
        if frames.dtype == int:
            retval = self.omegas[frames]
        else:
            retval = (frames + 0.5) * self.omegaDelta + self.omegaStart
        if scalar:
            retval = num.asscalar(retval)
        return retval
    def getFrameOmega(self, iFrame=None):
        """if iFrame is none, use internal counter"""
        assert self.omegas is not None,\
            """instance does not have omega information"""
        if iFrame is None:
            iFrame = self.iFrame
        if hasattr(iFrame, '__len__'):
            'take care of case nframes>1 in last call to read'
            retval = num.mean(self.omegas[iFrame])
        else:
            retval = self.omegas[iFrame]
        return retval
    def omegaToFrameRange(self, omega):
        assert self.omegas is not None,\
            'instance does not have omega information'
        assert self.omegaDelta is not None,\
            'instance does not have omega information'
        retval = omeToFrameRange(omega, self.omegas, self.omegaDelta)
        return retval
    def omegaToFrame(self, omega, float=False):
        assert self.omegas is not None,\
            'instance does not have omega information'
        if float:
            assert omega >= self.omegaMin and omega <= self.omegaMax,\
                'omega %g is outside of the range [%g,%g] for the reader' % (omega, self.omegaMin, self.omegaMax)
            retval = (omega - self.omegaStart)/self.omegaDelta - 0.5*self.omegaDelta
        else:
            temp = num.where(self.omegas == omega)[0]
            assert len(temp) == 1, 'omega not found, or found more than once'
            retval = temp[0]
        return retval
    def getFrameUseMask(self):
        """this is an optional toggle to turn the mask on/off"""
        assert isinstance(self.iFrame, int), \
            'self.iFrame needs to be an int for calls to getFrameUseMask'
        if self.useMask is None:
            retval = True
        else:
            assert len(self.useMask) == self.getNFrames(),\
                   "len(useMask) must be %d; yours is %d" % (self.getNFrames(), len(self.useMask))
            retval = self.useMask[self.iFrame]
        return retval
    @classmethod
    def __getNFrames(cls, fileBytes):
        retval = getNFramesFromBytes(fileBytes, cls.__nbytes_header, cls.__nbytes_frame)
        return retval
    def __nextFile(self):

        # close in case already have a file going
        self.close()

        fname, nempty = self.fileListR.pop()

        # open file
        fileBytes = os.stat(fname).st_size
        self.img = open(fname, mode='rb')

        # skip header for now
        self.img.seek(self.__nbytes_header, 0)

        # figure out number of frames
        self.nFramesRemain = self.__getNFrames(fileBytes)

        if nempty > 0:  # 1 or more empty frames
            if self.dark is None:
                scale = 1.0 / nempty
                self.dark = self.frame(dtype=self.__frame_dtype_float)
                for i in range(nempty):
                    self.dark = self.dark + num.fromfile(
                        self.img, **self.__readArgs
                        ).reshape(self.__nrows, self.__ncols) * scale
                self.dark.astype(self.__frame_dtype_dflt)
            else:
                self.img.seek(self.nbytesFrame*nempty, 1)
            self.nFramesRemain -= nempty

        if self.subtractDark and self.dark is None:
            raise RuntimeError, "Requested dark field subtraction, but no file or empty frames specified!"

        return
    @staticmethod
    def __convertFileInfo(fileInfo):
        if isinstance(fileInfo,str):
            fileList = [(fileInfo, 0)]
        elif hasattr(fileInfo,'__len__'):
            assert len(fileInfo) > 0, 'length zero'
            if hasattr(fileInfo[0],'__iter__'): # checking __len__ bad because has len attribute
                fileList = copy.copy(fileInfo)
            else:
                assert len(fileInfo) == 2, 'bad file info'
                fileList = [fileInfo]
        else:
            raise RuntimeError, 'do not know what to do with fileInfo '+str(fileInfo)
        # fileList.reverse()
        return fileList
    def readBBox(self, bbox, raw=True, doFlip=None):
        """
        with raw=True, read more or less raw data, with bbox = [(iLo,iHi),(jLo,jHi),(fLo,fHi)]

        careful: if raw is True, must set doFlip if want frames
        potentially flipped; can set it to a reader instance to pull
        the doFlip value from that instance
        """

        if raw:
            if hasattr(doFlip,'doFlip'):
                'probably a ReadGe instance, pull doFlip from it'
                doFlip = doFlip.doFlip
            doFlip = doFlip or False # set to False if is None
            reader = self.getRawReader(doFlip=doFlip)
        else:
            assert doFlip is None, 'do not specify doFlip if raw is True'
            reader = self.makeNew()

        nskip = bbox[2][0]
        bBox = num.array(bbox)
        sl_i = slice(*bBox[0])
        sl_j = slice(*bBox[1])
        'plenty of performance optimization might be possible here'
        if raw:
            retval = num.empty( tuple(bBox[:,1] - bBox[:,0]), dtype=self.__frame_dtype_read )
        else:
            retval = num.empty( tuple(bBox[:,1] - bBox[:,0]), dtype=self.__frame_dtype_dflt )
        for iFrame in range(retval.shape[2]):
            thisframe = reader.read(nskip=nskip)
            nskip = 0
            retval[:,:,iFrame] = copy.deepcopy(thisframe[sl_i, sl_j])
        if not raw and self.dead is not None:
            'careful: have already flipped, so need deadFlipped instead of dead here'
            mask = num.tile(self.deadFlipped[sl_i, sl_j].T, (retval.shape[2],1,1)).T
            retval = num.ma.masked_array(retval, mask, hard_mask=True, copy=False)
        return retval
    def __flip(self, thisframe):
        if self.doFlip:
            if self.flipArg == 'v':
                thisframe = thisframe[:, ::-1]
            elif self.flipArg == 'h':
                thisframe = thisframe[::-1, :]
            elif self.flipArg == 'vh' or self.flipArg == 'hv':
                thisframe = thisframe[::-1, ::-1]
            elif self.flipArg == 'cw90':
                thisframe = thisframe.T[:, ::-1]
            elif self.flipArg == 'ccw90':
                thisframe = thisframe.T[::-1, :]
            else:
                raise RuntimeError, "unrecognized flip token."
        return thisframe
    def getDark(self):
        if self.dark is None:
            retval = 0
        else:
            retval = self.dark
        return retval
    def read(self, nskip=0, nframes=1, sumImg=False):
        """
        sumImg can be set to True or to something like numpy.maximum
        """

        'get iFrame ready for how it is used here'
        self.iFrame = num.atleast_1d(self.iFrame)[-1]
        iFrameList = []
        multiframe = nframes > 1

        nFramesInv = 1.0 / nframes
        doDarkSub = self.subtractDark # and self.dark is not None

        if doDarkSub:
            assert self.dark is not None, 'self.dark is None'

        # assign storage array
        if sumImg:
            sumImgCallable = hasattr(sumImg,'__call__')
            imgOut = self.frame(dtype=self.__frame_dtype_float, mask=self.dead)
        elif multiframe:
            imgOut = self.frame(nframes=nframes, dtype=self.__frame_dtype_dflt, mask=self.dead)


        # now read data frames
        for i in range(nframes):

            #data = self.__readNext(nskip=nskip)
            #thisframe = data.reshape(self.__nrows, self.__ncols)
            data = self.__readNext(nskip=nskip) # .reshape(self.__nrows, self.__ncols)
            self.iFrame += nskip + 1
            nskip=0 # all done skipping once have the first frame!
            iFrameList.append(self.iFrame)
            # dark subtraction
            if doDarkSub:
                'used to have self.__frame_dtype_float here, but self.__frame_dtype_dflt does the trick'
                thisframe = self.frame(buffer=data,
                                       dtype=self.__frame_dtype_dflt, mask=self.dead) - self.dark
            else:
                thisframe = self.frame(buffer=data,
                                       mask=self.dead)

            # flipping
            thisframe = self.__flip(thisframe)

            # masking (True get zeroed)
            if self.mask is not None:
                if self.getFrameUseMask():
                    thisframe[self.mask] = 0

            # assign output
            if sumImg:
                if sumImgCallable:
                    imgOut = sumImg(imgOut, thisframe)
                else:
                    imgOut = imgOut + thisframe * nFramesInv
            elif multiframe:
                imgOut[i, :, :] = thisframe[:, :]
        'end of loop over nframes'

        if sumImg:
            # imgOut = imgOut / nframes # now taken care of above
            pass
        elif not multiframe:
            imgOut = thisframe

        if multiframe:
            'make iFrame a list so that omega or whatever can be averaged appropriately'
            self.iFrame = iFrameList
        return imgOut

    def __readNext(self, nskip=0):

        if self.img is None:
            raise RuntimeError, 'no image file set'

        nHave = 0

        nskipThis = nskip
        if nskipThis > 0 and data is not None:
            nskipThis = nskipThis - 1
            data = None
        if data is not None : nHave = 1
        #
        while self.nFramesRemain+nHave - nskipThis < 1:
            'not enough frames left in this file'
            nskipThis = nskipThis - self.nFramesRemain
            self.nFramesRemain = 0 # = self.nFramesRemain - self.nFramesRemain
            self.__nextFile()
        if nskipThis > 0:
            # advance counter past empty frames
            self.img.seek(self.nbytesFrame*nskipThis, 1)
            self.nFramesRemain -= nskipThis

        if data is None:
            # grab current frame
            data = num.fromfile(self.img, **self.__readArgs)
            data = num.array(data, **self.__castArgs)
            self.nFramesRemain -= 1

        return data
    def __call__(self, *args, **kwargs):
        return self.read(*args, **kwargs)

    def close(self):
        # if already have a file going, close it out
        if self.img is not None:
            self.img.close()
        return
    """
    getReadDtype function replaced by dtypeRead property
    """
    @classmethod
    def maxVal(cls):
        'maximum value that can be stored in the image pixel data type'
        # dtype = reader._ReadGE__frame_dtype
        # maxInt = num.iinfo(cls.__frame_dtype_read).max # bigger than it really is
        maxInt = 2 ** 14
        return maxInt
    @classmethod
    def getNFramesFromFileInfo(cls, fileInfo, lessEmpty=True):
        fileList = cls.__convertFileInfo(fileInfo)
        nFramesTot = 0
        for fname, nempty in fileList:
            fileBytes = os.stat(fname).st_size
            nFrames = cls.__getNFrames(fileBytes)
            if lessEmpty:
                nFrames -= nempty
            nFramesTot += nFrames
        return nFramesTot

    def indicesToMask(self, indices):
      """
      Indices can be a list of indices, as from makeIndicesTThRanges
      """
      mask = self.getEmptyMask()
      if hasattr(indices,'__len__'):
        for indThese in indices:
          mask[indThese] = True
      else:
        mask[indices] = True
      return mask
#
# Module functions
#
def omeToFrameRange(omega, omegas, omegaDelta):
    """
    check omega range for the frames in
    stead of omega center;
    result can be a pair of frames if the specified omega is
    exactly on the border
    """
    retval = num.where(num.abs(omegas - omega) <= omegaDelta*0.5)[0]
    return retval

def getNFramesFromBytes(fileBytes, nbytesHeader, nbytesFrame):
    assert (fileBytes - nbytesHeader) % nbytesFrame == 0,\
        'file size not correct'
    nFrames = int((fileBytes - nbytesHeader) / nbytesFrame)
    if nFrames*nbytesFrame + nbytesHeader != fileBytes:
        raise RuntimeError, 'file size not correctly calculated'
    return nFrames

def newGenericReader(ncols, nrows, *args, **kwargs):
    """ Currently just returns a Framer2DRC
    """

    # retval = Framer2DRC(ncols, nrows, **kwargs)
    filename = kwargs.pop('filename', None)
    retval = ReadGeneric(filename, ncols, nrows, *args, **kwargs)

    return retval

class ReaderDeprecationWarning(DeprecationWarning):
    """Warnings on use of old reader features"""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
