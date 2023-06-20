import pydicom as pyd
import os
import numpy as np
from constants import *
import cv2

class ImageReader():

    def execute(self, basepath, batch, filetype):
        filetypes = {
            FILE_TYPE_DICOM: DicomReader(),
            FILE_TYPE_PNG: PNGReader()
        }
        return filetypes.get(filetype, self).execute(basepath, batch, filetype)

    def normalize_image(self, image, doNormalization = True):
        if doNormalization:
            image = (image - np.min(image))/(np.max(image) - np.min(image))
        return image


class PNGReader(ImageReader):

    def execute(self, basepath, batch, filetype):
        png_pixel_data = [super(PNGReader, self).normalize_image(cv2.imread(os.path.join(basepath, x + filetype))) for x in batch]
        image_batch = np.array(png_pixel_data)
        return image_batch


class DicomReader(ImageReader):

    def execute(self, basepath, batch, filetype):
        dicom_data_list = [super(DicomReader, self).normalize_image(pyd.dcmread(os.path.join(basepath, x + filetype)).pixel_array) for x in batch]
        image_batch = np.array(dicom_data_list)
        return image_batch
