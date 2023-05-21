import pydicom as pyd
import os
import numpy as np
from constants import *
import cv2

class ImageReader():

    def execute(self, basepath, batch):
        pass

    def read_batch_images(self, basepath, batch, filetype):
        filetypes = {
            FILE_TYPE_DICOM: DicomReader(),
            FILE_TYPE_PNG: PNGReader()
        }
        return filetypes.get(filetype, self).execute(basepath, batch)


class PNGReader(ImageReader):

    def execute(self, basepath, batch):
        png_pixel_data = [cv2.imread(os.path.join(basepath, x + FILE_TYPE_PNG)) for x in batch]
        image_batch = np.array(png_pixel_data)
        return image_batch


class DicomReader(ImageReader):

    def execute(self, basepath, batch):
        dicom_data_list = [pyd.dcmread(os.path.join(basepath, x + FILE_TYPE_DICOM)) for x in batch]
        dicom_pixel_data = [x.pixel_array for x in dicom_data_list]
        image_batch = np.array(dicom_pixel_data)
        return image_batch
