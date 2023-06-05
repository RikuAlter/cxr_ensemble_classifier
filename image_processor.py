import cv2
import numpy as np
import random
from image_reader import ImageReader
from constants import FILE_TYPE_DICOM
from tqdm import tqdm
from numpy.fft import fft2, fftshift

class BM3DDenoiser:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def sigma_estimator(self, batch_size = 200, epoch = 20):

        total_sample_size = len(self.dataframe)

        for i in tqdm(range(epoch)):
            current_batch_size = batch_size * (i+1)
            sampled_set = self.dataframe.iloc[random.sample(range(total_sample_size), current_batch_size)]
            sampled_image_ids = sampled_set["image_id"]

            imageReader = ImageReader()

            sampled_images = imageReader.read_batch_images("train", sampled_image_ids, FILE_TYPE_DICOM)

            nps_mean_sum = 0

            for image in sampled_images:
                image = (image - np.min(image))/(np.max(image) - np.min(image))
                image_fft = fftshift(fft2(image))
                nps = np.abs(image_fft) ** 2

                mean_nps  = np.mean(nps)
                nps_mean_sum += mean_nps

            mean_nps_mean = nps_mean_sum/current_batch_size
            sigma_psd = np.sqrt(mean_nps_mean)
            print("Epoch :", i+1, "For batch size: ", current_batch_size, "Sigma PSD: ", sigma_psd)



def expand_channel_resize_image(images, dims, interpolationFlag, expand_dims=False):
    resized_images = []
    for image in images:
        normalized_image = (image - np.min(image)) / (np.max(image)-np.min(image))
        resized_image = resize_with_ratio(normalized_image, dims, interpolationFlag)
        if expand_dims:
            resized_image = np.expand_dims(resized_image, axis=-1)
        resized_images.append(resized_image)
    return np.asarray(resized_images)


def resize_with_ratio(image, dims, interpolationFlag):
    height, width = image.shape
    scale_factor = min(dims[0] / width, dims[1] / height)

    target_original_height = int(height * scale_factor)
    target_original_width = int(width * scale_factor)

    downscaled_image = cv2.resize(image, (target_original_width, target_original_height),
                                  interpolation=interpolationFlag)

    normalized_image = (downscaled_image - np.min(downscaled_image)) / (np.max(downscaled_image) - np.min(downscaled_image))

    pad_top = (dims[1] - target_original_height) // 2
    pad_bottom = dims[1] - target_original_height - pad_top
    pad_left = (dims[0] - target_original_width) // 2
    pad_right = dims[0] - target_original_width - pad_left

    padded_image = cv2.copyMakeBorder(normalized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return padded_image

def denoise_wavelet(image):
    pass

def denoise_bm3d(image):
    pass

def denoise_nlm(image):
    pass

def bm3d_sigma_estimator():
    pass
