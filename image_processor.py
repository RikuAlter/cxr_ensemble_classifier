import cv2
import numpy as np
import random
from image_reader import ImageReader
from constants import FILE_TYPE_DICOM, INTER_SPLINE, WAVELET_THRESHOLD_METHOD_BAYESHRINK, \
    UNPROCESSED_COLUMN_NAME_IMAGE_ID, WAVELET_TYPE_DB1, WAVELET_TYPE_SYM3, DENOISE_METHOD_WAVELET, DENOISE_METHOD_NLM, \
    DENOISE_METHOD_BM3D, DENOISE_PARAM_NLM_H, DENOISE_PARAM_SIGMA_PSD, DENOISE_PARAM_NLM_WAVELET_THRESHOLD, DENOISE_PARAM_NLM_WAVELET_TYPE, \
    DENOISE_PARAM_NLM_SEARCH_WINDOW_SIZE, DENOISE_PARAM_NLM_TEMPLATE_WINDOW_SIZE
from tqdm import tqdm
from numpy.fft import fft2, fftshift
from scipy.ndimage import zoom
import bm3d
from image_reader import ImageReader
from skimage.restoration import denoise_wavelet, estimate_sigma


class GenericDenoiser:

    def __init__(self, imreader=None):
        self.imreader = ImageReader() if imreader == None else imreader

    def execute(self, images, denoiseMethod=DENOISE_METHOD_WAVELET, sigma_psd = 0, h=10, templateWindowSize=7, searchWindowSize=21,
                wavelet=WAVELET_TYPE_SYM3, thresholdMethod=WAVELET_THRESHOLD_METHOD_BAYESHRINK):
        denoisers = {
            DENOISE_METHOD_BM3D: BM3DDenoiser(),
            DENOISE_METHOD_NLM: ClassicNLMDenoiser(),
            DENOISE_METHOD_WAVELET: WaveletDenoiser()
        }

        denoise_param ={
            DENOISE_PARAM_SIGMA_PSD : sigma_psd,
            DENOISE_PARAM_NLM_H: h,
            DENOISE_PARAM_NLM_TEMPLATE_WINDOW_SIZE: templateWindowSize,
            DENOISE_PARAM_NLM_SEARCH_WINDOW_SIZE: searchWindowSize,
            DENOISE_PARAM_NLM_WAVELET_TYPE: wavelet,
            DENOISE_PARAM_NLM_WAVELET_THRESHOLD: thresholdMethod
        }

        denoised_images = []

        denoiser = denoisers.get(denoiseMethod)

        for image in images:
            denoised_images.append(denoiser.execute(image, denoise_param))

        return np.asarray(denoised_images)

    def convert_to_uint8(self, image):
        return (image*255).astype("uint8")


class BM3DDenoiser(GenericDenoiser):

    def __init__(self, dataframe = None):
        super(BM3DDenoiser, self).__init__()
        self.dataframe = dataframe

    def execute(self, image, denoise_params):
        denoised_image = bm3d.bm3d(image, sigma_psd = denoise_params.get(DENOISE_PARAM_SIGMA_PSD))
        return denoised_image

    def sigma_estimator(self, imageBasePath, filetype, downsize_dim, interpolationFlag, batch_size = 100, epoch = 20):

        total_sample_size = len(self.dataframe)

        for i in tqdm(range(epoch)):
            current_batch_size = batch_size * (i+1)
            sampled_set = self.dataframe.iloc[random.sample(range(total_sample_size), current_batch_size)]
            sampled_image_ids = sampled_set[UNPROCESSED_COLUMN_NAME_IMAGE_ID]

            sampled_images = self.imreader.execute(imageBasePath, sampled_image_ids, filetype)

            sigma_psd_sum = 0

            for image in sampled_images:
                height, width = image.shape
                scale_factor = min(downsize_dim[0] / width, downsize_dim[1] / height)

                target_original_height = int(height * scale_factor)
                target_original_width = int(width * scale_factor)

                downscaled_image = cv2.resize(image, (target_original_width, target_original_height),
                                              interpolation=interpolationFlag)

                downscaled_image = (downscaled_image - np.min(downscaled_image))/(np.max(downscaled_image) - np.min(downscaled_image))
                image_fft = fftshift(fft2(downscaled_image))
                image_fft = (image_fft - np.min(image_fft))/(np.max(image_fft) - np.min(image_fft))
                nps = np.abs(image_fft) ** 2

                mean_nps = np.mean(nps)
                sigma_psd = np.sqrt(mean_nps)
                sigma_psd_sum += sigma_psd

            print("Epoch :", i+1, "For batch size: ", current_batch_size, "Sigma PSD: ", sigma_psd_sum/current_batch_size)


class ClassicNLMDenoiser(GenericDenoiser):

    def execute(self, image, denoise_param):
        image = super().convert_to_uint8(image)
        denoised_image = cv2.fastNlMeansDenoising(image, None, h=denoise_param.get(DENOISE_PARAM_NLM_H),
                                                  templateWindowSize=denoise_param.get(DENOISE_PARAM_NLM_TEMPLATE_WINDOW_SIZE),
                                                  searchWindowSize=denoise_param.get(DENOISE_PARAM_NLM_SEARCH_WINDOW_SIZE))
        denoised_image = self.imreader.normalize_image(denoised_image)
        return denoised_image


class WaveletDenoiser(GenericDenoiser):

    def execute(self, image, denoise_param):
        denoised_image = denoise_wavelet(image, method = denoise_param.get(DENOISE_PARAM_NLM_WAVELET_THRESHOLD),
                                         wavelet = denoise_param.get(DENOISE_PARAM_NLM_WAVELET_TYPE))
        return denoised_image


class ImageEnhancer:

    def __init__(self, clipLimit = 2, tileGridSize=(5,5)):
        self.clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = tileGridSize)

    def execute(self, images):

        enhanced_images = []

        for image in images:
            gray_image = (image*255).astype("uint8")
            enhanced_images.append(self.clahe.apply(gray_image))

        return np.asarray(enhanced_images)


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

    if interpolationFlag == INTER_SPLINE:
        downscaled_image = zoom(image, scale_factor, order = 3)
    else:
        downscaled_image = cv2.resize(image, (target_original_width, target_original_height),
                                      interpolation=interpolationFlag)

    pad_top = (dims[1] - target_original_height) // 2
    pad_bottom = dims[1] - target_original_height - pad_top
    pad_left = (dims[0] - target_original_width) // 2
    pad_right = dims[0] - target_original_width - pad_left

    padded_image = cv2.copyMakeBorder(downscaled_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return padded_image
