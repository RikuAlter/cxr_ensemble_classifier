import configparser
import logging
import os
import cv2

import boto3
from tqdm import tqdm

from constants import constants
from preprocessors.image_processor import GenericDenoiser, ImageEnhancer, resize_with_ratio
from preprocessors.image_reader import ImageReader


class InfoErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno in (logging.INFO, logging.ERROR)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('pipeline.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
info_error_filter = InfoErrorFilter()
stream_handler.addFilter(info_error_filter)

logger = logging.getLogger()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class ImageReaderPipeline:

    def __init__(self, filetype, source):
        self.filetype = filetype
        self.source = source

    def execute(self, ids):
        images = ImageReader().execute(basepath=self.source, batch=ids, filetype=self.filetype)
        logger.info("Completed reading image batch!")
        return images


class ProcessImagePipeline:

    def __init__(self, denoise_params):
        self.denoise_params = denoise_params

    def execute(self, images, resizeDimension=(600, 600)):
        logger.info("Starting processing of image batch..")
        images = GenericDenoiser().execute(images, denoiseParams=self.denoise_params)
        images = ImageEnhancer().execute(images)
        for _id in range(len(images)):
            images[_id] = resize_with_ratio(images[_id], resizeDimension, cv2.INTER_AREA)
        logger.info("Completed processing image batch!")
        return images


class UploadPipeline:

    def __init__(self, s3Config, iamProfile="default"):
        self.s3Config = configparser.ConfigParser()
        self.s3Config.read(s3Config)
        self.s3Client = boto3.Session(profile_name=iamProfile).client(constants.S3)

    def execute(self, source, ids, filetype):
        logger.info("Started upload image batch")
        targetBucket = self.s3Config.get(constants.S3, constants.S3_BUCKET)
        targetFolder = self.s3Config.get(constants.S3, constants.S3_IMAGE_PREFIX)
        for _id in tqdm(ids):
            self.s3Client.upload_file(os.path.join(source, _id + filetype), targetBucket, targetFolder + _id + filetype)
        logger.info("Completed uploading image batch!")
