import os
import boto3
import configparser
import pydicom as pyd
import bm3d
import cv2
import constants
import numpy as np
import time

from image_processor import resize_with_ratio
from tqdm import tqdm


class AWSPipeline:

    def __init__(self, bm3dSigma, claheClip, claheGrid):
        self.s3Config = configparser.ConfigParser()
        self.s3Config.read("aws_config.properties")
        self.session = None
        self.s3conn = None
        self.sigma = bm3dSigma
        self.clahe = cv2.createCLAHE(clipLimit = claheClip, tileGridSize = claheGrid)

    def establish_aws_session(self):
        self.session = boto3.Session(profile_name="default")
        return self

    def establish_s3_connection(self):
        self.s3conn = self.session.client(constants.S3)
        return self

    def upload_to_lake(self, filePath, fileName, fileType):
        bucket = self.s3Config.get(constants.S3, constants.S3_BUCKET)
        image_prefix = self.s3Config.get(constants.S3, constants.S3_IMAGE_PREFIX)
        self.s3conn.upload_file(os.path.join(filePath, fileName + fileType), bucket, image_prefix + fileName + fileType)

    def process_image(self, basepath, image_ids, filetype):

        for image_id in tqdm(image_ids):
            start_time = time.time()
            image = pyd.dcmread(os.path.join(basepath, image_id + filetype)).pixel_array
            image = (image - np.min(image))/(np.max(image) - np.min(image))
            image = bm3d.bm3d(image, sigma_psd = self.sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            image = (image*255).astype("uint8")
            image = self.clahe.apply(image)
            image = resize_with_ratio(image, (600,600), cv2.INTER_CUBIC)
            save_image(os.path.join("dataLake", "bm3d", image_id+constants.FILE_TYPE_PNG), image)

            bucket = self.s3Config.get(constants.S3, constants.S3_BUCKET)
            image_prefix = self.s3Config.get(constants.S3, constants.S3_IMAGE_PREFIX)
            s3conn = self.establish_aws_session().establish_s3_connection()
            self.s3conn.upload_file(os.path.join(os.path.join("dataLake", "bm3d"), image_id + constants.FILE_TYPE_PNG), bucket, image_prefix + image_id + constants.FILE_TYPE_PNG)


def save_image(path, image):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path, image)