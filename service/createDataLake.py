# The purpose of this script is to create a data lake in AWS S3
# Instead of uploading the DICOM or PNG images, we are uploading processed versions of the same
#
# This script will be responsible to orchestrate the various pipelines defined in
# ../pipelines and create a data lake in S3

# The operating steps are:
# 1 -> Extract a batch of image IDs from processed_label_data.csv
# 2 -> Preprocess the batch of images [Denoise -> Contrast Enhancement -> Resize]
# 3 -> Stores the images locally
# 4 -> Upload the images to S3 bucket

# Imports
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

import pandas as pd
import cv2
from pipelines.pipelines import ImageReaderPipeline, ProcessImagePipeline, UploadPipeline
from constants.constants import *


def orchestrate_data_lake_create(sourceIdFile, imageSource, idColumn, start_index, end_index):
    """

    Args:
        sourceIdFile: Dataframe - CSV, must store the list of ids to upload
        imageSource: String, source image folder
        idColumn: String, must match the column name representing the image IDs in sourceIDFile
        start_index: Integer, Starting index for the image ids to upload, image ids obtained from sourceIdFile
        end_index: Integer, Ending index for the same, if None provided

        Regarding denoising of the images:
            Patch based NLM was chosen with patch size of 15 and intermediary patch distance of 3 with H=0.2
        Regarding enhancement of images:
            CLAHE is being used for contrast enhancement with clip limit of 1.5 and grid size of 8 x 8
    """

    # Preparing the denoising parameters for Patch based NLM
    denoiseParams = {
        DENOISE_METHOD: DENOISE_METHOD_NLM_PATCH,
        DENOISE_PARAM_NLM_H: 0.2,
        DENOISE_PARAM_PATCH_NLM_PATCH_SIZE: 15,
        DENOISE_PARAM_PATCH_NLM_PATCH_DISTANCE: 3
    }

    idList = pd.read_csv(sourceIdFile)[start_index:end_index]
    images = ImageReaderPipeline(filetype=FILE_TYPE_DICOM, source=imageSource, ids=idList[idColumn]).execute()
    images = ProcessImagePipeline(denoise_params=denoiseParams).execute(images)
    for image in images: save_image("dataLake", image)
    UploadPipeline().execute(source="dataLake", ids=idList, filetype=FILE_TYPE_PNG)


def save_image(path, image):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path, image)


id_source_file = os.path.join(".", "processed_label_data.csv")
image_source = "train"
id_column = UNPROCESSED_COLUMN_NAME_IMAGE_ID
start_id = 0
end_id = 10

orchestrate_data_lake_create(sourceIdFile=id_source_file, imageSource=image_source, idColumn=id_column,
                             start_index=start_id, end_index=end_id)
