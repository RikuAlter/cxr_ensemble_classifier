import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from constants import *


warnings.simplefilter("ignore")


def fetch_encoded_labels(labels):
    encoder = MultiLabelBinarizer()
    return encoder.fit_transform(labels)


def build_processed_data(filepath):
    raw_data = pd.read_csv(filepath)
    detection_zone = [UNPROCESSED_COLUMN_NAME_X_MIN, UNPROCESSED_COLUMN_NAME_Y_MIN,
                      UNPROCESSED_COLUMN_NAME_X_MAX, UNPROCESSED_COLUMN_NAME_Y_MAX]

    class_detection_zones = []
    class_list = raw_data.loc[:, UNPROCESSED_COLUMN_NAME_CLASS_NAME].unique().tolist()
    class_list.remove(FINDING_NO_FINDING)
    for n_class in class_list:
        class_detection_zones.extend([n_class + "_" + x for x in detection_zone])

    columns = [UNPROCESSED_COLUMN_NAME_IMAGE_ID, PROCESSED_COLUMN_NAME_CLASS_NAMES, PROCESSED_COLUMN_NAME_CLASS_IDS]
    columns.extend(class_detection_zones)
    prepared_image_data = pd.DataFrame(columns=columns)
    grouped_df = raw_data.groupby(UNPROCESSED_COLUMN_NAME_IMAGE_ID)

    for im_id, group in tqdm(grouped_df):
        row_data = [im_id]
        row_data.extend([np.nan] * (len(columns) - 1))
        new_row = pd.Series(row_data, index=columns)
        anomaly_list = []
        anomaly_id = []
        detection_loc = [-1] * 56
        for _id, row in group.iterrows():
            anomaly = row[UNPROCESSED_COLUMN_NAME_CLASS_NAME]
            if anomaly not in anomaly_list:
                anomaly_list.append(row[UNPROCESSED_COLUMN_NAME_CLASS_NAME])
                anomaly_id.append(row[UNPROCESSED_COLUMN_NAME_CLASS_ID])
                if anomaly != FINDING_NO_FINDING :
                    new_row[anomaly + "_" + UNPROCESSED_COLUMN_NAME_X_MIN] = row[UNPROCESSED_COLUMN_NAME_X_MIN]
                    new_row[anomaly + "_" + UNPROCESSED_COLUMN_NAME_Y_MIN] = row[UNPROCESSED_COLUMN_NAME_Y_MIN]
                    new_row[anomaly + "_" + UNPROCESSED_COLUMN_NAME_X_MAX] = row[UNPROCESSED_COLUMN_NAME_X_MAX]
                    new_row[anomaly + "_" + UNPROCESSED_COLUMN_NAME_Y_MAX] = row[UNPROCESSED_COLUMN_NAME_Y_MAX]
        new_row[PROCESSED_COLUMN_NAME_CLASS_NAMES] = ",".join(str(x) for x in anomaly_list)
        new_row[PROCESSED_COLUMN_NAME_CLASS_IDS] = anomaly_id
        prepared_image_data = prepared_image_data.append(new_row, ignore_index=True)
    prepared_image_data = prepared_image_data.fillna(-1)
    return prepared_image_data
