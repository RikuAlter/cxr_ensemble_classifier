#Error Codes
AUGMENTATION_ERROR_MISSING_GENUINE_IMAGES = "Missing genuine images to create augments of."

#Column names from original data frame
UNPROCESSED_COLUMN_NAME_IMAGE_ID = "image_id"
UNPROCESSED_COLUMN_NAME_CLASS_ID = "class_id"
UNPROCESSED_COLUMN_NAME_CLASS_NAME = "class_name"
UNPROCESSED_COLUMN_NAME_X_MIN = "x_min"
UNPROCESSED_COLUMN_NAME_Y_MIN = "y_min"
UNPROCESSED_COLUMN_NAME_X_MAX = "x_max"
UNPROCESSED_COLUMN_NAME_Y_MAX = "y_max"

#Column names for processed data frame
PROCESSED_COLUMN_NAME_CLASS_IDS = "class_ids"
PROCESSED_COLUMN_NAME_CLASS_NAMES = "class_names"

#File Reader constants
FILE_TYPE_DICOM = ".dicom"
FILE_TYPE_PNG = ".png"
FINDING_NO_FINDING = "No finding"

#Image Processor Constants
INTER_SPLINE = "Spline"
WAVELET_THRESHOLD_METHOD_BAYESHRINK = "BayesShrink"
WAVELET_THRESHOLD_METHOD_VISUSHRINK = "VisuShrink"
WAVELET_TYPE_DB1 = "db1"
WAVELET_TYPE_DB2 = "db2"
WAVELET_TYPE_HAAR = "haar"
WAVELET_TYPE_SYM2 = "sym2"
WAVELET_TYPE_COIFI = "coifi"
DENOISE_METHOD_BM3D = "bm3d"
DENOISE_METHOD_NLM = "nlm"
DENOISE_METHOD_WAVELET = "wavelet"
DENOISE_PARAM_SIGMA_PSD = "sigma_psd"
DENOISE_PARAM_NLM_H = "nlm_h"
DENOISE_PARAM_NLM_TEMPLATE_WINDOW_SIZE = "templateWindowSize"
DENOISE_PARAM_NLM_SEARCH_WINDOW_SIZE = "searchWindowSize"
DENOISE_PARAM_NLM_WAVELET_TYPE = "wavelet"
DENOISE_PARAM_NLM_WAVELET_THRESHOLD = "threshold"


#Data Arranger constants
SPLITTER_SPLIT_COUNT = "n_splits"
SPLITTER_TEST_SIZE = "test_size"
SPLITTER_RANDOM_STATE = "random_state"
AUGMENTOR_ROTATION_RANGE = "rotation_range"
AUGMENTOR_WIDTH_SHIFT = "width_shift_range"
AUGMENTOR_HEIGHT_SHIFT = "height_shift_range"
AUGMENTOR_SHEAR_RANGE = "shear_range"
AUGMENTOR_ZOOM_RANGE = "zoom_range"
AUGMENTOR_FILL_MODE = "fill_mode"

#Anomaly Classes
NO_ANOMALY = "No finding"