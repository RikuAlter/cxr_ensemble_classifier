from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator
from constants import *
import ast


class DataArranger:

    def __init__(self, splitconfig, augmentconfig):
        self.splitconfig = splitconfig
        self.splitter = StratifiedShuffleSplit(n_splits=splitconfig.get(SPLITTER_SPLIT_COUNT),
                                               test_size=splitconfig.get(SPLITTER_TEST_SIZE),
                                               random_state=splitconfig.get(SPLITTER_RANDOM_STATE))
        self.augmentor = ImageDataGenerator(rotation_range=augmentconfig.get(AUGMENTOR_ROTATION_RANGE),
                                            width_shift_range=augmentconfig.get(AUGMENTOR_WIDTH_SHIFT),
                                            height_shift_range=augmentconfig.get(AUGMENTOR_HEIGHT_SHIFT),
                                            shear_range=augmentconfig.get(AUGMENTOR_SHEAR_RANGE),
                                            zoom_range=augmentconfig.get(AUGMENTOR_ZOOM_RANGE),
                                            fill_mode=augmentconfig.get(AUGMENTOR_FILL_MODE))

    def shuffle_split(self, x, y):
        return self.splitter.split(x, y)

    def class_count(self, n_classes, labels):
        class_count = [0] * n_classes
        for label in labels:
            label = ast.literal_eval(label)
            for x in label:
                class_count[x] = class_count[x] + 1

    def get_median(self, counts):
        class_count = len(counts)
        mid_id = int(class_count / 2)
        ordered_count = sorted(counts)
        return ordered_count[mid_id] if class_count % 2 == 0 else int(
            (ordered_count[mid_id] + ordered_count[mid_id + 1]) / 2)

    def balance_split(self, x, y):
        pass
