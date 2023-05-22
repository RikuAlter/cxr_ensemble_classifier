from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.image import ImageDataGenerator
from constants import *
import ast
import random

class DataArranger():

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
        return self.splitter.split(x,y)

    def get_class_count(self, n_classes, labels):
        class_count = [0]*n_classes
        for label in labels:
            label = ast.literal_eval(label)
            for x in label:
                class_count[x] = class_count[x] + 1
        return class_count

    def get_median(self, counts):
        class_count = len(counts)
        mid_id = int(class_count / 2)
        ordered_count = sorted(counts)
        return ordered_count[mid_id] if class_count % 2 == 0 else int((ordered_count[mid_id] + ordered_count[mid_id+1])/2)

    def get_median_drift(self, count, median):
        median_drift = []
        median_drift = [median-x for x in count]
        return median_drift

    def get_class_weight(self, labels, n_classes):
        class_weights = []
        for label in labels:
            weights = [0]*n_classes
            for x in label:
                weights[x] = 1/len(label) if len(label) > 0 else 0
            class_weights.append(weights)
        return class_weights

    def update_class_count(self, class_count, new_labels):
        for label in new_labels:
            for x in label:
                class_count[x] = class_count[x] + 1
        return class_count

    def get_class_indices(self, labels, n_classes):
        index_list = []
        for n_class in range(n_classes):
            class_indices = [x for x, label in enumerate(labels) if n_class in label]
            index_list.append(class_indices)
        return index_list

    def get_augmented_data(self, x, y, n_classes, min_genuine_threshold):
        class_count = self.get_class_count(n_classes, y)
        class_weights = self.get_class_weight(y, n_classes)
        class_ids = self.get_class_indices(y, n_classes)
        median = self.get_median(class_count)
        median_drift = self.get_median_drift(class_count, median)

        augmented_images = []
        augmented_labels = []

        for label in range(n_classes):
            max_drift = max(median_drift)
            n_class_ids = class_ids[label]
            genuine_sample_ids = []
            if median_drift[label] < 0:
                drift_strength = median_drift[label]/max_drift
                required_samples = int(median - median_drift[label])

                if drift_strength >= 0.6:
                    weight_threshold = 0.3
                elif drift_strength >= 0.3:
                    weight_threshold = 0.6
                else:
                    weight_threshold = 1.0
                genuine_sample_ids = [x for x, weight in enumerate(class_weights) if (weight_threshold-0.3) <= weight[label] <= weight_threshold]

                if len(genuine_sample_ids) < min_genuine_threshold:
                    upper_weigth_threshold = min(min(x[label] for x in class_weights if x[label] > weight_threshold) + 0.2, 1)
                    lower_weight_threshold = min(0, min(x[label] for x in class_weights if x[label] < (weight_threshold-0.3)) - 0.2)
                    genuine_sample_ids = [x for x, weight in enumerate(class_weights) if lower_weight_threshold <= weight[label] <= upper_weigth_threshold]

                if(len(genuine_sample_ids) == 0):
                    genuine_sample_ids = random.sample(n_class_ids,min_genuine_threshold)

                x_label = x[genuine_sample_ids]
                y_label = y[genuine_sample_ids]

                augmented_data = self.augmentor.flow(x_label, y_label, batch_size=required_samples, shuffle=True)
                n_label_augmented_images, n_label_augmented_labels = augmented_data.next()

                augmented_images.extend(n_label_augmented_images)
                augmented_labels.extend(n_label_augmented_labels)

                class_count = self.update_class_count(class_count, augmented_labels)
                median_drift = self.get_median_drift(class_count, median)
        return augmented_images, augmented_labels