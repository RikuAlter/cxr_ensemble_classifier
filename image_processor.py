import cv2
import numpy as np


def expand_channel_resize_image(images, dims, interpolationFlag, expand_dims=False):
    resized_images = []
    for image in images:
        normalized_image = (image - np.min(image)) / np.max(image)
        resized_image = resize_with_ratio(normalized_image, dims, interpolationFlag)
        resized_image = (resized_image * 255).astype(np.uint8)
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

    underlay = np.zeros(dims, dtype=np.uint8)

    pad_top = (dims[1] - target_original_height) // 2
    pad_bottom = dims[1] - target_original_height - pad_top
    pad_left = (dims[0] - target_original_width) // 2
    pad_right = dims[0] - target_original_width - pad_left

    padded_image = cv2.copyMakeBorder(downscaled_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return padded_image
