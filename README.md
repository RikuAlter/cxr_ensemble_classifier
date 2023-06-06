# Ensemble DiagnoVision

This repository is currently in the making.

Here is a brief overview of the different components:

1. Findings Processor: The label data for the images has multiple diagnosis present against per image, the task of this processor is to process this label data file and consolidate the details against one particular image as one entry in the dataframe.
2. Data Arranger: The dataset is heavily sekewed for some classes of diagnosis, the purpose of this module is to streamline processes like data augmentation, stratified split, etc, so that we can seamlessly incorporate them into out model training pipeline down the line.
3. Image Reader: The dataset currently being used consists of DICOM images. However, we wish to perform extensive training of the model on multiple datasets, thus we have written this module to streamline the process.
4. Image Processer: This module is to apply denoising and enhancement algorithms seamlessly.
