from typing import *
import numpy as np
import pandas as pd
import os
from utils.common_functions import *
import cv2
from pyefd import elliptic_fourier_descriptors, reconstruct_contour, plot_efd, normalize_efd


def elliptical_fourier(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = 255 // 2
    image[image > threshold] = 255
    image[image <= threshold] = 0

    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key = cv2.contourArea)

    max_contour = max_contour.reshape(max_contour.shape[0], max_contour.shape[2])
    coefficients = elliptic_fourier_descriptors(max_contour, order=10)

    coefficients = normalize_efd(coefficients)
    return coefficients.flatten()[3:]

##############################################################################################################
##############################################################################################################
#####################################   U S E   T H I S   O N L Y   ##########################################
##############################################################################################################
##############################################################################################################

def extract_features(pp_images):
    images_features = []

    for i, image in enumerate(pp_images):            
        print(f'Extracting features from image [EFD] {i + 1}...')
        features = elliptical_fourier(image)
        images_features.append(features)

    return np.array(images_features)



def run_elliptical_fourier(pp_dataset_path: str):
    """
        Extracts the EFD features from coefficints of the images using OpenCV built-in function 
        and saves them in the destination path
        @param pp_dataset_path: the path to the preprocessed dataset
        @param dest_path: the path to save the features
        @return: 2d numpy array of features (each row represents an image) and 1d numpy array of labels
    """

    images_features = []
    labels = []
    for dirpath, _, filenames in os.walk(pp_dataset_path):
        if not filenames:
            continue

        for file in filenames:
            if not file.endswith('.jpg') and not file.endswith('.JPG'):
                print(f'File {file} is not a jpg file. Skipping...')
                continue

            file_path = os.path.join(dirpath, file)

            # to avoid reading corrupted images
            image = cv2.imread(file_path)
            if image is None:
                print(f'File {file} is not a valid image. Skipping...')
                continue
            
            print(f'Extracting features from image [EFD]: {file}...')
            features = elliptical_fourier(image)
            images_features.append(features)
            labels.append(int(file[0]))

    return np.array(images_features), np.array(labels)