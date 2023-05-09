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

def run_elliptical_fourier(pp_dataset_path: str, dest_path: str):
    """
        Extracts the EFD features from coefficints of the images using OpenCV built-in function 
        and saves them in the destination path
        @param pp_dataset_path: the path to the preprocessed dataset
        @param dest_path: the path to save the features
        @return: 2d numpy array of features (each row represents an image) and 1d numpy array of labels
    """
    images, labels = read_images(pp_dataset_path)

    images_features = []
    for i in range(len(images)):
        print(f'Extracting features from image [EFD]: {i}...')
        features = elliptical_fourier(images[i])
        images_features.append(features)
    
    images_features = np.array(images_features)

    np.save(os.path.join(os.path.dirname(__file__), dest_path, 'efd_features.npy'), images_features)

    return images_features, labels