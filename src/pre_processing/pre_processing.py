from typing import *
import numpy as np
import os
import cv2
from scipy import ndimage
from utils.common_functions import read_images, change_gray_range

def preprocess_image(image: np.ndarray) -> np.ndarray:
    resize_ratio = 0.1

    image = cv2.resize(image, (int(image.shape[1] * resize_ratio), int(image.shape[0] * resize_ratio)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

    lower_bound = np.array([0, 133, 77])
    upper_bound = np.array([255, 173, 127])
    image = cv2.inRange(image, lower_bound, upper_bound)

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)

    image = ndimage.binary_fill_holes(image).astype(np.int8)
    image = change_gray_range(image, format=255)

    return image


##############################################################################################################
##############################################################################################################
#####################################   U S E   T H I S   O N L Y   ##########################################
##############################################################################################################
##############################################################################################################

def run_preprocessor(dataset_path: str, dest_path: str):
    """
        Preprocesses the images and saves them in the destination path
        @param dataset_path: the path to the dataset
        @param dest_path: the path to save the preprocessed images
        @return: array of preprocessed images and array of labels
    """

    images, labels = read_images(dataset_path)

    # preprocess all images
    pp_images = []
    for i in range(len(images)):
        print(f'Preprocessing image {i}...')
        images[i] = preprocess_image(images[i])
        cv2.imwrite(os.path.join(dest_path, f'{labels[i]}_{i}.JPG'), images[i])
        pp_images.append(images[i])
    
    return pp_images, labels