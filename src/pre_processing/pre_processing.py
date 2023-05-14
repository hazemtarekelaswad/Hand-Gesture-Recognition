from typing import *
import numpy as np
import os
import cv2
from scipy import ndimage
from utils.common_functions import read_images, change_gray_range

'''
Crops image from the left or the right side based on the white pixels count with the threshold,
and crops the other side (black side) with a fixed value of 20 pixels.
'''
def crop_image(image, threshold=40):
    left = image[:, :threshold]
    right = image[:, -threshold:]

    white_count_left = np.sum(left == 255)
    white_count_right = np.sum(right == 255)
    
    if white_count_left > white_count_right:
        image = image[:, threshold:]
    else:
        image = image[:, :-threshold]
    return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    resize_ratio = 0.1

    image = cv2.resize(image, (int(image.shape[1] * resize_ratio), int(image.shape[0] * resize_ratio)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

    # Hand has high CR value and low CB value
    lower_bound = np.array([0, 133, 77])
    upper_bound = np.array([255, 173, 127]) 
    image = cv2.inRange(image, lower_bound, upper_bound)

    image = cv2.erode(image, np.ones((5, 5), np.uint8), iterations=1)
    # image = cv2.dilate(image, np.ones((5, 5), np.uint8), iterations=1)

    image = ndimage.binary_fill_holes(image).astype(np.int8)
    image = change_gray_range(image, format=255)
    # image = crop_image(image)

    return image


def run_preprocessor(dataset_path: str, dest_path: str):
    """
        Preprocesses the images and saves them in the destination path
        @param dataset_path: the path to the dataset
        @param dest_path: the path to save the preprocessed images
        @return: array of preprocessed images and array of labels
    """

    for dirpath, _, filenames in os.walk(dataset_path):
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

            print(f'Preprocessing image {file_path}...')
            image = preprocess_image(image)
            cv2.imwrite(os.path.join(dest_path, f'{file}'), image)
            # labels.append(int(file[0]))
            
    # return np.array(labels)