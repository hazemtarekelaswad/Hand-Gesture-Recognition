from typing import *
import numpy as np
from skimage import io
from skimage.transform import rescale, resize
import os
import matplotlib.pyplot as plt
from utils.common_functions import *
import cv2
from scipy import ndimage
from utils.common_functions import read_images

def preprocess_image(image: np.ndarray) -> np.ndarray:

    resize_ratio = 0.1

    image = cv2.resize(image, (int(image.shape[1] * resize_ratio), int(image.shape[0] * resize_ratio)))
    # show_images([image])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    # show_images([image])

    lower_bound = np.array([0, 133, 77])
    upper_bound = np.array([255, 173, 127])
    image = cv2.inRange(image, lower_bound, upper_bound)
    # show_images([image])

    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=5)
    # show_images([image])

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    # show_images([image])

    image = ndimage.binary_fill_holes(image).astype(np.int8)
    image = change_gray_range(image, format=255)
    # show_images([image])

    # image = any2gray(image)
    # image = change_gray_range(image, format=255)
    # image = change_gray_range(rescale(image, 0.1, anti_aliasing=True))
    # image = convert_to_binary(image, threshold=140)
    return image


#* Should be called in main
def run_preprocessor(dataset_path: str, dest_path: str):
    images, labels = read_images(dataset_path)
    # preprocess all images
    for i in range(len(images)):
        images[i] = preprocess_image(images[i].copy())
        cv2.imwrite(os.path.join(dest_path, f'{labels[i]}_{i}.JPG'), images[i], )