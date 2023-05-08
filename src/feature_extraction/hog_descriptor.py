# this class is for extracting hog features from images, it's done manually


from pre_processing import pre_processing as pp
from utils import common_functions as cf
from utils import constants
import numpy as np
import cv2


class HogDescriptor:
    # resize the image to HOG_WIDHT and HOG_HEIGHT
    def resize_image(self, image):
        return cv2.resize(image, (constants.HOG_WIDHT, constants.HOG_HEIGHT))

    # calculate the gradient of the image
    def calculate_gradient(self, image):
        # calculate the gradient in x and y directions
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

        # calculate the magnitude and direction of the gradient
        magnitude, direction = cv2.cartToPolar(gx, gy)

        return magnitude, direction

    # calculate the histogram of the gradient for a block of the image (HOG_CELL_SIZE x HOG_CELL_SIZE)
    def calculate_histogram(self, magnitude, direction):
        # divide the image into blocks
        magnitude_blocks = cf.divide_image(magnitude, constants.HOG_CELL_SIZE)
        direction_blocks = cf.divide_image(direction, constants.HOG_CELL_SIZE)

        # calculate the histogram for each block
        histograms = []
        for i in range(magnitude_blocks.shape[0]):
            for j in range(magnitude_blocks.shape[1]):
                magnitude_block = magnitude_blocks[i, j]
                direction_block = direction_blocks[i, j]

                # calculate the histogram for the block
                histogram = self.calculate_histogram(
                    magnitude_block, direction_block)

                histograms.append(histogram)

        return histograms
