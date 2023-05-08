# this class is for extracting hog features from images, it's done manually


from pre_processing import pre_processing as pp
from utils.common_functions import *
from utils.constants import *
import numpy as np
import cv2


class HogDescriptor:
    # resize the image to HOG_WIDHT and HOG_HEIGHT
    def resize_image(self, image):
        return cv2.resize(image, (HOG_WIDHT, HOG_HEIGHT))

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
        magnitude_blocks = divide_image(magnitude, HOG_CELL_SIZE)
        direction_blocks = divide_image(direction, HOG_CELL_SIZE)

        # calculate the histogram for each block
        histograms = []
        for i in range(magnitude_blocks.shape[0]):
            for j in range(magnitude_blocks.shape[1]):
                magnitude_block = magnitude_blocks[i, j]
                direction_block = direction_blocks[i, j]

                # calculate the histogram for the block
                histogram = self.calculate_block_histogram(
                    magnitude_block, direction_block)

                histograms.append(histogram)

        return histograms

    def calculate_block_histogram(self, magnitude_block, direction_block):
        # calculate the histogram for the block
        histogram = np.zeros((HOG_BIN_COUNT, 1))
        for i in range(magnitude_block.shape[0]):
            for j in range(magnitude_block.shape[1]):
                magnitude = magnitude_block[i, j]
                direction = direction_block[i, j]

                # calculate the histogram for the pixel
                histogram = self.calculate_pixel_histogram(
                    magnitude, direction, histogram)

        return histogram

    def calculate_pixel_histogram(self, magnitude, direction, histogram):
        # calculate the bin index
        bin_index = int(direction / (360 / HOG_BIN_COUNT))

        # add the magnitude to the bin
        histogram[bin_index] += magnitude

        return histogram
