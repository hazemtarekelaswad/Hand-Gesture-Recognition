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
        print(magnitude_blocks.shape[0])

        # calculate the histogram for each block
        histograms = []
        for i in range(magnitude_blocks.shape[0]):
            magnitude_block = magnitude_blocks[i]
            direction_block = direction_blocks[i]
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

    def builtin_hog_descriptor(self, images):
        """
        Extracts the HOG features of the images using OpenCV built-in function
        @param images: the images (grayscaled for evaluation)
        @return: the HOG features of that image

        Example:
        hog_descriptor = HogDescriptor()
        features = hog_descriptor.builtin_hog_descriptor(
            images) # images is a list of images

        """
        hog_features = []
        win_size = (HOG_WIDHT, HOG_HEIGHT)
        block_size = (HOG_BLOCK_SIZE, HOG_BLOCK_SIZE)
        block_stride = (HOG_CELL_SIZE, HOG_CELL_SIZE)
        cell_size = (HOG_CELL_SIZE, HOG_CELL_SIZE)
        nbins = HOG_BIN_COUNT
        cv2_hog = cv2.HOGDescriptor(win_size, block_size,
                                    block_stride, cell_size, nbins)
        for image in images:
            current_image = image.copy()
            current_image = self.resize_image(current_image)
            hog_feature = cv2_hog.compute(current_image)
            hog_features.append(hog_feature)

        return hog_features

        # # extract hog features to have 3780 features per image
        # win_size = (128, 64)
        # block_size = (16, 16)
        # block_stride = (8, 8)
        # cell_size = (8, 8)
        # nbins = 9
        # img = images[0].copy()
        # # img = change_gray_range(any2gray(images[0]), 255)
        # img = cv2.resize(img, (128, 64))
        # print(img.shape)
        # hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        # feature = hog.compute(img)
        # print(feature.shape)

        # # show image
        # show_images([img], ['image'])
