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
        magnitude_cells = divide_image(magnitude, HOG_CELL_SIZE)
        direction_cells = divide_image(direction, HOG_CELL_SIZE)
        # magnitude blocks are now read as (numnbers are blocks)
        # [0, 1, 2, 3]
        # [4, 5, 6, 7]
        # [8, 9, 10, 11]
        # [12, 13, 14, 15]
        histograms = []
        for i in range(magnitude_cells.shape[0]):
            magnitude_cell = magnitude_cells[i]
            direction_cell = direction_cells[i]
            histogram = self.calculate_cell_histogram(
                magnitude_cell, direction_cell)
            histograms.append(histogram)
        return histograms

    def calculate_cell_histogram(self, magnitude_cell, direction_cell):
        # calculate the histogram for the block
        histogram = np.zeros((HOG_BIN_COUNT, 1))
        for i in range(magnitude_cell.shape[0]):
            for j in range(magnitude_cell.shape[1]):
                magnitude = magnitude_cell[i, j]
                direction = direction_cell[i, j]

                # calculate the histogram for the pixel
                histogram = self.calculate_pixel_histogram(
                    magnitude, direction, histogram)

        return histogram

    def calculate_pixel_histogram(self, magnitude, direction, histogram):
        # calculate the bin index
        bin_value = direction / (np.pi * 2 / HOG_BIN_COUNT)
        # get contribution of the two bins around the bin value
        bin_index_1 = int(bin_value) % (HOG_BIN_COUNT)
        bin_index_2 = (bin_index_1 + 1) % (HOG_BIN_COUNT)
        bin_contribution_1 = 1 - (bin_value - int(bin_value))
        bin_contribution_2 = 1 - bin_contribution_1
        # case bin_value = 9
        # bin_index_1 = 9 % 9 = 0
        # bin_index_2 = (0 + 1) % 9 = 1
        # bin_contribution_1 = 1 - (9 - 9) = 1
        # bin_contribution_2 = 1 - 1 = 0

        # case bin_value = 8.6
        # bin_index_1 = 8.6 % 9 = 8
        # bin_index_2 = (8 + 1) % 9 = 0
        # bin_contribution_1 = 1 - (8.6 - 8) = 0.4
        # bin_contribution_2 = 1 - 0.4 = 0.6

        # case bin_value = 0.1
        # bin_index_1 = 0.1 % 9 = 0
        # bin_index_2 = (0 + 1) % 9 = 1
        # bin_contribution_1 = 1 - (0.1 - 0) = 0.9
        # bin_contribution_2 = 1 - 0.9 = 0.1

        # add the contribution to the histogram
        histogram[bin_index_1] += magnitude * bin_contribution_1
        histogram[bin_index_2] += magnitude * bin_contribution_2

        return histogram

        # given blocks (probably 2x2) of histograms, concatenate them into one histogram (probably 1x36), and normalize it

    def normalize_blocks_histogram(self, blocks_histogram):
        # concatenate the histograms (3D array that came 2x2x9)
        histogram = np.concatenate(blocks_histogram, axis=0)
        # # normalize the
        # print("1 concat")
        # print(histogram)
        # # we nee to convert it to 1D array, since it's 2D array
        histogram = histogram.flatten()
        # print("2 concat")
        # print(histogram)
        if np.linalg.norm(histogram) == 0:
            return histogram

        histogram = histogram / np.linalg.norm(histogram)
        # print("after")
        # print(histogram)
        return histogram

    # function to extract the right 4 blocks histograms from the given list, then pass it to normalize_blocks_histogram function
    # total divisions would be 15x7 = 105, since the stride is 8
    def extract_feature_from_histogram(self, histograms):
        feature_vector = []
        histograms_2d = map_list_to_2D_nparray(
            histograms, HOG_WIDHT/HOG_CELL_SIZE)
        for i in range(0, int(HOG_HEIGHT/HOG_CELL_SIZE - 1), int(HOG_BLOCK_STRIDE/HOG_CELL_SIZE)):
            for j in range(0, int(HOG_WIDHT/HOG_CELL_SIZE - 1), int(HOG_BLOCK_STRIDE/HOG_CELL_SIZE)):
                block_histograms = histograms_2d[i:i+2, j:j+2]
                # print("main (no concat)")
                # print(block_histograms)
                feature_vector += self.normalize_blocks_histogram(
                    block_histograms).tolist()

        # make it 1D np.array
        feature_vector = np.array(feature_vector).flatten()
        return feature_vector

##############################################################################################################
##############################################################################################################
#####################################   U S E   T H I S   O N L Y   ##########################################
##############################################################################################################
##############################################################################################################

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

    def extract_features(self, images):
        features = []
        for image in images:
            current_image = image.copy()
            current_image = any2gray(current_image)
            current_image = change_gray_range(current_image, 255)
            resized_image = self.resize_image(current_image)
            magnitude, direction = self.calculate_gradient(resized_image)
            histogram = self.calculate_histogram(magnitude, direction)
            feature = self.extract_feature_from_histogram(histogram)
            features.append(feature)

        return features

    def error_calculation(self, features_manual: list, features_builtin: list) -> float:
        """
            Calculates the error between the two features
            @param features_manual: the features calculated manually
            @param features_builtin: the features calculated using the built-in function
            @return: the error between the two features
        """

        error = 0
        for i in range(len(features_manual)):
            error += np.sum(np.abs(features_manual[i] -
                                   features_builtin[i]))
        error = error / (len(features_manual) *
                         len(features_manual[0]))

        return error
