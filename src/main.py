import time
import cv2
from pre_processing import pre_processing as pp
from model_training import *
from feature_extraction import ef_descriptor as efd
from feature_extraction import hog_descriptor as hog
import os
import numpy as np
from utils.common_functions import read_images, change_gray_range


def classify(image):
    pp_image = pp.preprocess_image(image)
    efd_features = efd.elliptical_fourier(pp_image)
    # TODO: HOG descriptor for one image
   
    # hog_efd_features_custom = np.concatenate((hog_features_custom, efd_features), axis=1)
    # hog_efd_features_builtin = np.concatenate((hog_features_builtin, efd_features), axis=1)
    
    # TODO: Classify the image using the trained model
    # TODO: Return the image class

def run_pipline(src_path: str, dest_path: str):
    results = []
    times = []

    for dirpath, _, filenames in os.walk(src_path):
        if not filenames:
            continue

        for file in filenames:
            if not file.endswith('.jpg') and not file.endswith('.JPG'):
                print(f'File {file} is not a jpg file. Skipping...')
                continue

            file_path = os.path.join(dirpath, file)

            image = cv2.imread(file_path)
            if image is None:
                print(f'File {file} is not a valid image. Skipping...')
                continue
            
            print(f'Reading image {file_path}...')

            start_time = time.time()
            image_class = classify(image)
            end_time = time.time()
            
            print(f'Image {file_path} is classified as {image_class} in {end_time - start_time} seconds.')
            results.append(image_class)
            times.append(end_time - start_time)
    
    # TODO: write results and times to 2 files
            


def preprocessing():
    # PREPROCESSING
    pp.run_preprocessor(
        dataset_path = os.path.join(os.path.dirname(__file__), '../dataset'),
        dest_path = os.path.join(os.path.dirname(__file__), '../pp_dataset')
    )
    # np.save(os.path.join(os.path.dirname(__file__), '../features/labels.npy'), labels)

def feature_extraction():
    # FEATURE EXTRACTION [EFD]
    efd_features, efd_labels = efd.run_elliptical_fourier(
        pp_dataset_path = os.path.join(os.path.dirname(__file__), '../pp_dataset')
    )
    # append labels to the features at the first column
    np.save(os.path.join(os.path.dirname(__file__), '../features/efd_features.npy'), efd_features)
    np.save(os.path.join(os.path.dirname(__file__), '../features/efd_labels.npy'), efd_labels)


    pp_images, labels = read_images(os.path.join(os.path.dirname(__file__), '../pp_dataset'))
    np.save(os.path.join(os.path.dirname(__file__), '../features/labels.npy'), labels)

    # FEATURE EXTRACTION [HOG_BUILTIN]
    hog_descriptor = hog.HogDescriptor()
    hog_features_builtin = hog_descriptor.builtin_hog_descriptor(pp_images)
    np.save(os.path.join(os.path.dirname(__file__), '../features/hog_features_builtin.npy'), hog_features_builtin)


    # FEATURE EXTRACTION [HOG_CUSTOM]
    hog_features_custom = hog_descriptor.extract_features(pp_images)
    np.save(os.path.join(os.path.dirname(__file__), '../features/hog_features_custom.npy'), hog_features_custom)

    # FEATURE EXTRACTION [HOG + EFD]
    hog_efd_features_builtin = np.concatenate((hog_features_builtin, efd_features), axis=1)
    hog_efd_features_custom = np.concatenate((hog_features_custom, efd_features), axis=1)

    assert(efd_features.shape[0] == hog_features_builtin.shape[0] == hog_efd_features_builtin.shape[0])
    assert(efd_features.shape[1] + hog_features_builtin.shape[1] == hog_efd_features_builtin.shape[1])

    np.save(os.path.join(os.path.dirname(__file__), '../features/hog_efd_features_builtin.npy'), hog_efd_features_builtin)
    np.save(os.path.join(os.path.dirname(__file__), '../features/hog_efd_features_custom.npy'), hog_efd_features_custom)




if __name__ == "__main__":
    ## TRAINING PHASE

    # preprocessing()
    feature_extraction()


    ## TESTING PHASE

    # run_pipline(
    #     src_path = os.path.join(os.path.dirname(__file__), '../test_dataset'),
    #     dest_path = os.path.join(os.path.dirname(__file__), '../test_output')
    # )