import time
import cv2
from pre_processing import pre_processing as pp
from model_training import *
from feature_extraction import ef_descriptor as efd
from feature_extraction import hog_descriptor as hog
import os
import numpy as np
from utils.common_functions import read_images, show_images
import sys
from natsort import natsorted


'''
Takes an image and classifies it using the trained model
@param image: the image to be classified
@return: the class of the image, the duration of the classification
'''
def classify(image, model):
    start_time = time.time()

    pp_image = pp.preprocess_image(image)

    efd_features = efd.elliptical_fourier(pp_image, 10)

    hog_descriptor = hog.HogDescriptor()
    hog_features_builtin = hog_descriptor.builtin_hog_descriptor(pp_image)
    hog_features_custom = hog_descriptor.extract_features(pp_image)
    
    efd_features = efd_features.reshape(1, -1)
    hog_features_builtin = hog_features_builtin.reshape(1, -1)
    hog_features_custom = hog_features_custom.reshape(1, -1)

    hog_efd_features_builtin = np.concatenate((hog_features_builtin, efd_features), axis=1)
    hog_efd_features_custom = np.concatenate((hog_features_custom, efd_features), axis=1)
    
    # TODO: Classify the image using the trained model
    # pred_label = model.predict(hog_efd_features_builtin)
    
    end_time = time.time()

    # TODO: Return the image class
    # return pred_label, round(end_time - start_time, 3)
    return 0, round(end_time - start_time, 3)

def run_pipline(src_path: str, dest_path: str):
    results = []
    times = []
    # TODO: Load the model
    model = None

    # read images from src_path
    for file in natsorted(os.listdir(src_path)):
        if not file.endswith('.jpg') and not file.endswith('.JPG') and not file.endswith('.png'):
            print(f'File {file} is not a jpg or png file. Skipping...')
            continue
            
        file_path = os.path.join(src_path, file)

        image = cv2.imread(file_path)
        if image is None:
            print(f'File {file} is not a valid image. Skipping...')
            continue
        
        image_class, duration = classify(image, model)
        
        print(f'Image {file} is classified as {image_class} in {duration} seconds.')

        results.append(image_class)
        times.append(duration)

    # write results to dest_path
    results_path = os.path.join(dest_path, 'results.txt')
    with open(results_path, 'w') as f:
        for result in results:
            f.write(f'{result}\n')
    
    # write times to dest_path
    times_path = os.path.join(dest_path, 'time.txt')
    with open(times_path, 'w') as f:
        for t in times:
            f.write(f'{t}\n')


def preprocessing():
    # PREPROCESSING
    pp.run_preprocessor(
        dataset_path = os.path.join(os.path.dirname(__file__), '../dataset'),
        dest_path = os.path.join(os.path.dirname(__file__), '../pp_dataset')
    )

def feature_extraction():

    # READ IMAGES and SAVE LABELS
    pp_images, labels = read_images(os.path.join(os.path.dirname(__file__), '../pp_dataset'))
    np.save(os.path.join(os.path.dirname(__file__), '../features/labels.npy'), labels)

    # FEATURE EXTRACTION [EFD]
    efd_features = efd.extract_features(pp_images, 10)
    np.save(os.path.join(os.path.dirname(__file__), '../features/efd_features.npy'), efd_features)

    # FEATURE EXTRACTION [HOG_BUILTIN]
    hog_descriptor = hog.HogDescriptor()
    hog_features_builtin = hog_descriptor.builtin_hog_descriptor_all(pp_images)
    np.save(os.path.join(os.path.dirname(__file__), '../features/hog_features_builtin.npy'), hog_features_builtin)


    # FEATURE EXTRACTION [HOG_CUSTOM]
    hog_features_custom = hog_descriptor.extract_features_all(pp_images)
    np.save(os.path.join(os.path.dirname(__file__), '../features/hog_features_custom.npy'), hog_features_custom)

    # FEATURE EXTRACTION [HOG + EFD] built-in and custom
    hog_efd_features_builtin = np.concatenate((hog_features_builtin, efd_features), axis=1)
    hog_efd_features_custom = np.concatenate((hog_features_custom, efd_features), axis=1)

    assert(efd_features.shape[0] == hog_features_builtin.shape[0] == hog_efd_features_builtin.shape[0])
    assert(efd_features.shape[1] + hog_features_builtin.shape[1] == hog_efd_features_builtin.shape[1])

    np.save(os.path.join(os.path.dirname(__file__), '../features/hog_efd_features_builtin.npy'), hog_efd_features_builtin)
    np.save(os.path.join(os.path.dirname(__file__), '../features/hog_efd_features_custom.npy'), hog_efd_features_custom)




if __name__ == "__main__":
    ## TRAINING PHASE

    # preprocessing()
    # feature_extraction()

    
    ## TESTING PHASE
    # argv[1]: Relative path to the test dataset [Default: ../data]
    # argv[2]: Relative path to the output folder [Default: ../results]

    if len(sys.argv) > 3:
        print('Too many arguments. Expected 2 arguments.')
        exit(1)

    if len(sys.argv) == 1:
        sys.argv.append('../data')
        sys.argv.append('../results')
    elif len(sys.argv) == 2:
        sys.argv.append('../results')

    if not os.path.exists(os.path.join(os.path.dirname(__file__), sys.argv[1])):
        print(f'Dataset path "{sys.argv[1]}" does not exist.')
        exit(1)
    
    if not os.path.exists(os.path.join(os.path.dirname(__file__), sys.argv[2])):
        print(f'Output path "{sys.argv[2]}" does not exist.')
        exit(1)


    run_pipline(
        src_path = os.path.join(os.path.dirname(__file__), sys.argv[1]),
        dest_path = os.path.join(os.path.dirname(__file__), sys.argv[2])
    )