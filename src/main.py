from pre_processing import pre_processing as pp
from model_training import *
from feature_extraction import ef_descriptor as efd
from feature_extraction import hog_descriptor as hog
import os
import numpy as np
from utils.common_functions import read_images, change_gray_range


# Run the whole pipeline to a single image
def classify(image):
    pp_image = pp.preprocess_image(image)
    efd_features = efd.elliptical_fourier(pp_image)
    hog_descriptor = hog.HogDescriptor()
    hog_features_custom = hog_descriptor.extract_features_from_image(pp_image)
    # hog_features_builtin = hog_descriptor.builtin_hog_descriptor(pp_image)
    hog_efd_features_custom = np.concatenate((hog_features_custom, efd_features), axis=1)
    # hog_efd_features_builtin = np.concatenate((hog_features_builtin, efd_features), axis=1)
    
    ## TODO: Classify the image using the trained model


def preprocessing():
    # PREPROCESSING
    labels = pp.run_preprocessor(
        dataset_path = os.path.join(os.path.dirname(__file__), '../dataset'),
        dest_path = os.path.join(os.path.dirname(__file__), '../pp_dataset')
    )
    np.save(os.path.join(os.path.dirname(__file__), '../features/labels.npy'), labels)

def feature_extraction():
    # FEATURE EXTRACTION [EFD]
    efd_features = efd.run_elliptical_fourier(
        pp_dataset_path = os.path.join(os.path.dirname(__file__), '../pp_dataset')
    )
    np.save(os.path.join(os.path.dirname(__file__), '../features/efd_features.npy'), efd_features)



    pp_images, labels = read_images(os.path.join(os.path.dirname(__file__), '../pp_dataset'))
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


    # efd_features = np.load(os.path.join(os.path.dirname(__file__), '../features/efd_features.npy'))


if __name__ == "__main__":
    preprocessing()
    feature_extraction()
