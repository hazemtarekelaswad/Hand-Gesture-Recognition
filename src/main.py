from pre_processing import pre_processing as pp
from model_training import *
from feature_extraction import ef_descriptor as efd
from feature_extraction import hog_descriptor as hog
import os
import numpy as np

def main():
    # PREPROCESSING
    pp_images, labels = pp.run_preprocessor(
        dataset_path = os.path.join(os.path.dirname(__file__), '../dataset_sample'),
        dest_path = os.path.join(os.path.dirname(__file__), '../pp_dataset')
    )

    # FEATURE EXTRACTION [EFD]
    efd_features, labels = efd.run_elliptical_fourier(
        pp_dataset_path = os.path.join(os.path.dirname(__file__), '../pp_dataset'),
        dest_path = os.path.join(os.path.dirname(__file__), '../features')
    )

    # FEATURE EXTRACTION [HOG]
    hog_descriptor = hog.HogDescriptor()
    hog_features_builtin = hog_descriptor.builtin_hog_descriptor(pp_images)
    hog_features_custom = hog_descriptor.extract_features(pp_images)

    np.save(os.path.join(os.path.dirname(__file__), '../features/hog_features_builtin.npy'), hog_features_builtin)
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
    main()