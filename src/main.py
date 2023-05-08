from pre_processing import pre_processing as pp
from model_training import *
from feature_extraction import ef_descriptor as efd
import os
import numpy as np

def main():
    pp.run_preprocessor(
        dataset_path = os.path.join(os.path.dirname(__file__), '../dataset_sample'),
        dest_path = os.path.join(os.path.dirname(__file__), '../pp_dataset')
    )

    efd.run_elliptical_fourier(
        pp_dataset_path = os.path.join(os.path.dirname(__file__), '../pp_dataset'),
        dest_path = os.path.join(os.path.dirname(__file__), '../features')
    )

    features = np.load(os.path.join(os.path.dirname(__file__), '../features/efd_features.npy'))
    print(features)

if __name__ == "__main__":
    main()