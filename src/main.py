from pre_processing import pre_processing as pp
from model_training import *
from feature_extraction import *
import os

def main():
    pp.run_preprocessor(
        dataset_path = os.path.join(os.path.dirname(__file__), '../dataset_sample'),
        dest_path = os.path.join(os.path.dirname(__file__), '../pp_dataset')
    )

if __name__ == "__main__":
    main()