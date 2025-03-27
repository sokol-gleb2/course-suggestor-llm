from ..utils.config import PREPROCESSING, DATA_PATHS
import pandas as pd


def metadata_encoding(courses):
    # duration: numerical OR should it be part of max skill overlap
    # difficulty: one-hot
    # rating: numerical
    pass

if __name__ == "__main__":
    courses = pd.read_csv(DATA_PATHS["courses_clean"])

    meta_mat = metadata_encoding(courses)
    print(meta_mat)