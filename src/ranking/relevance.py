from ..utils.config import PREPROCESSING, DATA_PATHS
import pandas as pd
import numpy as np

def calculate_relevance(alpha, beta, gamma, delta, ner_mat, sbert_mat, tfidf_mat, meta_mat):
    """Calculates relevance for tunes alpha beta gamma delta"""
    return alpha * sbert_mat + beta * tfidf_mat + gamma * ner_mat + delta * meta_mat

if __name__ == "__main__":
    # Preprocess ---------------------------------------------------------------------------
    jobs = pd.read_csv(DATA_PATHS["jobs_clean"])
    courses = pd.read_csv(DATA_PATHS["courses_clean"])
    # --------------------------------------------------------------------------------------

    # NER ---------------------------------------------------------------------------------
    from ..features.ner import full_ner
    ner_mat = full_ner(courses, jobs)
    # --------------------------------------------------------------------------------------

    # SBERT --------------------------------------------------------------------------------
    from ..features.sbert import sbert
    sbert_mat = sbert(courses, jobs)
    # --------------------------------------------------------------------------------------

    # TFIDF --------------------------------------------------------------------------------
    from ..features.tfidf import tfidf
    tfidf_mat = tfidf(courses, jobs)
    # --------------------------------------------------------------------------------------

    # META ---------------------------------------------------------------------------------
    from ..features.metadata_encoding import metadata_encoding
    meta_mat = metadata_encoding(courses, jobs)
    # --------------------------------------------------------------------------------------

    # Relevance ----------------------------------------------------------------------------
    alpha = PREPROCESSING['alpha']
    beta = PREPROCESSING['beta']
    gamma = PREPROCESSING['gamma']
    delta = PREPROCESSING['delta']
    relvance_mat = calculate_relevance(alpha, beta, gamma, delta, ner_mat, sbert_mat, tfidf_mat, meta_mat)

    # Save
    np.save(DATA_PATHS['relevance_matrix'], relvance_mat)
    # --------------------------------------------------------------------------------------

    pass