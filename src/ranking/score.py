from ..utils.config import PREPROCESSING, DATA_PATHS
from ..ranking.max_skill_overlap import calc_max_skill_overlap
from ..ranking.relevance import calculate_relevance
import pandas as pd

def calculate_score(max_skill_overlap, relevance, lamda = 0.5):
    """Calculates the overall score of each course compared to job"""
    return lamda * relevance + (1 - lamda) * max_skill_overlap

if __name__ == "__main__":
    courses = pd.read_csv(DATA_PATHS["courses_clean"])

    max_skill_overlap = calc_max_skill_overlap(courses)
    alpha = PREPROCESSING['alpha']
    beta = PREPROCESSING['beta']
    gamma = PREPROCESSING['gamma']
    delta = PREPROCESSING['delta']
    relevance = calculate_relevance(alpha, beta, gamma, delta, ner_mat, sbert_mat, tfidf_mat, meta_mat)

    calculate_score(max_skill_overlap, relevance, PREPROCESSING['lambda'])