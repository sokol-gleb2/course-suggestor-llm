from ..utils.config import PREPROCESSING, DATA_PATHS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def sbert(courses, jobs):
    model = SentenceTransformer("all-mpnet-base-v2")

    courses_sbert = model.encode(courses['course_description'])
    jobs_sbert = model.encode(jobs['description'])
    return cosine_similarity(courses_sbert, jobs_sbert)

if __name__ == "__main__":
    jobs = pd.read_csv(DATA_PATHS["jobs_clean"])
    courses = pd.read_csv(DATA_PATHS["courses_clean"])

    sbert_mat = sbert(courses, jobs)
    print(sbert_mat)