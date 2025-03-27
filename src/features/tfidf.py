from ..utils.config import PREPROCESSING, DATA_PATHS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf(courses, jobs):
    corpus = pd.concat([jobs['description'], courses['course_description']], ignore_index=True)
    
    vectoriser = TfidfVectorizer()
    tfidf_mat = vectoriser.fit_transform(corpus)

    # Split the TF-IDF matrix into job and course matrices
    num_jobs = len(jobs['description'])

    tfidf_jobs = tfidf_mat[:num_jobs]    # First 'num_jobs' rows are job descriptions
    tfidf_courses = tfidf_mat[num_jobs:] # Remaining rows are course descriptions

    # Compute cosine similarity between jobs and courses
    tfidf_similarity_matrix = cosine_similarity(tfidf_jobs, tfidf_courses)  # Shape: [num_jobs x num_courses]

    return tfidf_similarity_matrix

if __name__ == "__main__":
    jobs = pd.read_csv(DATA_PATHS["jobs_clean"])
    courses = pd.read_csv(DATA_PATHS["courses_clean"])

    tfidf_mat = tfidf(courses, jobs)
    print(tfidf_mat)

    