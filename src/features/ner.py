from ..utils.config import PREPROCESSING, DATA_PATHS
import pandas as pd
import numpy as np
import spacy
from spacy import displacy

def full_ner(courses_df, jobs_df):
    """Calculates the full Jaccard similarity of NER of courses"""
    ner_matrix = np.zeros(shape=(jobs_df.shape[0], courses_df.shape[0]))

    NER = spacy.load("en_core_web_sm")

    for (c, course) in enumerate(courses_df['description']):
        course_ner = NER(course)
        for (j,job) in enumerate(jobs_df['description']):
            job_ner = NER(job)
            intersect = len(list(set(course_ner) & set(job_ner)))
            union = len(list(set(course_ner + job_ner)))
            ner_matrix[j, c] = intersect / (union + 1e-9)
    
    return ner_matrix


if __name__ == "__main__":
    jobs_df = pd.read_csv(DATA_PATHS["jobs_clean"])
    courses_df = pd.read_csv(DATA_PATHS["courses_clean"])

    print("Full NER:")
    print(full_ner(courses_df, jobs_df))

    