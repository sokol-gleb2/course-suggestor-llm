from ..utils.config import PREPROCESSING, DATA_PATHS
import pandas as pd
import numpy as np
import spacy
from spacy import displacy

def full_ner(courses_df, jobs_df):
    """Calculates the full Jaccard similarity of NER of courses"""
    ner_matrix = np.zeros(shape=(jobs_df.shape[0], courses_df.shape[0]))
    # ner_matrix = np.zeros(shape=(10, 10))

    NER = spacy.load("en_core_web_sm")

    # for (c, course) in enumerate(courses_df['course_description'][10:20]):
    #     print(course)
    #     course_ner = set(ent.text for ent in NER(course).ents)
    #     print("course: ")
    #     print(course_ner)
    #     for (j,job) in enumerate(jobs_df['description'][0:10]):
    #         job_ner = set(ent.text for ent in NER(job).ents)
    #         print("job: ")
    #         print(job_ner)
    #         intersect = len(course_ner & job_ner)
    #         union = len(course_ner | job_ner)
    #         ner_matrix[j, c] = intersect / (union + 1e-9)
    for (c, course) in enumerate(courses_df['course_description']):
        course_ner = set(ent.text for ent in NER(course).ents)
        for (j,job) in enumerate(jobs_df['description']):
            job_ner = set(ent.text for ent in NER(job).ents)
            intersect = len(course_ner & job_ner)
            union = len(course_ner | job_ner)
            ner_matrix[j, c] = intersect / (union + 1e-9)
    
    return ner_matrix


if __name__ == "__main__":
    jobs_df = pd.read_csv(DATA_PATHS["jobs_clean"])
    # courses_df = pd.read_csv(DATA_PATHS["courses_clean"])
    courses_df = pd.read_csv(DATA_PATHS["courses_clean"], encoding="utf-8", engine="python")


    print("Full NER:")
    print(full_ner(courses_df, jobs_df))

    