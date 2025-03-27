from ..utils.config import DATA_PATHS
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import ace_tools as tools
import numpy as np


def calc_max_skill_overlap(courses):
    """Calculate Jaccard of one-hot encoding for all course descriptions"""
    # Step 1: Convert to One-Hot Encoding
    vectorizer = CountVectorizer(binary=True)  # Binary = One-Hot Encoding
    one_hot_matrix = vectorizer.fit_transform(courses['course_description']).toarray()

    # Step 2: Compute Jaccard Similarity
    # Jaccard = (Intersection / Union)
    jaccard_similarity_matrix = 1 - pairwise_distances(one_hot_matrix, metric="jaccard")

    return jaccard_similarity_matrix

if __name__ == "__main__":

    courses = pd.read_csv(DATA_PATHS["courses_clean"])

    jaccard_similarity_matrix = calc_max_skill_overlap(courses)
    np.save(DATA_PATHS['max_skill_overlap_matrix'], jaccard_similarity_matrix)

    # Convert to DataFrame for better visualization
    df = pd.DataFrame(jaccard_similarity_matrix, 
                    index=[f"Course {i+1}" for i in range(len(courses['course_description']))], 
                    columns=[f"Course {i+1}" for i in range(len(courses['course_description']))])

    # Display matrix
    tools.display_dataframe_to_user(name="Jaccard Similarity Matrix", dataframe=df)