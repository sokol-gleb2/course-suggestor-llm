from ..utils.config import PREPROCESSING, DATA_PATHS
import pandas as pd
import csv

config = {
    "w1": 0.6,
    "w2": 0.3,
    "w3": 0.1
}

difficulty_map = {
    "Beginner": 1.0,
    "Intermediate": 0.6,
    "Advanced": 0.3,
}

def duration_score(h):
    # as in the formula
    if h < 3 or h > 40:
        return 0.3
    elif h >= 5 and h <= 20:
        return 1.0
    else:
        return 0.7


def metadata_encoding(courses):
    # duration: numerical OR should it be part of max skill overlap
    # difficulty: one-hot
    # rating: numerical
    # as in the formula
    courses["meta_score"] = 0
    for idx, course in courses.iterrows():
        review_tranformed = course["course_rating"] / 5.0
        meta_score = config["w1"] * review_tranformed + config["w2"] * difficulty_map[course["course_difficulty"]] + config["w3"] * duration_score(course["course_time"])
        courses.at[idx, 'meta_score'] = meta_score

    return courses

if __name__ == "__main__":
    courses = pd.read_csv(DATA_PATHS["courses_clean"])

    meta_courses = metadata_encoding(courses)
    meta_courses.to_csv(DATA_PATHS["courses_clean"], index=False, quoting=csv.QUOTE_ALL)