# Hyperparameters, file paths
# config.py

DATA_PATHS = {
    "courses_1_raw": "data/raw/courses_1_raw.csv",
    "courses_2_raw": "data/raw/courses_2_raw.csv",
    "jobs_raw": "data/raw/jobs_raw.csv",
    "courses_clean": "data/processed/courses_clean.csv",
    "jobs_clean": "data/processed/jobs_clean.csv",
    "courseList_job_judge_scores": "data/scores/courseList_job_judge_scores.json",
    "relevance_matrix": "data/matrices/relevance.npy",
    "max_skill_overlap_matrix": "data/matrices/max_skill_overlap.npy",
    "score_matrix": "data/matrices/score.npy",
}

# Preprocessing parameters
PREPROCESSING = {
    "alpha": 1,
    "beta": 1,
    "gamma": 1,
    "delta": 1,
    "lambda": 0.5
}
