import pandas as pd
from ..utils.config import PREPROCESSING, DATA_PATHS


def load_data():
    """Loads raw course and job datasets."""
    # Load Data
    # jobs_1 = pd.read_csv(DATA_PATHS["jobs_1_raw"])
    # # Define the list of location keywords
    # location_keywords = [' EUR..', ' EUROPE', 'EUROPE', ' EUROPEA..', ' EUROPEAN..', ' UK', 'EUROPEAN TIMEZO..', 'GERMANY', 'SPAIN', 'UK', 'WORLDWIDE']
    # # Create a boolean mask based on whether each row contains any of the specified keywords
    # mask = jobs_1['Location'].str.contains('|'.join(location_keywords), na=False, case=False)
    # # Filter the DataFrame using the boolean mask
    # jobs_1 = jobs_1[mask]

    # JOBS
    jobs = pd.read_csv(DATA_PATHS["jobs_raw"])

    # Courses 1
    courses_1 = pd.read_csv(DATA_PATHS["courses_1_raw"])

    # Courses 2
    courses_2 = pd.read_csv(DATA_PATHS["courses_2_raw"])

    return courses_1, courses_2, jobs

if __name__ == "__main__":
    courses_1, courses_2, jobs = load_data()
    print(courses_2['course_time'].unique())