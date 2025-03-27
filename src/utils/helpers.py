# Miscellaneous utilities
from ..utils.config import DATA_PATHS
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Preprocess ---------------------------------------------------------------------------
    jobs = pd.read_csv(DATA_PATHS["jobs_clean"])
    courses = pd.read_csv(DATA_PATHS["courses_clean"])
    # courses = pd.read_csv(DATA_PATHS["courses_clean"], engine="python", on_bad_lines="warn")
    c = courses.sample(n=10, random_state=42)
    print(type(c))
    # --------------------------------------------------------------------------------------