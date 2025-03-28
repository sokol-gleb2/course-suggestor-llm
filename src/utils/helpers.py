# Miscellaneous playground
from ..utils.config import DATA_PATHS
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Preprocess ---------------------------------------------------------------------------
    jobs = pd.read_csv(DATA_PATHS["jobs_clean"])
    courses = pd.read_csv(DATA_PATHS["courses_clean"])
    # courses = pd.read_csv(DATA_PATHS["courses_clean"], engine="python", on_bad_lines="warn")

    x = ["PPO Agent", "Random Baseline"]
    reward = [7.56, -1.85]
    course_counts = [3.21, 5.9]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.bar(x, reward, color='skyblue', label='Avg Reward')
    ax2.plot(x, course_counts, color='darkred', marker='o', label='Avg Courses Used')

    ax1.set_ylabel('Average Reward', color='blue')
    ax2.set_ylabel('Average Courses Used', color='darkred')
    plt.title('PPO Agent vs Random Baseline Performance')
    plt.grid(True)
    plt.show()


    # --------------------------------------------------------------------------------------