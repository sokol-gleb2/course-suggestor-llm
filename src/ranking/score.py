from ..utils.config import PREPROCESSING, DATA_PATHS
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random

def relevance(j, c):
    return relevance_matrix[j, c]

def MSO(i, j):
    return max_skill_overlap[i, j]

# ===== Objective function =====
def make_evaluator(job_index):
    def evaluate(individual):
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]

        if not selected_indices:
            return -np.inf,

        # Relevance score
        rel_score = relevance(job_index, selected_indices).sum()

        # MSO score (excluding self-comparisons)
        mso_score = sum(
            MSO(i, j) for i in selected_indices for j in selected_indices if i != j
        )
        mso_score = mso_score if mso_score != 0 else 1e-6  # Avoid div-by-zero

        score = rel_score + 1.0 / mso_score
        return score,
    return evaluate

# ===== GA setup =====
def setup_ga(job_index):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: random.randint(0, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(courses))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", make_evaluator(job_index))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

# ===== Run GA =====
def run_ga_for_job(job_index, n_gen=50, pop_size=100):
    toolbox = setup_ga(job_index)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen,
                        stats=stats, halloffame=hof, verbose=True)
    
    best_indices = [i for i, bit in enumerate(hof[0]) if bit == 1]
    best_score = hof[0].fitness.values[0]
    return best_indices, best_score




if __name__ == "__main__":
    courses = pd.read_csv(DATA_PATHS['courses_clean'])
    max_skill_overlap = np.load(DATA_PATHS['max_skill_overlap_matrix'])
    relevance_matrix = np.load(DATA_PATHS['relevance_matrix'])

    job_idx = 0  # or any job index you're interested in
    best_courses, score = run_ga_for_job(job_idx)
    print("Selected course indices:", best_courses)
    print("Objective score:", score)