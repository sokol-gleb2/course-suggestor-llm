# from vertexai.preview.language_models import ChatModel, InputOutputTextPair

# pip install -U google-genai

# Run this before running the script:
# CHANGE THE API KEY
# export GOOGLE_CLOUD_PROJECT=your-project-id
# export GOOGLE_CLOUD_LOCATION=us-central1
# export GOOGLE_GENAI_USE_VERTEXAI=True
# export GOOGLE_API_KEY="YOUR_KEY_HERE"
from secret.secret import GC_KEY
from ..utils.config import PREPROCESSING, DATA_PATHS
from ..ranking.score import run_ga_for_job
import pandas as pd
from google import genai
from google.genai import types
import json
import random
import numpy as np
import re


client = genai.Client(api_key=GC_KEY)

def format_prompt(job_description, course_list):
    course_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(course_list))
    prompt = f"""
        You're an expert career advisor.

        A person wants to get the following job:
        "{job_description}"

        They are considering taking the following courses:
        {course_str}

        Would these courses likely be sufficient to get them hired?
        Respond with one of the following exactly:
        - Job Secured
        - Jobless
    """
    return prompt.strip()

def query_gemini_for_judgement(job_description, course_list):
    prompt = format_prompt(job_description, course_list)
    response = client.models.generate_content(
        model="gemini-1.5-pro", contents=prompt
    )
    output = response.text.strip()

    if "secured" in output.lower():
        return 1  # Job Secured
    elif "jobless" in output.lower():
        return 0  # Jobless
    else:
        return -1  # Invalid / ambiguous output

def pick_course_selection(job):
    if random.random() <= 0.7:
        # Calculate good suggestions
        courses_indices = run_ga_for_job(job) # these are indices
        courses_to_pass = [courses.iloc[i] for i in courses_indices]
        pass
    else:
        # Get random (bad) suggestions
        courses_to_pass = courses.sample(n=random.randrange(5, 10, 1), random_state=42).to_numpy()
    
    return courses_to_pass


def parse_gemini_courses(response_text):
    # Converting Gemini's list into Python readable format
    course_list = []
    pattern = r"\d+\.\s*(.*?):\s*(.*)"

    for line in response_text.strip().split("\n"):
        match = re.match(pattern, line)
        if match:
            title = match.group(1).strip()
            desc = match.group(2).strip()
            course_list.append((title, desc))

    return course_list


def query_gemini_for_courses(job):
    # This is for evaluating our model
    prompt = f"""
        You are a career advisor and online course recommender.

        A user wants to become a "{job}".

        Please recommend 5-10 online courses they should take to prepare for this job. For each course, include:

        - A clear course title
        - A one-line description of what it teaches

        Respond in the following format:
        1. <Course Title>: <One-line Description>
        2. ...

    """
    response = client.models.generate_content(
        model="gemini-1.5-pro", contents=prompt
    )
    output = response.text.strip()

    return parse_gemini_courses(output)


if __name__ == "__main__":
    jobs = pd.read_csv(DATA_PATHS["jobs_clean"])
    courses = pd.read_csv(DATA_PATHS["courses_clean"])

    score_mat = np.load(DATA_PATHS['score_matrix'])
    
    courseList_job_judge_scores = []
    for j in range(len(jobs)):
        job = jobs.iloc[j]
        courses = []
        courses_picked = pick_course_selection(j)
        for c in courses_picked:
            courses.append(f""" Course title: {c["title"]}; Course description: {c["description"]}; """)
        courseList_job_judge_scores.append({
            "job": job,
            "course_list": courses,
            "label": query_gemini_for_judgement(f"""Job Title: {job["title"]}; Job Description: {job["description"]}""")
        })
        with open(DATA_PATHS["courseList_job_judge_scores"], 'w') as f:
            json.dump(courseList_job_judge_scores, f, indent=4)
