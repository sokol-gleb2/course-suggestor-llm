from ..utils.config import PREPROCESSING, DATA_PATHS
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def clean_text(text):
    """Basic text preprocessing (lowercasing, removing special chars, non-English words)."""
    text = text.lower()
    text = re.sub(r"\W+", " ", text)  # Remove non-word characters
    return text.strip()

def extract_skills(text):
    """Extract skills using spaCy NER."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "SKILL"]  # Custom NER labels

def clean_job_description(text):
    # Remove "Apply now" and similar phrases
    text = re.sub(r'Apply now', '', text, flags=re.IGNORECASE)
    
    # Remove multiple spaces, newlines, and excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_jobs(jobs_df):
    """Preprocess jobs dataset"""

    # Remove Apply Now and all that wasted space --------------------------------------------
    jobs_df = jobs_df.copy()
    jobs_df['description'] = jobs_df['description'].apply(clean_job_description)
    # ---------------------------------------------------------------------------------------

    # Remove nan ----------------------------------------------------------------------------
    jobs_df.dropna()
    # ---------------------------------------------------------------------------------------
    
    return jobs_df


def parse_hours(s):
    # 1) Try pattern like "2 months at 10 hours a week"
    match = re.search(r'(\d+)\s+months?\s+at\s+(\d+)\s+hours?\s+a\s+week', s)
    if match:
        months = int(match.group(1))
        hours_per_week = int(match.group(2))
        return months * 4 * hours_per_week

    # 2) Otherwise, try direct hours (e.g. "Approx. 11 hours", "11 hours (approx)")
    match = re.search(r'(\d+)\s+hours?', s)
    if match:
        return int(match.group(1))
    
    return None  # If something doesn’t match


def process_courses_1(courses_1):
    # Ensure working with a copy
    courses_1 = courses_1.copy()

    # Removing non-English rows
    courses_1 = courses_1[courses_1['Course Title'].astype(str).map(lambda x: x.isascii())]

    # Replace empty strings with NaN
    courses_1.loc[:, 'What you will learn'] = courses_1['What you will learn'].replace('', np.nan)

    # Create a mask for rows where description is NaN
    desc_is_empty = courses_1['What you will learn'].isna()

    # Ensure 'Modules' and 'Skill gain' are lists before processing
    courses_1.loc[:, 'Modules'] = courses_1['Modules'].apply(lambda x: x if isinstance(x, list) else [])
    courses_1.loc[:, 'Skill gain'] = courses_1['Skill gain'].apply(lambda x: x if isinstance(x, list) else [])

    # Create masks for non-empty lists
    modules_is_nonempty = courses_1['Modules'].apply(lambda x: len(x) > 0)
    skills_is_nonempty = courses_1['Skill gain'].apply(lambda x: len(x) > 0)

    # Fill description with skills if skills is nonempty
    use_summary_mask = desc_is_empty & skills_is_nonempty
    courses_1.loc[use_summary_mask, 'What you will learn'] = courses_1.loc[use_summary_mask, 'Skill gain'].apply(lambda x: ' '.join(x))

    # Fill description with modules if skills is empty
    use_skills_mask = desc_is_empty & modules_is_nonempty & ~skills_is_nonempty
    courses_1.loc[use_skills_mask, 'What you will learn'] = courses_1.loc[use_skills_mask, 'Modules'].apply(lambda x: ' '.join(x))

    # Remove any rows that still have empty or NaN description
    courses_1 = courses_1.dropna(subset=['What you will learn'])

    # Convert Duration column
    courses_1.loc[:, 'Duration'] = courses_1['Duration'].replace('', np.nan)
    courses_1 = courses_1.dropna(subset=['Duration'])
    courses_1.loc[:, 'Duration'] = courses_1['Duration'].astype(str).apply(parse_hours)

    # Convert Level column
    courses_1.loc[:, 'Level'] = courses_1['Level'].replace('', np.nan)
    courses_1 = courses_1.dropna(subset=['Level'])
    courses_1.loc[:, 'Level'] = courses_1['Level'].replace(
        {
            "Beginner level": "Beginner",
            "Intermediate level": "Intermediate",
            "Advanced level": "Advanced"
        }
    )

    # Rating conversion ---------------------------------------------------------------------
    # Not going this cause it's like 600 rows removed
    # Might replpace with 'Unkown' <- which lowers rating
    # courses_1.loc[:, 'Rating'] = courses_1['Rating'].replace('', np.nan)
    # courses_1 = courses_1.dropna(subset=['Rating'])
    # ---------------------------------------------------------------------------------------

    # Drop useless columns
    courses_1 = courses_1.drop(['Modules', 'Skill gain', 'Offered By', 'Keyword', 'Review', 'Schedule', 'Instructor'], axis=1)

    # Column renaming
    courses_1 = courses_1.rename(columns={
        'What you will learn': 'course_description',
        'Course Url': 'course_url',
        'Rating': 'course_rating',
        'Duration': 'course_time',
        'Level': 'course_difficulty',
        'Course Title': 'course_title'
    })

    return courses_1


def process_courses_2(courses_2):
    # Ensure working with a copy
    courses_2 = courses_2.copy()

    # Removing non-English rows
    courses_2 = courses_2[courses_2['course_title'].astype(str).map(lambda x: x.isascii())]

    # Replace empty strings with NaN
    courses_2.loc[:, 'course_description'] = courses_2['course_description'].replace('', np.nan)

    # Create a mask for rows where description is NaN
    desc_is_empty = courses_2['course_description'].isna()

    # Ensure 'course_summary' and 'course_skills' are lists before processing
    courses_2.loc[:, 'course_summary'] = courses_2['course_summary'].apply(lambda x: x if isinstance(x, list) else [])
    courses_2.loc[:, 'course_skills'] = courses_2['course_skills'].apply(lambda x: x if isinstance(x, list) else [])

    # Create masks for non-empty lists
    summary_is_nonempty = courses_2['course_summary'].apply(lambda x: len(x) > 0)
    skills_is_nonempty = courses_2['course_skills'].apply(lambda x: len(x) > 0)

    # Fill description with summary if summary is nonempty
    use_summary_mask = desc_is_empty & summary_is_nonempty
    courses_2.loc[use_summary_mask, 'course_description'] = courses_2.loc[use_summary_mask, 'course_summary'].apply(lambda lst: ' '.join(lst))

    # Fill description with skills if summary is empty but skills are nonempty
    use_skills_mask = desc_is_empty & ~summary_is_nonempty & skills_is_nonempty
    courses_2.loc[use_skills_mask, 'course_description'] = courses_2.loc[use_skills_mask, 'course_skills'].apply(lambda lst: ' '.join(lst))

    # Remove any rows that still have empty or NaN description
    courses_2.loc[:, 'course_description'] = courses_2['course_description'].replace('', np.nan)
    courses_2 = courses_2.dropna(subset=['course_description'])

    # Convert course_time column
    courses_2.loc[:, 'course_time'] = courses_2['course_time'].replace('', np.nan)
    courses_2 = courses_2.dropna(subset=['course_time'])
    courses_2.loc[:, 'course_time'] = courses_2['course_time'].replace(
        {
            '3 - 6 Months': 120,
            '1 - 3 Months': 40,
            '1 - 4 Weeks': 15,
            'Less Than 2 Hours': 2
        }
    ).infer_objects(copy=False)

    # Convert course_difficulty column
    courses_2.loc[:, 'course_difficulty'] = courses_2['course_difficulty'].replace('', np.nan)
    courses_2 = courses_2.dropna(subset=['course_difficulty'])
    courses_2 = courses_2[courses_2['course_difficulty'] != 'Mixed'] # Only 77 rows

    # Rating conversion ---------------------------------------------------------------------
    # Not going this cause it's like 600 rows removed
    # Might replpace with 'Unkown' <- which lowers rating
    # courses_2.loc[:, 'course_rating'] = courses_2['course_rating'].replace('', np.nan)
    # courses_2 = courses_2.dropna(subset=['course_rating'])
    # ---------------------------------------------------------------------------------------

    # Drop useless columns
    courses_2 = courses_2.drop(['course_organization', 'course_certificate_type', 'course_reviews_num', 'course_students_enrolled', 'course_skills', 'course_summary'], axis=1)

    return courses_2

def preprocess_courses(courses_1, courses_2):
    """Combines two course datasets together into one - changing and merging columns into one"""

    courses_1 = process_courses_1(courses_1)
    print(courses_1.head())

    courses_2 = process_courses_2(courses_2)
    print(courses_2.head())
    
    
    # Merge ---------------------------------------------------------------------------------
    courses = pd.concat([courses_1, courses_2], ignore_index=True)
    courses.drop_duplicates(subset='course_url', keep='first', inplace=True)
    # ---------------------------------------------------------------------------------------

    return courses



if __name__ == "__main__":
    from ..data.load_data import load_data

    courses_1, courses_2, jobs = load_data()
    
    courses = preprocess_courses(courses_1, courses_2)
    jobs_clean = preprocess_jobs(jobs)

    # Save processed data
    courses.to_csv(DATA_PATHS["courses_clean"], index=False)
    jobs_clean.to_csv(DATA_PATHS["jobs_clean"], index=False)