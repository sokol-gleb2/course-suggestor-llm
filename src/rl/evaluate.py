def evaluate_rl_agent(env, model, judge_model, n_episodes=100):
    successes, total_courses = 0, 0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
        course_list = env._get_current_course_list()
        prediction = judge_model.predict(env.job_desc, course_list)
        if prediction == 1:
            successes += 1
        total_courses += len(course_list)

    return {
        "success_rate": successes / n_episodes,
        "avg_courses": total_courses / n_episodes
    }

def evaluate_gemini_baseline(jobs, gemini_recommender, judge_model):
    successes, total_courses = 0, 0
    for job in jobs:
        course_list = gemini_recommender(job['title'])  # list of course names
        prediction = judge_model.predict(job, course_list)
        if prediction == 1:
            successes += 1
        total_courses += len(course_list)
    return {
        "success_rate": successes / len(jobs),
        "avg_courses": total_courses / len(jobs)
    }

