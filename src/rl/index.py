import gym
from gym import spaces

class CourseRecommenderEnv(gym.Env):
    def __init__(self, course_pool, judge_model, tokenizer, job_desc, alpha=0.5, beta=0.3, max_courses=6):
        super().__init__()
        self.course_pool = course_pool
        self.judge_model = judge_model
        self.tokenizer = tokenizer
        self.job_desc = job_desc
        self.alpha = alpha
        self.beta = beta
        self.max_courses = max_courses

        self.action_space = spaces.Discrete(len(course_pool))  # each course = 1 action
        self.observation_space = spaces.MultiBinary(len(course_pool))  # binary mask: courses added
        self.reset()

    def reset(self):
        self.recommended = []
        self.selected_mask = [0] * len(self.course_pool)
        return self.selected_mask

    def _get_current_course_list(self):
        return [self.course_pool[i] for i, sel in enumerate(self.selected_mask) if sel == 1]

    def _evaluate_courses(self):
        course_list = self._get_current_course_list()
        input_text = f"Job: {self.job_desc}\nCourses:\n" + "\n".join(course_list)
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.judge_model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze()
            prediction = torch.argmax(probs).item()
        return prediction  # 0 = jobless, 1 = job secured

    def step(self, action):
        done = False
        reward = 0.0

        if self.selected_mask[action] == 1:
            # Already selected
            return self.selected_mask, -1.0, False, {}

        self.selected_mask[action] = 1
        current_courses = self._get_current_course_list()

        if len(current_courses) >= self.max_courses:
            done = True

        outcome = self._evaluate_courses()
        if outcome == 1:
            reward = 10 - self.alpha * len(current_courses)
            done = True
        else:
            reward = -self.beta * len(current_courses)

        return self.selected_mask, reward, done, {}

    def render(self):
        print("Selected:", self._get_current_course_list())

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    import torch

    # Load your judge model
    judge_model = DistilBertForSequenceClassification.from_pretrained("./distilbert-job-evaluator")
    tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert-job-evaluator")

    # Dummy course pool
    course_pool = [
        "Intro to Python", "Linear Algebra", "SQL Basics", "Deep Learning", "Data Structures",
        "Statistics 101", "Machine Learning", "Data Visualization"
    ]

    # Define job
    job = "Machine Learning Engineer"

    # Create env
    env = CourseRecommenderEnv(course_pool, judge_model, tokenizer, job)
    check_env(env)

    # Train PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)

    # Test
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
    print(f"Final Reward: {reward}")

