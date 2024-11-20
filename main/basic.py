import gym
import gym_multi_car_racing
import numpy as np

env = gym.make(
    "MultiCarRacing-v0",
    num_agents=2,
    direction="CCW",
    use_random_direction=True,
    backwards_flag=True,
    h_ratio=0.25,
    use_ego_color=False,
)

obs = env.reset()
done = False
total_reward = 0

# Random actions for demonstration purposes
while not done:
    action = np.random.uniform(low=-1.0, high=1.0, size=(2, 3))  # Random actions
    obs, reward, done, info = env.step(action)
    total_reward += sum(reward)
    env.render()

print("individual scores:", total_reward)
