import numpy as np
from stable_baselines3 import PPO
from gym import Env, spaces
import matplotlib.pyplot as plt
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Custom environment for a single agent
class SimplePitStopEnv(Env):
    def __init__(self):
        super(SimplePitStopEnv, self).__init__()
        # Observation space: [fuel level, tire wear]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Action space: 0 = keep driving, 1 = pit stop
        self.action_space = spaces.Discrete(2)
        # Initialize state variables
        self.reset()

    def reset(self):
        self.fuel_level = np.random.uniform(0.3, 1.0)
        self.tire_wear = np.random.uniform(0.3, 1.0)
        self.done = False
        self.step_count = 0
        self.last_action = None
        return np.array([self.fuel_level, self.tire_wear], dtype=np.float32)

    def step(self, action):
        # Simulate fuel consumption and tire wear per step
        fuel_consumption_rate = 0.05
        tire_wear_rate = 0.05

        reward = 0
        if self.last_action == 1:
            reward = -40  # Penalty for consecutive pit stops

        if action == 0 and self.fuel_level > 0.2 and self.tire_wear > 0.2:
            reward += 100  # Encourage safe driving

        if action == 0:  # Continue driving
            self.fuel_level -= fuel_consumption_rate
            self.tire_wear -= tire_wear_rate
            reward = 50  # Reward for driving safely
            if self.fuel_level <= 0.1 or self.tire_wear <= 0.1:
                reward = -10.0  # Penalty for risky driving

        elif action == 1:  # Pit stop
            reward = -10.0  # Penalty for taking a pit stop
            if self.fuel_level > 0.5 and self.tire_wear > 0.5:
                reward -= 40.0  # Additional penalty for unnecessary pit stops
            elif self.fuel_level <= 0.1 or self.tire_wear <= 0.1:
                reward += 100.0  # Reward for necessary pit stops

            self.fuel_level = 1.0  # Refuel
            self.tire_wear = 1.0  # Replace tires

            # Check if the last action was also a pit stop

        self.step_count += 1
        if self.step_count >= 500000:  # End the episode after 200 steps
            self.done = True

        # Update last action
        self.last_action = action

        # Observation is the new state
        observation = np.array([self.fuel_level, self.tire_wear], dtype=np.float32)
        return observation, reward, self.done, {}

    def render(self, mode="human"):
        return self.fuel_level, self.tire_wear


# Create the environment
env = SimplePitStopEnv()

# Track rewards per episode
reward_history = []

# Train the PPO model
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01, gamma=0.99, learning_rate=1e-5)
num_episodes = 10  # Number of episodes to simulate
episode_rewards = []

# Train PPO over multiple episodes
for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0
    done = False
    timer = 0
    while not done:
        timer += 1
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if timer % 1000 == 0:
            tyre_print, fuel_print = env.render()
            print(
                f"Fuel: {fuel_print:.2f}, Tire Wear: {tyre_print:.2f}, and the action is: {action}, and the reward for this time step is: {reward}, the total reward is, {total_reward}. we are on time step {timer}"
            )
    episode_rewards.append(total_reward)
    reward_history.append((episode + 1, total_reward))

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Save the trained model
model.save("simple_pitstop_model")

# Evaluation: Test the model and track rewards
test_episodes = 50
test_rewards = []
for test_episode in range(test_episodes):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)
    print(f"Test Episode {test_episode + 1}: Total Reward = {total_reward}")

# Plot: Total Reward Per Training Episode
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_episodes + 1), episode_rewards, label="Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward Per Episode During Training")
plt.legend()
plt.grid()
plt.show()

# Plot: Test Rewards
plt.figure(figsize=(12, 6))
plt.bar(
    range(1, test_episodes + 1),
    test_rewards,
    color="orange",
    alpha=0.7,
    label="Test Rewards",
)
plt.xlabel("Test Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward Per Test Episode")
plt.legend()
plt.grid()
plt.show()

# Plot: Cumulative Rewards During Training
cumulative_rewards = np.cumsum(episode_rewards)
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_episodes + 1), cumulative_rewards, label="Cumulative Rewards")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards During Training")
plt.legend()
plt.grid()
plt.show()
