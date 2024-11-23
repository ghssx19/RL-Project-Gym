import numpy as np
from stable_baselines3 import PPO
from gym import Env, spaces
import matplotlib.pyplot as plt
import os
import torch.nn as nn

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

        # Penalize consecutive pit stops
        if self.last_action == 1 and action == 1:
            reward -= 40  # Penalty for consecutive pit stops

        if action == 0 and self.fuel_level > 0.2 and self.tire_wear > 0.2:
            reward += 100  # Encourage safe driving

        if action == 0:  # Continue driving
            self.fuel_level -= fuel_consumption_rate
            self.tire_wear -= tire_wear_rate
            reward += 50  # Reward for driving safely
            if self.fuel_level <= 0.1 or self.tire_wear <= 0.1:
                reward -= 10.0  # Penalty for risky driving

        elif action == 1:  # Pit stop
            reward -= 10.0  # Penalty for taking a pit stop
            if self.fuel_level > 0.5 and self.tire_wear > 0.5:
                reward -= 40.0  # Additional penalty for unnecessary pit stops
            elif self.fuel_level <= 0.1 or self.tire_wear <= 0.1:
                reward += 100.0  # Reward for necessary pit stops

            self.fuel_level = 1.0  # Refuel
            self.tire_wear = 1.0  # Replace tires

        self.step_count += 1
        if self.step_count >= 1000:  # End the episode after 1000 steps
            self.done = True

        # Update last action
        self.last_action = action

        # Observation is the new state
        observation = np.array([self.fuel_level, self.tire_wear], dtype=np.float32)
        return observation, reward, self.done, {}

    def render(self, mode="human"):
        print(f"Fuel Level: {self.fuel_level:.2f}, Tire Wear: {self.tire_wear:.2f}")


# Create the environment
env = SimplePitStopEnv()

# Custom policy settings
policy_kwargs = dict(
    net_arch=[
        dict(pi=[64, 64], vf=[64, 64])
    ],  # Two hidden layers of size 64 for policy and value
    activation_fn=nn.ReLU,
)

# Train the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=1,
)

# Train the model
model.learn(total_timesteps=500_000)

# Save the trained model
model.save("simple_pitstop_model")

# Evaluation: Test the model
test_episodes = 10
test_rewards = []
for episode in range(test_episodes):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    test_rewards.append(total_reward)
    print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

# Plot: Test Rewards
plt.figure(figsize=(12, 6))
plt.bar(range(1, test_episodes + 1), test_rewards, color="orange", alpha=0.7)
plt.xlabel("Test Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward Per Test Episode")
plt.grid()
plt.show()
