import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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
        self.reset()

    def reset(self):
        self.fuel_level = np.random.uniform(0.3, 1.0)
        self.tire_wear = np.random.uniform(0.3, 1.0)
        self.done = False
        self.step_count = 0
        self.last_action = None
        return np.array([self.fuel_level, self.tire_wear], dtype=np.float32)

    def step(self, action):
        fuel_consumption_rate = 0.05
        tire_wear_rate = 0.05

        reward = 0
        if action == 0:  # Keep driving
            self.fuel_level = max(0.0, self.fuel_level - fuel_consumption_rate)
            self.tire_wear = max(0.0, self.tire_wear - tire_wear_rate)
            reward += 50  # Reward for safe driving

            if self.fuel_level <= 0.1 or self.tire_wear <= 0.1:
                reward -= 1000.0  # Heavy penalty for critical levels
                self.done = True  # End episode

        elif action == 1:  # Pit stop
            reward -= 1.0  # Penalty for taking a pit stop
            if self.fuel_level > 0.5 and self.tire_wear > 0.5:
                reward -= 40.0  # Penalty for unnecessary pit stop
            elif self.fuel_level <= 0.1 or self.tire_wear <= 0.1:
                reward += 200.0
            else:
                reward += 50.0  # Reward for a timely pit stop

            self.fuel_level = 1.0
            self.tire_wear = 1.0

        self.step_count += 1
        if self.step_count >= 1000:
            self.done = True

        observation = np.array([self.fuel_level, self.tire_wear], dtype=np.float32)

        n = 1

        if self.step_count % n == 0:
            print(
                f"Step: {self.step_count}, "
                f"Fuel Level: {self.fuel_level:.2f}, "
                f"Tire Wear: {self.tire_wear:.2f}, "
                f"Action Taken: {'Pit Stop' if action == 1 else 'Keep Driving'}, "
                f"Reward: {reward:.2f}"
            )

        return observation, reward, self.done, {}

    def render(self, mode="human"):
        print(f"Fuel Level: {self.fuel_level:.2f}, Tire Wear: {self.tire_wear:.2f}")


class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardTrackerCallback, self).__init__(verbose)
        self.epoch_rewards = []  # Store rewards per epoch
        self.epoch_agg_rewards = 0  # Aggregate rewards for the current epoch
        self.steps_in_epoch = 0

    def _on_step(self) -> bool:
        # Accumulate rewards and steps
        self.epoch_agg_rewards += self.locals["rewards"][
            0
        ]  # Assuming single environment
        self.steps_in_epoch += 1

        # Determine number of environments
        num_envs = len(self.training_env.envs)

        # Check if epoch ended
        if self.steps_in_epoch >= num_envs * self.model.n_steps:
            # Store the average reward per epoch
            self.epoch_rewards.append(self.epoch_agg_rewards / self.steps_in_epoch)
            # Reset counters
            self.epoch_agg_rewards = 0
            self.steps_in_epoch = 0

        return True

    def plot_rewards(self):
        epochs = range(len(self.epoch_rewards))
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.epoch_rewards, label="Epoch Rewards")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title("Training Rewards Over Time")
        plt.legend()
        plt.grid()
        plt.show()


# Create the environment
env = SimplePitStopEnv()

# Custom policy settings
policy_kwargs = dict(
    net_arch=[
        dict(pi=[64, 64], vf=[64, 64])
    ],  # Two hidden layers of size 64 for policy and value
    activation_fn=nn.ReLU,
)

# Instantiate the model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=7e-5,
    n_steps=2048,
    batch_size=64,
    n_epochs=40,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=1,
)

# Instantiate the callback
reward_tracker = RewardTrackerCallback()

# Train the model with the callback
model.learn(total_timesteps=500_000, callback=reward_tracker)

# Save the trained model
model.save("Pit_Stop_Model_PPO_MlpPolicy(gibby)")

# Plot the training rewards
reward_tracker.plot_rewards()

# Evaluation: Test the model
test_episodes = 10
test_rewards = []
for episode in range(test_episodes):
    obs = env.reset()
    print(f"[TEST START] Episode {episode + 1}, Initial Obs: {obs}")
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
