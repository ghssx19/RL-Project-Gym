import gymnasium as gymmnasium
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gym

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if not hasattr(gym.spaces.Box, "shape"):

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    gym.spaces.Box.shape = shape


# Observation wrapper to preprocess observations
class PreprocessObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(PreprocessObservation, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        # Transpose observations to (channels, height, width)
        return np.transpose(obs, (2, 0, 1))


# Create a rendering environment
def make_env_render():
    env = gymmnasium.make("CarRacing-v3")
    env = PreprocessObservation(env)
    return env


if __name__ == "__main__":
    # Load the pre-trained model
    model = PPO.load(
        "/home/souren/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v1.zip",
        device=device,
    )

    # Create a rendering environment
    env_render = make_env_render()

    # Reset the environment
    obs = env_render.reset()

    # Run multiple episodes
    num_episodes = 5  # Adjust the number of episodes as desired

    for episode in range(num_episodes):
        obs = env_render.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env_render.step(action)
            total_reward += reward
        print(f"Episode {episode + 1} finished with reward: {total_reward}")
