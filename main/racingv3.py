import gym
from gym.wrappers import TimeLimit, GrayScaleObservation, ResizeObservation, FrameStack
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time
import warnings
import cv2

# Ensure the Box space has a shape attribute
if not hasattr(gym.spaces.Box, "shape"):

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    gym.spaces.Box.shape = shape


# Observation Wrapper to transpose and debug observation shapes
class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


# Debug and convert observations to grayscale
class DebugObservation(gym.ObservationWrapper):
    def observation(self, observation):
        print("Pre-GrayScale Shape:", observation.shape)
        if observation.ndim == 3 and observation.shape[2] == 1:
            return observation
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)


# Loading and configuring the CarRacing-v3 environment
env = gym.make("CarRacing-v3")
env = TimeLimit(env, max_episode_steps=1000)
env = TransposeImage(env)
env = DebugObservation(env)
env = FrameStack(env, num_stack=4)
env = DummyVecEnv([lambda: env])

print("Loading Pre-Trained Model...")
model_path = "/home/souren/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v0.zip"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = RecurrentPPO.load(
        model_path, custom_objects={"learning_rate": 0.0, "clip_range": 0.2}
    )

print("Running Pre-Trained Model...")
obs = env.reset()
state = None
episode_start = True
total_reward = 0
done = False

while not done:
    action, state = model.predict(
        obs, state=state, episode_start=episode_start, deterministic=True
    )
    obs, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.05)
    total_reward += reward
    episode_start = False  # Reset episode_start after the first step

print("Score for the car:", total_reward)
