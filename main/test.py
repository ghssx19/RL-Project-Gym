import gym
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import cv2
from multi_car_racing.gym_multi_car_racing import MultiCarRacing
from gym.wrappers import TimeLimit, ResizeObservation, GrayScaleObservation, FrameStack
import gym
import gym_multi_car_racing
from gym_multi_car_racing import MultiCarRacing

from gym.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import numpy as np
import time
import warnings
import os


# Monkey-patch gym.spaces.Box to add 'shape' attribute if missing
if not hasattr(gym.spaces.Box, "shape"):

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    gym.spaces.Box.shape = shape

# Load the saved high-level model
print("Loading the saved high-level model...")
pit_model = PPO.load("pit_stop_rl_agent")
print("High-level model loaded successfully!")

# Load the low-level model
low_level_model_paths = [
    "/home/gibran/Documents/RL-Project-Gym/ppo_lstm-CarRacing-v0/ppo_lstm-CarRacing-v0.zip",
    "/home/gibran/Documents/RL-Project-Gym/ppo_lstm-CarRacing-v0/ppo_lstm-CarRacing-v0.zip",
]

low_level_models = []
for i, path in enumerate(low_level_model_paths):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RecurrentPPO.load(
            path,
            custom_objects={"learning_rate": 0.0, "clip_range": 0.2},
        )
    low_level_models.append(model)


# Custom wrapper to extract and preprocess single-agent observations
class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env, agent_id):
        super(SingleAgentWrapper, self).__init__(env)
        self.agent_id = agent_id
        self.action_space = env.action_space

    def reset(self):
        obs = self.env.reset()
        return obs[self.agent_id]

    def step(self, action):
        # Apply the action only to the specified agent
        actions = [
            action if i == self.agent_id else [0, 0, 0]
            for i in range(self.env.num_agents)
        ]
        obs, rewards, done, info = self.env.step(actions)
        # Since 'done' can be a bool or list, ensure it's handled correctly
        if isinstance(done, bool):
            agent_done = done
        else:
            agent_done = done[self.agent_id]
        return obs[self.agent_id], rewards[self.agent_id], agent_done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)


# Wrapper to transpose observations to channels-first format
class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        if len(obs_shape) == 3:
            # For shape (H, W, C)
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
                dtype=np.uint8,
            )
        elif len(obs_shape) == 4:
            # For shape (N, H, W, C)
            num_frames = obs_shape[0]
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(num_frames, obs_shape[1], obs_shape[2]),
                dtype=np.uint8,
            )
        else:
            raise ValueError(f"Unexpected observation space shape: {obs_shape}")

    def observation(self, observation):
        observation = np.array(observation)
        if observation.ndim == 4:
            # Observation shape is (N, H, W, C)
            observation = np.squeeze(observation, axis=3)
            # Now shape is (N, H, W), which is (C, H, W)
        elif observation.ndim == 3:
            # Observation shape is (H, W, C)
            observation = observation.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
        return observation


# Set your desired maximum number of steps per episode
# Initialize the multi-agent environment directly
multi_env = MultiCarRacing(
    num_agents=2,
    direction="CCW",
    use_random_direction=True,
    backwards_flag=True,
    h_ratio=0.25,
    use_ego_color=False,
)


# Load the trained higher-level RL agent
pit_model = PPO.load("pit_stop_rl_agent")

num_agents = 2

# Create per-agent environments
agent_envs = []
for agent_id in range(num_agents):
    agent_env = SingleAgentWrapper(multi_env, agent_id)
    agent_env = ResizeObservation(agent_env, 64)
    agent_env = GrayScaleObservation(agent_env, keep_dim=True)
    agent_env = FrameStack(agent_env, num_stack=2)
    agent_env = TransposeImage(agent_env)
    agent_env = DummyVecEnv([lambda agent_env=agent_env: agent_env])
    agent_envs.append(agent_env)

# Run the simulation
print("Running Pre-Trained Models...")
obs = [env.reset() for env in agent_envs]
states = [None for _ in range(num_agents)]
episode_starts = [True for _ in range(num_agents)]
total_rewards = [0 for _ in range(num_agents)]
done = False
step_counter = 0
