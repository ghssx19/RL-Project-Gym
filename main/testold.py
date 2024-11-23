import gym
import gym_multi_car_racing
from gym_multi_car_racing import MultiCarRacing

from gym.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import numpy as np
import time
import warnings

# Monkey-patch gym.spaces.Box to have 'shape' attribute if it doesn't exist
if not hasattr(gym.spaces.Box, "shape"):

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    gym.spaces.Box.shape = shape


# Custom wrapper to extract and preprocess single-agent observations
class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)
        self.action_space = env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        # Apply the action only to the specified agent
        # Assuming the environment expects a list of actions for all agents
        actions = [action]  # Wrap the single action in a list
        obs, rewards, done, info = self.env.step(actions)
        # Since 'done' can be a bool or list, ensure it's handled correctly
        return obs[0], rewards[0], done, info

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
            # Now shape is (N, H, W)
        elif observation.ndim == 3:
            # Observation shape is (H, W, C)
            observation = observation.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
        return observation


# Set your desired maximum number of steps per episode
max_steps_per_episode = 2000  # Adjust as needed

# Initialize the multi-agent environment directly
multi_env = MultiCarRacing(
    num_agents=1,
    direction="CCW",
    use_random_direction=True,
    backwards_flag=True,
    h_ratio=0.25,
    use_ego_color=False,
)

# Wrap the environment with TimeLimit to set the maximum episode steps
multi_env = TimeLimit(multi_env, max_episode_steps=max_steps_per_episode)

# Create per-agent environments
agent_env = SingleAgentWrapper(multi_env)
agent_env = ResizeObservation(agent_env, shape=(64, 64))  # Updated shape
agent_env = GrayScaleObservation(agent_env, keep_dim=True)
agent_env = FrameStack(agent_env, num_stack=2)
agent_env = TransposeImage(agent_env)

# Vectorize the environment
agent_env = DummyVecEnv([lambda: agent_env])

# Load the pre-trained models
print("Loading Pre-Trained Models for Both Cars...")
low_level_model_paths = [
    "/home/souren/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v0.zip"
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = RecurrentPPO.load(
        low_level_model_paths[0],
        custom_objects={"learning_rate": 0.0, "clip_range": 0.2},
    )

# Run the simulation
print("Running Pre-Trained Models...")
obs = agent_env.reset()
states = (
    None  # RecurrentPPO expects states per environment, which is 1 after vectorization
)
episode_starts = np.ones((agent_env.num_envs, 1), dtype=bool)
total_rewards = np.zeros((agent_env.num_envs,), dtype=np.float32)
done = False

while not done:
    action, states = model.predict(
        obs, state=states, episode_start=episode_starts, deterministic=True
    )

    # Take a step in the environment
    obs, rewards, done, info = agent_env.step(action)

    # Render the environment
    agent_env.envs[0].render()

    # Accumulate rewards
    total_rewards += rewards

    # Update episode starts
    episode_starts = done

    # Sleep to control the simulation speed
    time.sleep(0.05)

print("Individual scores for each car:", total_rewards)
