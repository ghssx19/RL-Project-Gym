import gym
import gym_multi_car_racing
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
    def __init__(self, env, agent_id):
        super(SingleAgentWrapper, self).__init__(env)
        self.agent_id = agent_id
        # The action space is the same as for a single agent
        self.action_space = env.action_space
        # The observation space will be updated after wrappers are applied

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
        # Since 'done' is a boolean, we return it directly
        return obs[self.agent_id], rewards[self.agent_id], done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)


# Wrapper to transpose observations to channels-first format
class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        # Adjust the observation space to match the new shape after transpose
        if len(obs_shape) == 3:
            # For shape (H, W, C)
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(obs_shape[2], obs_shape[0], obs_shape[1]),  # (C, H, W)
                dtype=np.uint8,
            )
        elif len(obs_shape) == 4:
            # For shape (N, H, W, C)
            num_frames = obs_shape[0]
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(num_frames, obs_shape[1], obs_shape[2]),  # (C, H, W)
                dtype=np.uint8,
            )
        else:
            raise ValueError(f"Unexpected observation space shape: {obs_shape}")

    def observation(self, observation):
        # Convert LazyFrames to NumPy array
        observation = np.array(observation)
        if observation.ndim == 4:
            # Observation shape is (N, H, W, C)
            # Squeeze the singleton channel dimension (axis=3)
            observation = np.squeeze(observation, axis=3)
            # Now shape is (N, H, W), which is (C, H, W)
            # No need to transpose
        elif observation.ndim == 3:
            # Observation shape is (H, W, C)
            # Transpose to (C, H, W)
            observation = observation.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
        return observation


# Initialize the multi-agent environment
multi_env = gym.make(
    "MultiCarRacing-v0",
    num_agents=2,
    direction="CCW",
    use_random_direction=True,
    backwards_flag=True,
    h_ratio=0.25,
    use_ego_color=False,
)

# Create a list to hold the per-agent environments
agent_envs = []

for agent_id in range(2):
    # Wrap the multi-agent environment for each agent
    agent_env = SingleAgentWrapper(multi_env, agent_id)

    # Apply the same preprocessing as during training
    agent_env = ResizeObservation(agent_env, 64)
    agent_env = GrayScaleObservation(agent_env, keep_dim=True)
    agent_env = FrameStack(agent_env, num_stack=2)
    agent_env = TransposeImage(agent_env)  # From HWC to CHW

    # Create a vectorized environment for the agent
    agent_env = DummyVecEnv([lambda agent_env=agent_env: agent_env])

    agent_envs.append(agent_env)

# Load the pre-trained models for each agent
print("Loading Pre-Trained Models for Both Cars...")

low_level_model_paths = [
    "/home/souren/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v0.zip",
    "/home/souren/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v0.zip",
]

low_level_models = []
for i, path in enumerate(low_level_model_paths):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RecurrentPPO.load(
            path,
            custom_objects={
                "learning_rate": 0.0,  # Set to 0.0 since we're not training
                "clip_range": 0.2,  # Provide a default value
            },
        )
    low_level_models.append(model)

# Run the simulation
print("Running Pre-Trained Models...")
obs = [env.reset() for env in agent_envs]
states = [None, None]
episode_starts = [True, True]
total_rewards = [0, 0]
done = False  # Initialize the done flag

while not done:
    actions = []
    for i in range(2):
        # obs[i] has shape (n_env, C, H, W), where n_env=1
        action, states[i] = low_level_models[i].predict(
            obs[i], state=states[i], episode_start=episode_starts[i], deterministic=True
        )
        actions.append(action)
        episode_starts[i] = False  # Since we reset at the beginning

    # Apply actions to the multi-agent environment
    combined_actions = [actions[i][0] for i in range(2)]
    obs_raw, rewards_raw, done, _ = multi_env.step(combined_actions)
    multi_env.render()
    time.sleep(0.05)

    # Update observations and rewards for each agent
    for i in range(2):
        if not done:
            # Process observation through wrappers
            obs[i], _, _, _ = agent_envs[i].step(actions[i])
            total_rewards[i] += rewards_raw[i]

print("Individual scores for each car:", total_rewards)
