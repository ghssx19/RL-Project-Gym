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

os.environ["CUDA_VISIBLE_DEVICES"] = ""


FPS = 50
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

# Set the maximum number of steps per episode
max_steps_per_episode = 3000  # Adjust as needed

# Wrap the environment with TimeLimit to set the maximum episode steps
multi_env = TimeLimit(multi_env, max_episode_steps=max_steps_per_episode)

# Initialize variables for fuel and tire levels
num_agents = 2
fuel_levels = [1.0 for _ in range(num_agents)]  # Start with full fuel tanks
tire_tread_levels = [1.0 for _ in range(num_agents)]  # Start with new tires

# Define consumption rates
fuel_consumption_rate = 0.09  # Adjust as needed
tire_wear_rate = 0.09  # Adjust as needed

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

# Load the pre-trained models
print("Loading Pre-Trained Models for Both Cars...")
# low_level_model_paths = [
#     "/home/souren/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v0.zip",
#     "/home/souren/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v0.zip",
# ]
low_level_model_paths = [
    "/home/gibran/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v0.zip",
    "/home/gibran/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v0.zip",
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

# Define the custom environment for the higher-level RL agent
import gym
from gym import spaces


class PitStopDecisionEnv(gym.Env):
    def __init__(self):
        super(PitStopDecisionEnv, self).__init__()
        # Observation space: fuel level and tire tread level
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Action space: 0 (continue driving), 1 (pit stop)
        self.action_space = spaces.Discrete(2)
        # Initialize environment state
        self.reset()

    def reset(self):
        # Reset fuel and tire levels
        self.fuel_level = 1.0
        self.tire_tread_level = 1.0
        self.done = False
        return np.array([self.fuel_level, self.tire_tread_level], dtype=np.float32)

    def step(self, action):
        # Simulate fuel consumption and tire wear
        dt = 1.0  # Time step
        self.fuel_level -= fuel_consumption_rate * dt
        self.tire_tread_level -= tire_wear_rate * dt
        self.fuel_level = max(self.fuel_level, 0.0)
        self.tire_tread_level = max(self.tire_tread_level, 0.0)
        # print(fuel_levels, tire_tread_levels)

        reward = 0.0
        if action == 0:  # Continue driving
            if self.fuel_level > 0.1 and self.tire_tread_level > 0.1:
                reward += 1.0  # Positive reward for safe driving
            else:
                reward -= 10.0  # Negative reward for risking running out
        elif action == 1:  # Pit stop
            if self.fuel_level < 0.1 or self.tire_tread_level < 0.1:
                reward += 10.0
                self.fuel_level = 1.0
                self.tire_tread_level = 1.0
            else:
                reward -= 10  # Negative reward for pitting (cost)
                self.fuel_level = 1.0
                self.tire_tread_level = 1.0

        # Check if out of fuel or tires
        if self.fuel_level <= 0.0 or self.tire_tread_level <= 0.0:
            self.done = True
            reward -= 100.0  # Large penalty for running out

        observation = np.array(
            [self.fuel_level, self.tire_tread_level], dtype=np.float32
        )
        return observation, reward, self.done, {}

    def render(self, mode="human"):
        pass  # No rendering needed


# Train the higher-level RL agent
print("Training the Higher-Level RL Agent...")
from stable_baselines3 import PPO

# Create the environment
pit_env = PitStopDecisionEnv()

# Create the PPO model
pit_model = PPO("MlpPolicy", pit_env, verbose=1)

# Train the model
pit_model.learn(total_timesteps=3000)

# Save the trained model
pit_model.save("pit_stop_rl_agent")
print("Higher-level RL agent trained and saved as 'pit_stop_rl_agent'")

# Load the trained higher-level RL agent
pit_model = PPO.load("pit_stop_rl_agent")

# Run the simulation
print("Running Pre-Trained Models...")
obs = [env.reset() for env in agent_envs]
states = [None for _ in range(num_agents)]
episode_starts = [True for _ in range(num_agents)]
total_rewards = [0 for _ in range(num_agents)]
done = False
step_counter = 0
while not done:
    actions = []
    for i in range(num_agents):
        # Get the action from the low-level model
        action, states[i] = low_level_models[i].predict(
            obs[i], state=states[i], episode_start=episode_starts[i], deterministic=True
        )
        actions.append(action)
        episode_starts[i] = False  # Reset episode_starts after the first step

    combined_actions = [actions[i][0] for i in range(num_agents)]

    # Update fuel and tire levels for each car
    for i in range(num_agents):
        fuel_levels[i] -= fuel_consumption_rate * (1.0 / FPS)
        fuel_levels[i] = max(fuel_levels[i], 0.0)
        tire_tread_levels[i] -= tire_wear_rate * (1.0 / FPS)
        tire_tread_levels[i] = max(tire_tread_levels[i], 0.0)
        if step_counter % 100 == 0:
            print(
                f"Car {i}: Fuel={fuel_levels[i]:.4f}, Tires={tire_tread_levels[i]:.4f}"
            )

    # Use the higher-level RL agent to decide whether to pit
    for i in range(num_agents):
        pit_obs = np.array([fuel_levels[i], tire_tread_levels[i]], dtype=np.float32)
        pit_obs = pit_obs.reshape(1, -1)  # Add batch dimension
        pit_action, _ = pit_model.predict(pit_obs)
        # For now, just print the decision
        if step_counter % 100 == 0:
            if pit_action == 1:
                print(f"Car {i} should pit according to the RL agent.")
                if pit_action == 1:  # Pit stop
                    print(f"Car {i} is pitting! Resetting fuel and tire levels.")

            else:
                print(f"Car {i} should continue driving according to the RL agent.")

    # Step the environment
    obs_raw, rewards_raw, done, _ = multi_env.step(combined_actions)
    multi_env.render()
    time.sleep(0.05)

    for i in range(num_agents):
        obs[i], agent_reward, agent_done, _ = agent_envs[i].step(actions[i])
        total_rewards[i] += agent_reward

        if agent_done:
            print(f"Agent {i} is done. Resetting agent.")
            obs[i] = agent_envs[i].reset()
            episode_starts[i] = True  # Reset episode start for the agent
            states[i] = None  # Reset the model's internal state
            # Optionally reset fuel and tire levels for the agent
            fuel_levels[i] = 1.0
            tire_tread_levels[i] = 1.0

    # Increment step counter
    step_counter += 1

print("Individual scores for each car:", total_rewards)
