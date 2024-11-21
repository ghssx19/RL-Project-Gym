import gym
import multi_car_racing.gym_multi_car_racing
from multi_car_racing.gym_multi_car_racing import MultiCarRacing

from gym.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import numpy as np
import time
import warnings
import cv2
import pandas as pd


# Monkey-patch gym.spaces.Box to have 'shape' attribute if it doesn't exist
if not hasattr(gym.spaces.Box, "shape"):

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    gym.spaces.Box.shape = shape


# Custom wrapper to extract and preprocess single-agent observations and track fuel
class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env, agent_id, initial_fuel=1.0):
        """
        Wrapper for single-agent observations and fuel tracking.
        :param env: The multi-agent environment.
        :param agent_id: ID of the agent (0 or 1).
        :param initial_fuel: Initial fuel level for the agent.
        """
        super(SingleAgentWrapper, self).__init__(env)
        self.agent_id = agent_id
        self.action_space = env.action_space
        self.fuel = initial_fuel  # Start with full fuel tank

    def reset(self):
        """
        Reset the environment and the agent's fuel level.
        :return: The initial observation for the agent.
        """
        obs = self.env.reset()
        self.fuel = 1.0  # Reset fuel to full tank
        return obs[self.agent_id]

    def step(self, action):
        """
        Perform a step in the environment for the agent.
        :param action: The action to take.
        :return: A tuple (observation, reward, done, info).
        """
        obs, rewards, done, info = self.env.step(action)

        # Decrease fuel based on speed
        speed = info["speed"][self.agent_id]  # Assuming 'speed' is available in info
        self.fuel -= 0.001 * speed  # Adjust consumption rate as needed

        # End episode if fuel runs out
        if self.fuel <= 0:
            done = True  # The car runs out of fuel
            print(f"Car {self.agent_id} ran out of fuel!")

        return obs[self.agent_id], rewards[self.agent_id], done, info

    def refuel(self):
        """
        Refuel the agent's fuel tank to full.
        """
        self.fuel = 1.0


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


# Render the environment with fuel levels overlaid
def render_with_fuel(env, agent_fuels):
    """
    Render the environment and overlay fuel levels.
    :param env: The racing environment to render.
    :param agent_fuels: List of current fuel levels for each agent.
    """
    # Render the environment and get the frame as an RGB image
    frame = env.render(mode="rgb_array")
    height, width, _ = frame.shape

    # Iterate through the agents and overlay their fuel levels as text
    for i, fuel in enumerate(agent_fuels):
        text = f"Car {i} Fuel: {fuel:.2f}"  # Format the fuel level
        position = (10, 30 + i * 30)  # Position the text on the frame
        frame = cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1,  # Font scale
            (255, 255, 255),  # Text color (white)
            2,  # Line thickness
            cv2.LINE_AA,  # Line type
        )

    # Display the frame in a window using OpenCV
    cv2.imshow("Race Simulation", frame)
    cv2.waitKey(1)  # Delay to update the window


# Initialize the multi-agent environment
multi_env = MultiCarRacing(
    num_agents=2,
    direction="CCW",
    use_random_direction=True,
    backwards_flag=True,
    h_ratio=0.25,
    use_ego_color=False,
)

# Set the maximum number of steps per episode
max_steps_per_episode = 2000  # Adjust as needed
multi_env = TimeLimit(multi_env, max_episode_steps=max_steps_per_episode)

# Create per-agent environments
agent_envs = []
for agent_id in range(2):
    agent_env = SingleAgentWrapper(multi_env, agent_id)
    agent_env = ResizeObservation(agent_env, 64)
    agent_env = GrayScaleObservation(agent_env, keep_dim=True)
    agent_env = FrameStack(agent_env, num_stack=2)
    agent_env = TransposeImage(agent_env)
    agent_env = DummyVecEnv([lambda agent_env=agent_env: agent_env])
    agent_envs.append(agent_env)

# Load the pre-trained models
print("Loading Pre-Trained Models for Both Cars...")
low_level_model_paths = [
    "/home/gibran/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/benchmark/ppo_lstm-CarRacing-v0/ppo_lstm-CarRacing-v0.zip",
    "/home/gibran/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/benchmark/ppo_lstm-CarRacing-v0/ppo_lstm-CarRacing-v0.zip",
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

# Initialize fuel log for each car
fuel_log = [[] for _ in range(2)]

# Run the simulation
print("Running Pre-Trained Models...")
obs = [env.reset() for env in agent_envs]
states = [None, None]
episode_starts = [True, True]
total_rewards = [0, 0]
done = False

while not done:
    actions = []
    for i in range(2):
        # Predict actions using pre-trained models
        action, states[i] = low_level_models[i].predict(
            obs[i], state=states[i], episode_start=episode_starts[i], deterministic=True
        )
        actions.append(action)
        episode_starts[i] = False  # Reset episode_starts after the first step

    # Combine actions and step the environment
    combined_actions = [actions[i][0] for i in range(2)]
    obs_raw, rewards_raw, done, _ = multi_env.step(combined_actions)

    # Log fuel levels for each car
    for i in range(2):
        fuel_log[i].append(agent_envs[i].envs[0].fuel)

    # Render with fuel overlay
    render_with_fuel(multi_env, [agent_env.envs[0].fuel for agent_env in agent_envs])

    # Step individual agent environments
    for i in range(2):
        obs[i], _, _, _ = agent_envs[i].step(actions[i])
        total_rewards[i] += rewards_raw[i]

# Print final scores
print("Individual scores for each car:", total_rewards)

# Save fuel logs to CSV files
for i, log in enumerate(fuel_log):
    df = pd.DataFrame(log, columns=["fuel"])
    df.to_csv(f"car_{i}_fuel_log.csv", index=False)
    print(f"Fuel log for Car {i} saved to car_{i}_fuel_log.csv")
