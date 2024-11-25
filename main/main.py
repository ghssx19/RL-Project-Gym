import gym
import gym_multi_car_racing
from gym_multi_car_racing import MultiCarRacing
from gym.wrappers import TimeLimit, ResizeObservation, GrayScaleObservation, FrameStack
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time
import warnings
import os
import traceback
import random
from datetime import datetime
import json

random.seed(time.time_ns())
fuel_consumption_rate = random.uniform(0.02, 0.1)
print(f"the fuel consumption rate is{fuel_consumption_rate}")
tire_wear_rate = random.uniform(0.02, 0.1)
print(f"the tire wear level is{tire_wear_rate}")

# Monkey-patching gym.spaces.Box to add 'shape' attribute
if not hasattr(gym.spaces.Box, "shape"):

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    gym.spaces.Box.shape = shape

# 2. Load the saved high-level model
print("Loading the saved high-level model...")
pit_model_path = (
    "/home/souren/Documents/RL-Project-Gym/main/models/bettersimplemodel2.zip"
)

if not os.path.exists(pit_model_path):
    raise FileNotFoundError(f"High-level model not found at {pit_model_path}")
pit_model = PPO.load(pit_model_path)
print("High-level model loaded successfully!")

# 3. Load the low-level model
low_level_model_path = (
    "/home/souren/Documents/RL-Project-Gym/main/models/LSTM_PPO_Low_LeveL_Driver.zip"
)

if not os.path.exists(low_level_model_path):
    raise FileNotFoundError(f"Low-level model not found at {low_level_model_path}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    low_level_model = RecurrentPPO.load(
        low_level_model_path,
        custom_objects={
            "learning_rate": 0.0,
            "clip_range": 0.2,
            "lr_schedule": lambda _: 0.0,
        },
    )
print("Low-level model loaded successfully!")


# 4. Wrapper to extract and preprocess single-agent observations
class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)
        self.action_space = env.action_space

    def reset(self):
        obs = self.env.reset()
        # print(f"SingleAgentWrapper.reset() observation shape: {obs.shape}")
        return obs[0]

    def step(self, action):
        # Apply the action to the single agent
        actions = [action]  # Since num_agents=1
        obs, rewards, done, info = self.env.step(actions)
        if isinstance(done, bool):
            agent_done = done
        else:
            agent_done = done[0]
        # print(f"SingleAgentWrapper.step() observation shape: {obs.shape}")  # Debug
        return obs[0], rewards[0], agent_done, info

    def render(self, mode="human", **kwargs):

        return self.env.render(mode=mode, **kwargs)


# Wrapper to transpose observations to channels-first format
class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        # print(f"TransposeImage: original observation shape: {obs_shape}")  # Debug
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
        # print(f"After Transpose: {observation.shape}")  # Debug
        return observation


# 5. Initialize the single-agent environment
multi_env = MultiCarRacing(
    num_agents=1,  # Single agent
    direction="CCW",
    use_random_direction=True,
    backwards_flag=True,
    h_ratio=0.25,
    use_ego_color=False,
)

# Wrap the environment with TimeLimit to set the maximum episode steps
max_steps_per_episode = 2000
multi_env = TimeLimit(multi_env, max_episode_steps=max_steps_per_episode)

# Create the single-agent environment with preprocessing
agent_env = SingleAgentWrapper(multi_env)
agent_env = ResizeObservation(agent_env, shape=(64, 64))
agent_env = GrayScaleObservation(agent_env, keep_dim=True)
agent_env = FrameStack(agent_env, num_stack=2)
agent_env = TransposeImage(agent_env)
agent_env = DummyVecEnv([lambda: agent_env])

# 6. Run the simulation
print("Running Pre-Trained Models...")
obs = agent_env.reset()
states = None
episode_start = True
total_rewards = 0.0
done = False
step_counter = 0

# Initialize variables for fuel and tire levels
fuel_level = 1.0
tire_tread_level = 1.0


import json


# Correcting the write function
def write_status_to_file(file_path, fuel_level, tire_wear, laptime):
    status = {
        "fuel_level": fuel_level,
        "tire_wear": tire_wear,
        "laptime": laptime,
    }
    try:
        with open(file_path, "w") as f:
            json.dump(status, f)
    except Exception as e:
        print(f"Error writing status to file: {e}")


# Call this function to write data before reading
write_status_to_file("status.json", 75.5, 10.2, 330.7)

from datetime import datetime, timedelta

# Making sure the fuel consumption rate and tire wear rate are within the range
print(f"the fuel consumption rate is{fuel_consumption_rate}")
print(f"the tire wear level is{tire_wear_rate}")
FPS = 50
timer = 0
start_time = datetime.now()
while not done:

    try:
        # Get the action from the low-level model
        action, states = low_level_model.predict(
            obs, state=states, episode_start=episode_start, deterministic=True
        )
        episode_start = False  # Reset episode_start after the first step

        # Update fuel and tire levels
        fuel_level -= fuel_consumption_rate * (1.0 / FPS)
        fuel_level = max(fuel_level, 0.0)
        tire_tread_level -= tire_wear_rate * (1.0 / FPS)

        tire_tread_level = max(tire_tread_level, 0.0)
        high_level_obersevation = np.array(
            [fuel_level, tire_tread_level], dtype=np.float32
        )
        action_high_level, _ = pit_model.predict(
            high_level_obersevation, deterministic=True
        )
        if step_counter % 10 == 0:
            print(
                f"Car 0: Fuel={fuel_level:.4f}, Tires={tire_tread_level:.4f}, Model action {action_high_level}"
            )
        #! if here
        if action_high_level == 1:
            print(
                f"Pitting: Car 0: Fuel={fuel_level:.4f}, Tires={tire_tread_level:.4f}, Model action {action_high_level}"
            )
            # Take a step in the environment with the chosen action
            obs_raw, rewards_raw, done, _ = multi_env.step([action])
            multi_env.render()
            # time.sleep(0.05)  # Control simulation speed
            pit_penalty = 15  # Add a time penalty for the pit stop
            pit_stop_time += pit_penalty
            start_time -= timedelta(seconds=pit_penalty)
            pit_stop_time = 0
            obs, agent_reward, agent_done, _ = agent_env.step([action])
            total_rewards += agent_reward

            # if agent_done:
            #     print(f"Agent 0 is done. Resetting agent.")
            #     obs = agent_env.reset()
            #     episode_start = True  # Reset episode start for the agent
            #     states = None  # Reset the model's internal state
            #     # Optionally reset fuel and tire levels for the agent
            #     fuel_level = 1.0
            #     tire_tread_level = 1.0

            fuel_level = 1.0
            tire_tread_level = 1.0
            step_counter += 1
            pit_stop_time = 5

        #! else here
        else:
            obs_raw, rewards_raw, done, _ = multi_env.step([action])

            multi_env.render()
            # time.sleep(0.05)

            obs, agent_reward, agent_done, _ = agent_env.step([action])
            total_rewards += agent_reward

            if agent_done:
                print(f"Agent 0 is done. Resetting agent.")
                obs = agent_env.reset()
                episode_start = True
                states = None

                fuel_level = 1.0
                tire_tread_level = 1.0
                print(f"the lapt time is {timer}")

            step_counter += 1
            pit_stop_time = 0
        current_time = datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()
        timer = elapsed_time + pit_stop_time
        write_status_to_file(
            "/home/souren/Documents/RL-Project-Gym/multi_car_racing/gym_multi_car_racing/status.json",
            fuel_level,
            tire_tread_level,
            timer,
        )
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        traceback.print_exc()
        break


print("Individual scores for the car:", total_rewards)
fuel_level = 1.0
tire_tread_level = 1.0
timer = 0
write_status_to_file(
    "/home/souren/Documents/RL-Project-Gym/multi_car_racing/gym_multi_car_racing/status.json",
    fuel_level,
    tire_tread_level,
    timer,
)
