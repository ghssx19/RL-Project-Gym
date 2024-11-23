import gym
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import time
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from stable_baselines3.common.vec_env import DummyVecEnv

# Monkey-patch gym.spaces.Box to add 'shape' attribute if missing
if not hasattr(gym.spaces.Box, "shape"):

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    gym.spaces.Box.shape = shape

# Load the high-level model (better simple model)
print("Loading the saved high-level model...")
high_level_model = PPO.load(
    "/home/souren/Documents/RL-Project-Gym/multi_car_racing/bettersimplemodel.zip"
)
print("High-level model loaded successfully!")

# Load the low-level model
low_level_model_path = "/home/souren/Documents/RL-Project-Gym/rl-baselines3-zoo/logs/ppo_lstm/CarRacing-v0_1/CarRacing-v0.zip"
print("Loading the saved low-level model...")
low_level_model = RecurrentPPO.load(
    low_level_model_path,
    custom_objects={"learning_rate": 0.0, "clip_range": 0.2},
)
print("Low-level model loaded successfully!")


# Custom environment with preprocessing
class CustomCarRacingEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomCarRacingEnv, self).__init__(env)
        self.env = ResizeObservation(env, 64)  # Resize observations to 64x64
        self.env = GrayScaleObservation(
            self.env, keep_dim=True
        )  # Grayscale observations
        self.env = FrameStack(self.env, num_stack=2)  # Stack 2 frames
        self.env = DummyVecEnv([lambda: self.env])  # Vectorize environment
        self.state = None

    def reset(self):
        self.state = None
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


# Initialize the environment
env = gym.make("CarRacing-v0")
custom_env = CustomCarRacingEnv(env)

# Initialize variables
obs = custom_env.reset()
print(f"Processed observation shape: {obs.shape}")
high_level_state = None
low_level_state = None
episode_start = True
done = False
step_counter = 0
total_reward = 0

# Variables for fuel and tire tread levels
fuel_level = 1.0
tire_tread = 1.0
fuel_consumption_rate = 0.09
tire_wear_rate = 0.09
FPS = 50  # Frames per second

# Run the simulation
print("Running the hierarchical model...")
while not done:
    # High-level model determines whether to pit
    high_level_obs = np.array([fuel_level, tire_tread], dtype=np.float32).reshape(1, -1)
    high_level_action, _ = high_level_model.predict(high_level_obs, deterministic=True)

    if high_level_action == 1:  # Pit stop
        print(
            f"Pitting: Fuel={fuel_level:.4f}, Tires={tire_tread:.4f}, Step={step_counter}"
        )
        fuel_level, tire_tread = 1.0, 1.0
        continue

    # Preprocess observation for low-level model
    obs = np.squeeze(obs, axis=0)  # Remove batch dimension
    obs = obs.transpose(1, 0, 2, 3).reshape(1, -1, 64, 64)  # Adjust shape for the model

    # Predict using the low-level model
    low_level_action, low_level_state = low_level_model.predict(
        obs,
        state=low_level_state,
        episode_start=episode_start,
        deterministic=True,
    )
    episode_start = False

    # Step environment
    obs, reward, done, _ = custom_env.step(low_level_action)
    total_reward += reward

    # Update fuel and tire levels
    fuel_level = max(fuel_level - fuel_consumption_rate / FPS, 0.0)
    tire_tread = max(tire_tread - tire_wear_rate / FPS, 0.0)

    if step_counter % 50 == 0:
        print(f"Step: {step_counter}, Fuel: {fuel_level:.4f}, Tire: {tire_tread:.4f}")

    # Render and sleep
    try:
        custom_env.env.envs[0].envs[0].render(mode="human")
    except AssertionError as e:
        print(f"Render mode issue: {e}")
        break  # Exit simulation if rendering fails

    time.sleep(1.0 / FPS)

    if done:
        print(f"Episode finished with total reward: {total_reward}")
        obs = custom_env.reset()
        high_level_state = None
        low_level_state = None
        episode_start = True

    step_counter += 1

custom_env.env.envs[0].envs[0].close()
