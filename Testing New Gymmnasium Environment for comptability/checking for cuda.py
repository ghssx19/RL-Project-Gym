import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Define your custom CNN
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute the output size of the CNN
        with torch.no_grad():
            obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(obs).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


# Observation wrapper to preprocess observations
class PreprocessObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.new_shape = (64, 64)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, *self.new_shape),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.new_shape, interpolation=cv2.INTER_AREA)
        obs = obs.reshape(1, *self.new_shape)
        return obs


# Frame skip wrapper
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=2):
        super(SkipFrame, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


# Create a list of environments
def make_env(seed):
    def _init():
        env = gym.make("CarRacing-v3", seed=seed)
        env = SkipFrame(env, skip=2)
        env = PreprocessObservation(env)
        return env

    return _init


if __name__ == "__main__":
    # Check if CUDA is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_envs = 8  # Adjust based on your hardware
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    env = VecFrameStack(env, n_stack=4)

    # Adjust batch_size and n_steps accordingly
    n_steps = 1024  # Reduced n_steps
    batch_size = (
        n_steps * num_envs
    )  # Ensure batch_size is a multiple of n_steps * num_envs

    # Define policy kwargs with a smaller CNN
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        device=device,
        learning_rate=0.001,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
    )

    # Train the model
    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("ppo_car_racing")

    # ================== Add Your Rendering Code Below ==================

    # Create a rendering environment
    def make_env_render():
        env = gym.make("CarRacing-v3", render_mode="human")
        env = PreprocessObservation(env)
        return env

    env_render = DummyVecEnv([make_env_render])

    # Load your trained model if not already in memory
    # If you've just trained the model, you can use the model object directly
    # If you're running this code separately, uncomment the next line to load the model
    # model = PPO.load("ppo_car_racing", env=env_render, device=device, policy_kwargs=policy_kwargs)

    # Run multiple episodes
    num_episodes = 5  # Adjust the number of episodes as desired

    for episode in range(num_episodes):
        obs = env_render.reset()
        done = False
        total_reward = 0
        while not done:
            # Get action from the trained model
            action, _states = model.predict(obs)
            # Take action in the environment
            obs, reward, done, info = env_render.step(action)
            # Since env_render is a vectorized environment, extract the first element
            done = done[0]
            reward = reward[0]
            total_reward += reward
            # Rendering is handled automatically when using render_mode='human'
        print(f"Episode {episode + 1} finished with reward: {total_reward}")
