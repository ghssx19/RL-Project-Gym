import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
)  # Use SubprocVecEnv for parallelization
from stable_baselines3.common.evaluation import evaluate_policy

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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


# Create the environment with parallelization
def make_env():
    def _init():
        env = gym.make("CarRacing-v3")
        env = PreprocessObservation(env)
        return env

    return _init


if __name__ == "__main__":
    # Number of parallel environments
    num_envs = 10  # Adjust this number based on available system resources

    # Use SubprocVecEnv for parallelized environments
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # Initialize the model
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        device=device,
        learning_rate=0.001,
        n_steps=2048,  # With 8 environments, this results in 2048 * 8 = 16384 total steps per update
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    # Train the model
    total_timesteps = 1000000
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("ppo_car_racing")

    # Evaluate the model
    # Note: For evaluation, it's better to use a single environment
    eval_env = SubprocVecEnv(
        [make_env() for _ in range(1)]
    )  # Single environment for evaluation
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, render=False
    )
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    # Create a rendering environment (no parallelization for rendering)
    def make_env_render():
        env = gym.make("CarRacing-v3", render_mode="human")
        env = PreprocessObservation(env)
        return env

    env_render = SubprocVecEnv([make_env_render()])

    # Reset the environment
    obs = env_render.reset()

    # Run multiple episodes
    num_episodes = 1  # Adjust the number of episodes as desired

    for episode in range(num_episodes):
        obs = env_render.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env_render.step(action)
            done = done[0]  # Extract the boolean from the array
            total_reward += reward[0]
        print(f"Episode {episode + 1} finished with reward: {total_reward}")
