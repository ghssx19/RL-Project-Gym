# testcode for loading the enviropnmennt

import gymnasium as gym

# Initialize the CarRacing environment
env = gym.make("CarRacing-v3", render_mode="human")

# Reset the environment to get the initial observation and additional info
observation, info = env.reset()

# Loop to take random actions in the environment
for _ in range(10000):
    action = env.action_space.sample()  # Take a random action
    observation, reward, terminated, truncated, info = env.step(action)

    # Check if the episode has terminated or truncated
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment
env.close()
