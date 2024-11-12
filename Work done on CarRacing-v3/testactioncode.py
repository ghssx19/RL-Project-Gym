# Test code to input certain actions and see how the care reacts

import gymnasium as gym
import numpy as np

# Initialize the CarRacing environment
env = gym.make("CarRacing-v3", render_mode="human")
observation, info = env.reset()


# Function to set actions for controlling the car
def get_action(steering, acceleration, brake):
    # Ensure action values are within bounds
    steering = np.clip(steering, -1, 1)
    acceleration = np.clip(acceleration, 0, 1)
    brake = np.clip(brake, 0, 1)
    return np.array([steering, acceleration, brake])


# Sample control loop
try:
    for x in range(1000):
        # Set your desired control values
        steering = -0.95  # Adjust between -1 (left) and 1 (right)
        acceleration = 0.95  # Adjust between 0 (no throttle) and 1 (full throttle)
        brake = 0.15  # Adjust between 0 (no brake) and 1 (full brake)

        # Get the action array based on the values above
        action = get_action(steering, acceleration, brake)

        # Step through the environment with the chosen action
        observation, reward, terminated, truncated, info = env.step(action)

        # Reset environment if episode ends
        if terminated or truncated:
            observation, info = env.reset()

finally:
    env.close()  # Close the environment once done
