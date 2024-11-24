import gymnasium as gym
import numpy as np
import pygame

# Initialize the CarRacing-v3 environment
env = gym.make("CarRacing-v3", render_mode="human")


# Define key mappings
def get_action(keys):
    action = np.array([0.0, 0.0, 0.0])  # [steering, acceleration, brake]

    if keys[pygame.K_LEFT]:
        action[0] = -1.0  # full left
    elif keys[pygame.K_RIGHT]:
        action[0] = 1.0  # full right

    if keys[pygame.K_UP]:
        action[1] = 1.0  # accelerate
    if keys[pygame.K_DOWN]:
        action[2] = 0.8  # brake

    return action


# Initialize Pygame for capturing keyboard inputs
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Car Racing Human Control")
clock = pygame.time.Clock()

# Run the environment continuously
try:
    while True:
        # Reset the environment for each new episode
        observation, info = env.reset(seed=3)
        done = False

        while not done:
            # Capture events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    pygame.quit()
                    env.close()
                    exit()  # Exit the loop and the program if the user closes the window

            # Get the state of all keys
            keys = pygame.key.get_pressed()
            action = get_action(keys)

            # Step the environment with the chosen action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Control frame rate
            clock.tick(30)

finally:
    # Cleanup
    env.close()
    pygame.quit()
