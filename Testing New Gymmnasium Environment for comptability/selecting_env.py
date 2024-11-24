import gymnasium as gym
import pygame
import numpy as np
import time

# Initialize the CarRacing-v3 environment
env = gym.make(
    "CarRacing-v3", render_mode="rgb_array"
)  # Start with "rgb_array" mode for previews

# Initialize Pygame for capturing keyboard inputs and screen display
pygame.init()
screen_width, screen_height = 1000, 1000
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Car Racing Track Preview")


# Track preview function with resizing
def preview_track(seed, display_time=3):
    env.reset(seed=seed)
    frame = env.render()  # Render the track as an image
    resized_frame = pygame.transform.scale(
        pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
        (screen_width, screen_height),
    )  # Resize to match Pygame screen dimensions
    screen.blit(resized_frame, (0, 0))  # Display resized frame
    pygame.display.flip()  # Update display
    time.sleep(display_time)  # Display track for a few seconds


# Preview multiple tracks and let the user select one
available_seeds = range(10)  # Choose how many seeds to preview, e.g., seeds 0 to 9
chosen_seed = None

for seed in available_seeds:
    print(f"Previewing track with seed: {seed}")
    preview_track(seed)
    env.close()  # Close and re-open to refresh track rendering

    # Ask user if they want this track
    user_input = input(f"Do you want to select track with seed {seed}? (y/n): ")
    if user_input.lower() == "y":
        chosen_seed = seed
        print(f"Selected track with seed: {seed}")
        break

# Reinitialize Pygame and environment in "human" mode after selection
pygame.quit()  # Close previous Pygame display
env = gym.make("CarRacing-v3", render_mode="human")

# Run the selected track with human control
if chosen_seed is not None:
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
