import gymnasium as gym
import numpy as np
import pygame
import cv2  # Import cv2 if not already imported
from gymnasium import Wrapper


class FuelGaugeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FuelGaugeWrapper, self).__init__(env)
        self.max_fuel = 100.0
        self.fuel = self.max_fuel
        self.refuel_radius = 2.0

    def reset(self, **kwargs):
        self.fuel = self.max_fuel
        observation, info = self.env.reset(**kwargs)
        self.define_refuel_zone()
        return observation, info

    def define_refuel_zone(self):
        self.refuel_tile_index = 10  # Adjust as needed
        try:
            tile = self.env.unwrapped.track[self.refuel_tile_index]
            if len(tile) >= 3:
                self.refuel_zone = (tile[1], tile[2])  # x_center and y_center
                print(f"Refuel Zone defined at: {self.refuel_zone}")
            else:
                print(
                    f"Error: Tile at index {self.refuel_tile_index} does not have enough coordinates."
                )
                self.refuel_zone = (0, 0)
        except IndexError:
            print(
                f"Error: track does not have a tile at index {self.refuel_tile_index}."
            )
            self.refuel_zone = (0, 0)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Fuel consumption
        car_speed = np.linalg.norm(self.env.unwrapped.car.hull.linearVelocity)
        fuel_consumption_rate = car_speed * 0.001
        self.fuel -= fuel_consumption_rate
        self.fuel = max(self.fuel, 0)

        # Refueling logic
        car_position = self.env.unwrapped.car.hull.position
        distance_squared = (car_position[0] - self.refuel_zone[0]) ** 2 + (
            car_position[1] - self.refuel_zone[1]
        ) ** 2

        if distance_squared < self.refuel_radius**2 and car_speed < 0.1:
            self.fuel = self.max_fuel
            print("Refueled to maximum fuel.")

        info["fuel"] = self.fuel
        print(f"Fuel: {self.fuel}, Terminated: {terminated}, Truncated: {truncated}")
        return observation, reward, terminated, truncated, info


class FuelGaugeRenderingWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FuelGaugeRenderingWrapper, self).__init__(env)
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Car Racing with Fuel Gauge")

    def render(self):
        # Since render_mode="rgb_array", env.render() returns the frame
        frame = self.env.render()

        # Correct the frame orientation
        frame = np.rot90(frame, 1)
        frame = np.flipud(frame)

        # Ensure frame is uint8 and contiguous
        frame = frame.astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        fuel_percentage = self.env.fuel / self.env.max_fuel

        # Draw fuel bar on the frame
        bar_length = 200  # Length of the fuel bar
        bar_height = 20  # Height of the fuel bar
        x, y = 10, 10  # Position of the fuel bar
        fuel_bar = int(bar_length * fuel_percentage)  # Calculate fuel bar length

        cv2.rectangle(frame, (x, y), (x + bar_length, y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + fuel_bar, y + bar_height), (0, 255, 0), -1)

        # Draw refuel zone
        self.draw_refuel_zone_on_frame(frame)

        # Convert frame to Pygame surface
        frame_surface = pygame.surfarray.make_surface(frame)
        frame_surface = pygame.transform.scale(
            frame_surface, (self.screen_width, self.screen_height)
        )

        # Blit the frame onto the screen
        self.screen.blit(frame_surface, (0, 0))

        # Update the display
        pygame.display.flip()

        # Capture events
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Pygame QUIT event detected.")
                pygame.quit()
                raise SystemExit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("Escape key pressed. Exiting.")
                    pygame.quit()
                    raise SystemExit

        return keys

    def draw_refuel_zone_on_frame(self, frame):
        refuel_x, refuel_y = self.env.refuel_zone
        print(f"Drawing Refuel Zone at world coordinates: ({refuel_x}, {refuel_y})")

        frame_size = frame.shape[0]
        # Adjust `world_range` based on actual environment coordinates
        world_range = 50  # Example adjustment

        # Transform world coordinates to pixel coordinates
        pixel_x = int((refuel_x + world_range) / (2 * world_range) * frame_size)
        pixel_y = int((world_range - refuel_y) / (2 * world_range) * frame_size)

        print(f"Transformed to pixel coordinates: ({pixel_x}, {pixel_y})")

        radius = int(self.env.refuel_radius / (2 * world_range) * frame_size)
        radius = max(radius, 1)
        print(f"Refuel zone drawn with radius: {radius}")

        # Ensure pixel coordinates are within frame bounds
        pixel_x = np.clip(pixel_x, 0, frame_size - 1)
        pixel_y = np.clip(pixel_y, 0, frame_size - 1)

        # OpenCV uses BGR format; bright red is (0, 0, 255)
        cv2.circle(frame, (pixel_x, pixel_y), radius, (0, 0, 255), -1)


def get_action_from_keys(keys):
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if keys[pygame.K_q]:
        return "exit"
    if keys[pygame.K_r]:
        return "reset"

    # Steering
    if keys[pygame.K_LEFT]:
        action[0] = -0.25
    elif keys[pygame.K_RIGHT]:
        action[0] = 0.25
    else:
        action[0] = 0.0

    # Acceleration and Brake
    if keys[pygame.K_UP]:
        action[1] = 0.05
    else:
        action[1] = 0.0

    if keys[pygame.K_DOWN]:
        action[2] = 1.0
    else:
        action[2] = 0.0

    return action


# Initialize environment with wrappers
env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",
    max_episode_steps=1000,
)
env = FuelGaugeWrapper(env)
env = FuelGaugeRenderingWrapper(env)

# Run the environment
done = False
obs, info = env.reset(seed=3)

while not done:
    try:
        keys = env.render()
    except SystemExit:
        break  # Exit the loop if SystemExit is raised

    action = get_action_from_keys(keys)

    if isinstance(action, str):
        if action == "exit":
            print("Exit command received.")
            break
        elif action == "reset":
            obs, info = env.reset()
            continue

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Print fuel status and termination flags for debugging
    print(
        f"Fuel: {info.get('fuel', 'N/A')}, Terminated: {terminated}, Truncated: {truncated}"
    )

env.close()
pygame.quit()
