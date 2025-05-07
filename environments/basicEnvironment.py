import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class PlaneNavigationEnv(gym.Env):
    """Custom 2D plane navigation environment with wind and no fly zones."""
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self):
        super(PlaneNavigationEnv, self).__init__()
        
        # render is not initalized yet
        self.render_initialized = False

        self.size = 100  # Size of the 2D world
        self.max_steps = 200
        self.step_count = 0
        
        # Wind field: small vectors
        #np.random.seed
        self.wind_field = np.random.uniform(-0.5, 0.5, (self.size, self.size, 2))
        
        # No-fly zones
        self.no_fly_zones = [((30, 40), (30, 40)), ((60, 70), (10, 20))]

        # Create no-fly zone binary map
        self.no_fly_map = np.zeros((self.size, self.size), dtype=np.float32)
        for (x_range, y_range) in self.no_fly_zones:
            self.no_fly_map[x_range[0]:x_range[1]+1, y_range[0]:y_range[1]+1] = 1.0
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # New observation space includes:
        # - Current position (2)
        # - Goal position (2)
        # - Direction to goal (2)
        # - Normalized distance to goal (1)
        # - Progress through episode (1)
        # - Wind field (flattened 2D array with 2 channels = 100×100×2)
        # - No-fly zone map (flattened 2D array = 100×100)
        wind_field_size = self.size * self.size * 2
        no_fly_zone_size = self.size * self.size
        
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0.0, high=self.size, shape=(2,), dtype=np.float32),
            "goal": spaces.Box(low=0.0, high=self.size, shape=(2,), dtype=np.float32),
            "direction": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "distance": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "progress": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "wind_field": spaces.Box(low=-0.5, high=0.5, shape=(self.size, self.size, 2), dtype=np.float32),
            "no_fly_map": spaces.Box(low=0.0, high=1.0, shape=(self.size, self.size), dtype=np.float32)
        })
        
        # Goal
        self.goal = np.random.rand(2) * 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.position = np.random.rand(2) * 100
        self.goal = self.goal = np.random.rand(2) * 100
        
        self.step_count = 0

        distance_to_goal = np.linalg.norm(self.goal - self.position)
        direction_to_goal = (self.goal - self.position) / (distance_to_goal + 1e-8)  # Normalized direction vector
        
        observation = [
            self.position.copy(),
            self.goal.copy(),
            direction_to_goal,
            np.array([distance_to_goal / self.size], dtype=np.float32),
            np.array([self.step_count / self.max_steps], dtype=np.float32),
            self.wind_field.copy(),
            self.no_fly_map.copy()
        ]
        
        info = {}
        
        return observation, info

    def step(self, action):
        self.step_count += 1
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Get wind at the current position
        x_idx = np.clip(int(self.position[0]), 0, self.size - 1)
        y_idx = np.clip(int(self.position[1]), 0, self.size - 1)
        wind = self.wind_field[x_idx, y_idx]
        
        # Update position
        self.position += action + wind
        self.position = np.clip(self.position, 0, self.size)
        
        # Check no-fly zones
        in_no_fly_zone = False
        for (x_range, y_range) in self.no_fly_zones:
            if x_range[0] <= self.position[0] <= x_range[1] and y_range[0] <= self.position[1] <= y_range[1]:
                in_no_fly_zone = True
                break
        
        # Reward calculation
        distance_to_goal = np.linalg.norm(self.goal - self.position)
        reward = -distance_to_goal * 0.01
        
        terminated = False
        truncated = False
        
        if in_no_fly_zone:
            reward = -100.0
            terminated = True
        elif distance_to_goal < 5.0:
            reward = 100.0
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        direction_to_goal = (self.goal - self.position) / (distance_to_goal + 1e-8)  # Normalized direction vector
        
        observation = [
            self.position.copy(),
            self.goal.copy(),
            direction_to_goal,
            np.array([distance_to_goal / self.size], dtype=np.float32),
            np.array([self.step_count / self.max_steps], dtype=np.float32),
            self.wind_field.copy(),
            self.no_fly_map.copy()
        ]
        
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if not self.render_initialized:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.render_initialized = True
        
        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        
        # Draw no-fly zones
        for (x_range, y_range) in self.no_fly_zones:
            self.ax.fill_betweenx(y=[y_range[0], y_range[1]], x1=x_range[0], x2=x_range[1], color='red', alpha=0.5)
        
        # Draw goal
        self.ax.plot(self.goal[0], self.goal[1], 'go', markersize=10)
        
        # Draw plane
        self.ax.plot(self.position[0], self.position[1], 'bo')
        
        self.ax.set_title(f"Step {self.step_count}")
        
        plt.pause(0.01)  # Tiny pause to allow plot to update
        
    def close(self):
        if self.render_initialized:
            plt.ioff()
            plt.close()
