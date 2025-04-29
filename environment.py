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
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=self.size, shape=(10,), dtype=np.float32)
        
        # Wind field: small vectors
        self.wind_field = np.random.uniform(-0.5, 0.5, (self.size, self.size, 2))
        
        # No-fly zones
        self.no_fly_zones = [((30, 40), (30, 40)), ((60, 70), (10, 20))]
        
        # Goal
        self.goal = np.random.rand(2) * 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.position = np.random.rand(2) * 100
        self.goal = self.goal = np.random.rand(2) * 100
        
        self.step_count = 0

        distance_to_goal = np.linalg.norm(self.goal - self.position)

        wind_at_position = self.wind_field[10, 10]
        direction_to_goal = (self.goal - self.position) / (distance_to_goal + 1e-8)  # Normalized direction vector
        
        observation = np.concatenate([
            self.position.copy(),              # Current position (2)
            self.goal.copy(),                  # Goal position (2)
            wind_at_position,                  # Current wind vector (2)
            direction_to_goal,                 # Direction to goal (2)
            [distance_to_goal / self.size],    # Normalized distance to goal (1)
            [self.step_count / self.max_steps] # Progress through episode (1)
        ])
        
        # observation = np.concatenate([self.position.copy(), self.goal.copy()])
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

        wind_at_position = self.wind_field[x_idx, y_idx]
        direction_to_goal = (self.goal - self.position) / (distance_to_goal + 1e-8)  # Normalized direction vector
        
        observation = np.concatenate([
            self.position.copy(),              # Current position (2)
            self.goal.copy(),                  # Goal position (2)
            wind_at_position,                  # Current wind vector (2)
            direction_to_goal,                 # Direction to goal (2)
            [distance_to_goal / self.size],    # Normalized distance to goal (1)
            [self.step_count / self.max_steps] # Progress through episode (1)
        ])
        
        #observation = np.concatenate([self.position.copy(), self.goal.copy()])
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
