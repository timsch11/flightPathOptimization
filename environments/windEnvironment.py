import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class PlaneNavigationEnv(gym.Env):
    """Custom 2D plane navigation environment with wind and no fly zones in a 10x10 grid."""
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self, heuristic: bool = True):
        super(PlaneNavigationEnv, self).__init__()
        
        # render is not initalized yet
        self.render_initialized = False

        self.grid_size = 10  # New 10x10 grid
        self.max_steps = 100
        self.step_count = 0

        self.norm_const = (2* ((self.grid_size - 1) ** 2)) ** (1/2)
        
        # Define action and observation space
        # Actions: move in one of 4 directions (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        self.heuristic = heuristic
        
        # Observation: position (2), goal position (2), 
        # full wind vector field (10x10x2), grid map with obstacles (100), 
        # progress (1)
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            'goal': spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32),
            'wind_field': spaces.Box(low=-1.0, high=1.0, shape=(self.grid_size, self.grid_size, 2), dtype=np.float32),
            'grid_map': spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int32),
            'progress': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        # Wind field: vector for each cell in the grid
        # Random wind vectors with magnitude between 0 and 0.5 in random directions
        self.wind_magnitudes = np.random.uniform(0, 0.12, (self.grid_size, self.grid_size))
        self.wind_directions = np.random.uniform(0, 2*np.pi, (self.grid_size, self.grid_size))
        self.wind_field = np.zeros((self.grid_size, self.grid_size, 2))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                magnitude = self.wind_magnitudes[i, j]
                direction = self.wind_directions[i, j]
                self.wind_field[i, j, 0] = magnitude * np.cos(direction)  # x component
                self.wind_field[i, j, 1] = magnitude * np.sin(direction)  # y component

        # # add "jetstream"
        # for i in range(2, 7):
        #         self.wind_field[i, 6, 0] = 1
        #         self.wind_field[i, 6, 1] = 0
        
        # No-fly zones (grid cells marked as obstacles)
        self.grid_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.no_fly_zones = [(3, 3), (3, 4), (4, 3), (4, 4), (7, 1), (7, 2)]
        
        for x, y in self.no_fly_zones:
            self.grid_map[x, y] = 1  # Mark as obstacle
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset position and goal, ensuring they're not in no-fly zones
        while True:
            self.position = np.array([np.random.randint(0, self.grid_size), 
                                      np.random.randint(0, self.grid_size)])
            if self.grid_map[self.position[0], self.position[1]] == 0:
                break
                
        while True:
            self.goal = np.array([np.random.randint(0, self.grid_size), 
                                  np.random.randint(0, self.grid_size)])
            if (self.grid_map[self.goal[0], self.goal[1]] == 0 and 
                not np.array_equal(self.goal, self.position)):
                break
        
        self.step_count = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def _get_observation(self):
        return {
            'position': self.position.copy(),
            'goal': self.goal.copy(),
            'wind_field': self.wind_field.copy(),  # Include the entire wind field
            'grid_map': self.grid_map.copy(),
            'progress': np.array([self.step_count / self.max_steps])
        }

    def step(self, action):
        self.step_count += 1
        
        # Action mapping: 0: up, 1: right, 2: down, 3: left
        action_map = [
            np.array([0, 1]),   # up
            np.array([1, 0]),   # right
            np.array([0, -1]),  # down
            np.array([-1, 0])   # left
        ]
        
        # Get wind at current position
        wind = self.wind_field[self.position[0], self.position[1]]
        
        # Calculate new position with action and wind influence
        move_direction = action_map[action]
        
        # Wind effect (probabilistic - wind might push agent in its direction)
        wind_effect = np.zeros(2, dtype=np.int32)
        wind_magnitude = np.linalg.norm(wind)
        
        # Apply wind with probability based on magnitude
        if np.random.random() < wind_magnitude:
            if abs(wind[0]) > abs(wind[1]):  # Stronger horizontal wind
                wind_effect[0] = 1 if wind[0] > 0 else -1
            else:  # Stronger vertical wind
                wind_effect[1] = 1 if wind[1] > 0 else -1
        
        # Update position
        new_position = self.position + move_direction + wind_effect
        
        # Clip to ensure we stay on the grid
        new_position = np.clip(new_position, 0, self.grid_size - 1)
        
        # Check if new position is in a no-fly zone
        in_no_fly_zone = False
        if self.grid_map[new_position[0], new_position[1]] == 1:
            in_no_fly_zone = True
            # Don't move into no-fly zone
            new_position = self.position
        else:
            self.position = new_position
        
        # Reward calculation
        if self.heuristic:
            distance_to_goal = np.linalg.norm(self.goal - self.position)
            reward = -1.0 - distance_to_goal / self.norm_const  # Small penalty for each step to encourage efficiency                                                 
                                                                # distance_to_goal / self.norm_const is in [0, 1]

        else:
            reward = -1.0
        
        terminated = False
        truncated = False
        
        if in_no_fly_zone:
            reward = -100.0
        elif np.array_equal(self.position, self.goal):
            reward = 100.0
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True
        
        observation = self._get_observation()
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if not self.render_initialized:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.render_initialized = True
        
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        
        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, fill=False, edgecolor='gray'))
        
        # Draw no-fly zones
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid_map[i, j] == 1:
                    self.ax.add_patch(plt.Rectangle((i-0.5, j-0.5), 1, 1, color='red', alpha=0.5))
        
        # Draw wind field (arrows)
        X, Y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        U = self.wind_field[:, :, 0].T  # x component of wind
        V = self.wind_field[:, :, 1].T  # y component of wind
        
        # Plot arrows with length proportional to wind strength
        self.ax.quiver(X, Y, U, V, scale=5, color='skyblue')
        
        # Draw goal
        self.ax.plot(self.goal[0], self.goal[1], 'go', markersize=15, markeredgecolor='darkgreen')
        
        # Draw plane
        self.ax.plot(self.position[0], self.position[1], 'bo', markersize=15, markeredgecolor='darkblue')
        
        self.ax.set_title(f"Step {self.step_count} / {self.max_steps}")
        self.ax.grid(True)
        
        plt.pause(0.1)  # Pause to allow plot to update
        
    def close(self):
        if self.render_initialized:
            plt.ioff()
            plt.close()