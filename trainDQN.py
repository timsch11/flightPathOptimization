from environments.windEnvironment import PlaneNavigationEnv
from rl.algorithms.dqn import DQN
from rl.policy.epsilongreedy import EpsilonGreedyPolicy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Ensure model cache directory exists
os.makedirs("modelCache", exist_ok=True)

# Training parameters
EPISODES = 500
MAX_STEPS = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPSILON_START = 1.0
EPSILON_DECAY = 0.998
EPSILON_MIN = 0.05
SAVE_INTERVAL = 100000
PRINT_INTERVAL = 100


def state_to_tensor(observation):
    """Convert dictionary observation to a flat tensor for DQN input"""
    # Extract all components from the observation dictionary
    position = torch.tensor(observation['position'], dtype=torch.float32)
    goal = torch.tensor(observation['goal'], dtype=torch.float32)
    wind_field = torch.tensor(observation['wind_field'], dtype=torch.float32)
    grid_map = torch.tensor(observation['grid_map'], dtype=torch.float32)
    progress = torch.tensor(observation['progress'], dtype=torch.float32)
    
    # Calculate direction vector from current position to goal
    direction = goal - position
    
    # Calculate Euclidean distance to goal
    distance = torch.norm(direction).unsqueeze(0)
    
    # Normalize direction for better learning
    direction_norm = direction / (distance + 1e-8)  # Avoid division by zero
    
    # Process wind field - we can flatten it or use a CNN approach
    # Here we'll flatten it for simplicity, but downsample first to reduce dimensionality
    flattened_wind = wind_field.flatten()
    
    # Flatten grid map (no-fly zones)
    flattened_grid = grid_map.flatten()
    
    # Concatenate all features into a single tensor
    return torch.cat([
        position,          # 2 values: x, y
        goal,              # 2 values: x, y
        direction_norm,    # 2 values: normalized direction vector
        distance,          # 1 value: distance to goal
        progress,          # 1 value: progress through episode
        flattened_wind,    # grid_size^2 * 2 values: complete wind field
        flattened_grid     # grid_size^2 values: no-fly zones
    ])

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define the environment
    env = PlaneNavigationEnv()
    
    # Calculate observation dimension
    observation_dict, _ = env.reset()
    observation_tensor = state_to_tensor(observation_dict)
    observation_dim = len(observation_tensor)
    
    print(f"Observation dimension: {observation_dim}")
    
    # nn
    model = torch.nn.Sequential(
        torch.nn.Linear(observation_dim, 256),  # Increased network size for larger state
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4)  # 4 actions (up, right, down, left)
    )
    
    # set up the optimizer and loss function
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create policy with high initial exploration
    policy = EpsilonGreedyPolicy(epsilon=EPSILON_START, decay=EPSILON_DECAY, min_epsilon=EPSILON_MIN, decay_step_interval=15)
    
    # Initialize DQN agent
    dqn = DQN(
        model=model,
        actionSpace=np.arange(4),  # 4 discrete actions
        policy=policy,
        replayBufferSize=BUFFER_SIZE,
        train=True,
        minBufferedActionsBeforeTraining=BATCH_SIZE,
        batchSize=BATCH_SIZE,
        optimizer=optimizer,
        criterion=criterion,
        recalibrationInterval=250,  # Update target network every 100 steps
        exponentialDecay=0.99  # Discount factor
    )
    
    # Metrics tracking
    rewards_history = []
    success_history = []  # 1 for successful episodes, 0 for failures
    steps_history = []
    steps_opt_surplus = []
    
    # Train for specified number of episodes
    for episode in tqdm(range(EPISODES), desc="Training"):
        # Reset environment
        observation_dict, _ = env.reset()
        observation = state_to_tensor(observation_dict)
        
        episode_reward = 0
        step_count = 0

        # calculate manhattan distance
        opt_steps = abs(observation_dict['position'][0] - observation_dict['goal'][0]) + abs(observation_dict['position'][1] - observation_dict['goal'][1])
        
        # Run episode
        for step in range(MAX_STEPS):
            # Select action
            action = dqn.predict(observation)
            
            # Execute action
            new_observation_dict, reward, terminated, truncated, _ = env.step(action)
            new_observation = state_to_tensor(new_observation_dict)
            
            # Store transition in replay buffer
            dqn.bufferLastAction(reward, new_observation)
            
            # Update current observation
            observation = new_observation
            episode_reward += reward
            step_count += 1
            
            # Render occasionally
            # if episode % PRINT_INTERVAL == 0 and episode > 0:
            #     env.render()
            
            # If episode ended
            if terminated or truncated:
                break
        
        # Train model after each episode
        if episode >= 5:  # Allow some episodes for buffer filling
            dqn.train_model(batches=20)  # Train on multiple batches
        
        # Record metrics
        rewards_history.append(episode_reward)
        success_history.append(1 if reward > 10 else 0)  # Reward > 10 means we reached goal
        steps_history.append(step_count)
        steps_opt_surplus.append(step_count - opt_steps)
        
        # Print progress
        if episode % PRINT_INTERVAL == 0:
            recent_rewards = np.mean(rewards_history[-PRINT_INTERVAL:])
            recent_successes = np.mean(success_history[-PRINT_INTERVAL:]) * 100
            recent_steps = np.mean(steps_history[-PRINT_INTERVAL:])
            print(f"Episode {episode}/{EPISODES}, Avg Reward: {recent_rewards:.2f}, Success Rate: {recent_successes:.1f}%, Avg Steps: {recent_steps:.1f}, Epsilon: {policy.currentEpsilon:.2f}")
        
        # Save model periodically
        if episode % SAVE_INTERVAL == 0 and episode > 0:
            torch.save(model.state_dict(), f"modelCache/grid_dqn_model_{episode}")
    
    # Save final model
    torch.save(model.state_dict(), "modelCache/grid_ddqn_model_jetstream_V1000")
    
        
    # Create the plot with subplots side by side
    plt.style.use('seaborn-v0_8')  # Use a clean, modern style
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(16, 8))

    window_size = 50
    # Calculate moving average
    moving_avg = []
    for i in range(len(steps_opt_surplus)):
        if i < window_size - 1:
            # For early points, use all available data
            avg = np.mean(steps_opt_surplus[:i+1])
        else:
            # Use last 50 items
            avg = np.mean(steps_opt_surplus[i-window_size+1:i+1])
        moving_avg.append(avg)


    # First subplot: Line plots for steps_opt_surplus and steps_history
    x = range(len(steps_opt_surplus))  # Assuming all lists have same length
    ax1.plot(x, moving_avg, 
            color='#2E8B57', 
            linewidth=2.5, 
            marker='o', 
            markersize=4,
            label=f'Path length difference compared to mahattan distance, moving average (window={window_size})',
            alpha=0.8)

    # Customize first subplot
    ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax1.set_title('Path length progress', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax1.set_facecolor('#F8F9FA')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#666666')
    ax1.spines['bottom'].set_color('#666666')

    # Second subplot: Moving average of success_history
    # Calculate moving average
    moving_avg = []
    for i in range(len(success_history)):
        if i < window_size - 1:
            # For early points, use all available data
            avg = np.mean(success_history[:i+1])
        else:
            # Use last 25 items
            avg = np.mean(success_history[i-window_size+1:i+1])
        moving_avg.append(avg)

    ax2.plot(x, moving_avg, 
            color='#4A90E2', 
            linewidth=2.5, 
            marker='^', 
            markersize=4,
            label=f'P(Success) moving average (window={window_size})',
            alpha=0.8)

    # Customize second subplot
    ax2.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Success Rate', fontsize=14, fontweight='bold')
    ax2.set_title('Success probability moving average', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax2.set_facecolor('#F8F9FA')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#666666')
    ax2.spines['bottom'].set_color('#666666')

    # Set y-axis limits for success rate (0 to 1)
    ax2.set_ylim(0, 1)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot as a high-quality image
    plt.savefig('flight_optimization_plot.png', 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')

    # Display the plot
    plt.show()

    # Close environment
    env.close()

if __name__ == '__main__':
    main()