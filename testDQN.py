from environments.windEnvironment import PlaneNavigationEnv  # Updated import
from rl.algorithms.dqn import DQN
from rl.policy.epsilongreedy import EpsilonGreedyPolicy
from rl.policy.greedy import GreedyPolicy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# Ensure model cache directory exists
os.makedirs("modelCache", exist_ok=True)

# Training parameters
EPISODES = 1000
MAX_STEPS = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPSILON_START = 0.05
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
SAVE_INTERVAL = 10000
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
    
    # Neural network model
    model = torch.nn.Sequential(
        torch.nn.Linear(observation_dim, 1024),  # Increased network size for larger state
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4)  # 4 actions (up, right, down, left)
    )

    model.load_state_dict(torch.load("modelCache/grid_dqn_model_final"))
    
    # Set up the optimizer and loss function
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create policy with high initial exploration
    policy = EpsilonGreedyPolicy(epsilon=EPSILON_START, decay=EPSILON_DECAY, min_epsilon=EPSILON_MIN)
    policy = GreedyPolicy()

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
        recalibrationInterval=250,  # Update target network every 250 steps
        exponentialDecay=0.99  # Discount factor
    )
    
    # Metrics tracking
    rewards_history = []
    success_history = []  # 1 for successful episodes, 0 for failures
    steps_history = []
    
    # Train for specified number of episodes
    for episode in tqdm(range(EPISODES), desc="Training"):
        # Reset environment
        observation_dict, _ = env.reset()
        observation = state_to_tensor(observation_dict)
        
        episode_reward = 0
        step_count = 0
        
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
            env.render()
            
            # If episode ended
            if terminated or truncated:
                break
        
        # Train model after each episode
        if episode >= 5:  # Allow some episodes for buffer filling
            dqn.train_model(batches=10)  # Train on multiple batches
        
        # Record metrics
        rewards_history.append(episode_reward)
        success_history.append(1 if reward > 10 else 0)  # Reward > 10 means we reached goal
        steps_history.append(step_count)
        
        # Print progress
        if episode % PRINT_INTERVAL == 0:
            recent_rewards = np.mean(rewards_history[-PRINT_INTERVAL:])
            recent_successes = np.mean(success_history[-PRINT_INTERVAL:]) * 100
            recent_steps = np.mean(steps_history[-PRINT_INTERVAL:])
            print(f"Episode {episode}/{EPISODES}, Avg Reward: {recent_rewards:.2f}, Success Rate: {recent_successes:.1f}%, Avg Steps: {recent_steps:.1f}")
        
        # Save model periodically
        if episode % SAVE_INTERVAL == 0 and episode > 0:
            torch.save(model.state_dict(), f"modelCache/grid_dqn_model_{episode}")
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    # Calculate moving average success rate
    window_size = 50  # Fixed indentation here
    success_rate = [np.mean(success_history[max(0, i-window_size):i+1]) for i in range(len(success_history))]
    plt.plot(success_rate)
    plt.title('Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim([0, 1])
    
    plt.subplot(1, 3, 3)
    plt.plot(steps_history)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    # Close environment
    env.close()

if __name__ == '__main__':
    main()