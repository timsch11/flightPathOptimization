from environments.basicEnvironment import PlaneNavigationEnv
from rl.algorithms.dqn import DQN
from rl.policy.greedy import GreedyPolicy
from rl.policy.epsilongreedy import EpsilonGreedyPolicy
import torch
import numpy as np


STEP_LIMIT = 180


def main():

    torch.set_default_tensor_type(torch.FloatTensor)
    # Define the environment
    env = PlaneNavigationEnv()
    
    # Get observation and action dimensions
    observation_dim = env.observation_space.shape[0]  # 2D position
    action_dim = env.action_space.shape[0]  # 2D movement vector

    # Create a discrete action space for DQN
    # We'll map these discrete actions to continuous movements
    discrete_actions = np.array([
        [1.0, 0.0],     # Right
        [-1.0, 0.0],    # Left
        [0.0, 1.0],     # Up
        [0.0, -1.0],    # Down
        [0.7, 0.7],     # Up-Right
        [-0.7, 0.7],    # Up-Left
        [0.7, -0.7],    # Down-Right
        [-0.7, -0.7],   # Down-Left
        [0.4, 0.0],     # Slight Right
        [-0.4, 0.0],    # Slight Left
        [0.0, 0.4],     # Slight Up
        [0.0, -0.4],    # Slight Down
    ], dtype=np.float32)
    
    # init model - adjust input and output dimensions
    model = torch.nn.Sequential(
        torch.nn.Linear(observation_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(), 
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, len(discrete_actions))
    )

    model.load_state_dict(torch.load("modelCache/model_C0"))
    
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Create policy
    policy = EpsilonGreedyPolicy(epsilon=0.4, decay=0.99, min_epsilon=0.15)
    
    # Initialize DQN agent
    dqn = DQN(
        model=model, 
        actionSpace=np.arange(len(discrete_actions)),  # Action indices
        policy=policy,
        replayBufferSize=10000,
        train=True,
        batchSize=50,
        minBufferedActionsBeforeTraining=500,
        optimizer=optimizer,
        criterion=criterion
    )
    
    for scenario in range(5000):

        print(scenario)
        
        # Initialize environment and get initial state
        observation, info = env.reset()
        total_reward = 0

        for episode in range(STEP_LIMIT):
            #env.render()
            
            # Convert observation to tensor for DQN
            state_tensor = torch.tensor(observation, dtype=torch.float32)
            
            # Use DQN to predict action index
            action_idx = dqn.predict(state_tensor)
            
            # Map discrete action index to continuous action vector
            continuous_action = discrete_actions[action_idx]
            
            # Execute action in environment
            new_observation, reward, terminated, truncated, info = env.step(continuous_action)
            total_reward += reward
            
            # Buffer the last action for training
            new_state_tensor = torch.tensor(new_observation, dtype=torch.float32)
            dqn.bufferLastAction(reward, new_state_tensor)
            
            # Print debug information

            #print(f"Step {episode}: Action={continuous_action}, Reward={reward:.2f}, Total={total_reward:.2f}")
            #print(f"Position: {new_observation}")

            # Update current observation
            observation = new_observation
            
            # if destination is reached or scenario is ended
            if terminated or truncated or episode == STEP_LIMIT:
                #print(f"Episode {episode} finished! Final position: {observation}, Total reward: {total_reward:.2f}")
                
                # Train the model if we have enough samples
                dqn.train_model(batches=10)
                    
                # Reset for next episode
                observation, info = env.reset()
                total_reward = 0
                
                if episode >= STEP_LIMIT:  # Last episode
                    break
                continue

    env.close()

    print("saving model...")
    torch.save(dqn.predictorNetwork.state_dict(), "modelCache/model_C0")


if __name__ == '__main__':
    main()