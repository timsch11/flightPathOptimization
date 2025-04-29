from rl.policy.policy import Policy
import torch
import numpy as np


class REINFORCE:
    def __init__(self, model: torch.nn.modules.container, actionSpace: np.ndarray, policy: Policy, train: bool = False, batchSize: int = 5, exponentialDecay: float = 0.9, optimizer: torch.optim = None):
        # store model reference
        self.network = model

        # store action space and used 'decision' policy
        self.action_space = actionSpace
        self.policy = policy

        # store training hyperparameters
        self.isTraining = train
        self.batchSize = batchSize
        self.bufferedActions = 0
        self.exponentialDecay = exponentialDecay

        # buffer episode (state, action, reward, new state)
        self.buffer = list()

        # optimizer and loss function (only for training)
        self.optimizer = optimizer

        if train:
            if optimizer is None:
                raise TypeError("For training you need to pass an optimizer")
            
        # init state and action trackers
        self.lastState = None
        self.lastAction = None

    def setTraining(self, value: bool):
        self.isTraining = value

    def predict(self, state: np.ndarray) -> np.ndarray:
        #predicts the best action to take in state <state> and returns this action
        
        # predict q values, no training yet -> no gradients needed
        with torch.no_grad():
            valuesPrediction = self.network(state)

        # pick action based on policy
        action = self.policy.pickFromDistribution(self.action_space, valuesPrediction)

        # store state action pair
        if self.isTraining:
            self.lastState = state
            self.lastAction = action

        return action
    
    def bufferLastAction(self, reward: float, newState: np.ndarray):
        #buffers the last state, action pair and its corresponding reward and resulting state

        # check if training is activated and at least one prediction has already been made
        if not self.isTraining:
            raise RuntimeError("Training is not activated")
        
        elif self.lastAction is None or self.lastState is None:
            raise RuntimeError("No previous prediction has been made")
            
        # append to buffer and increment its counter
        self.buffer.append([self.lastState, self.lastAction, reward, newState])

        self.bufferedActions += 1

    def train_model(self, batches: int = 1):
        if self.bufferedActions == 0:
            return  # Nothing to train on

        ### calculate return
        buffer_returns = [0] * len(self.buffer)
        
        # Calculate returns with exponential decay (backward)
        cumulative_reward = 0
        for i in range(self.bufferedActions - 1, -1, -1):
            cumulative_reward = self.buffer[i][2] + (self.exponentialDecay * cumulative_reward)
            buffer_returns[i] = cumulative_reward
        
        # randomly sample <batches> batches
        for _ in range(batches):
            # cache batch size
            batch_size = min(self.batchSize, self.bufferedActions)  # in case buffer is smaller than batch size

            ### randomly sample from replay buffer
            # generate batch_size random indices to sample from buffer
            batch_indices = np.random.choice(range(len(self.buffer)), batch_size, replace=False)
            
            # construct batch from random indices
            batch = [self.buffer[i] for i in batch_indices]

            # create tensors for states and corresponding returns
            states = torch.stack([torch.as_tensor(sample[0], dtype=torch.float32) for sample in batch])
            actions = torch.tensor([sample[1] for sample in batch], dtype=torch.int64)
            returns = torch.tensor([buffer_returns[i] for i in batch_indices], dtype=torch.float32)
                
            # Get action probabilities from the network
            action_prob_dist = self.network(states)
            
            # Get the log probabilities of the actions that were actually taken
            action_log_probs = torch.log(torch.nn.functional.softmax(action_prob_dist, dim=1))
            selected_action_log_probs = action_log_probs[range(batch_size), actions]
            
            ### objective: maximize expected return
            # Negative sign because we want to maximize the objective (and optimizers minimize by default)
            loss = -torch.mean(selected_action_log_probs * returns)
            
            ### optimization
            # reset gradients
            self.optimizer.zero_grad()

            # compute current gradients
            loss.backward()

            # gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            # optimization step
            self.optimizer.step()

        del self.buffer
        self.buffer = list()
        self.bufferedActions = 0
