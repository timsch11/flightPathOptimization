from rl.policy.policy import Policy
import torch
import numpy as np
import copy

from typing import Callable, Any
from collections import deque


class DQN:
    def __init__(self, model: torch.nn.modules.container, actionSpace: np.ndarray, policy: Policy, replayBufferSize: int, train: bool = False, recalibrationInterval: int = 1000, minBufferedActionsBeforeTraining: int = 250, batchSize: int = 5, exponentialDecay: float = 0.9, optimizer: torch.optim = None, criterion: Callable[[Any, Any], torch.Tensor] = None):
        # initalize prediction and target networks
        self.predictorNetwork = model
        self.targetNetwork = copy.deepcopy(model)

        # store action space and used policy
        self.action_space = actionSpace
        self.policy = policy

        # initalize replay buffer
        self.replayBufferSize = replayBufferSize
        self.replayBuffer = ReplayBuffer(capacity=replayBufferSize)    # list of [state, action, reward, new state] pairs
        self.minBufferedActionsBeforeTraining = minBufferedActionsBeforeTraining
        self.bufferedActions = 0

        # store training hyperparameters
        self.isTraining = train
        self.batchSize = batchSize
        self.recalibrationInterval = recalibrationInterval
        self.exponentialDecay = exponentialDecay

        # optimizer and loss function (only for training)
        self.optimizer = optimizer
        self.criterion = criterion

        if train:
            if optimizer is None:
                raise TypeError("For training you need to pass an optimizer")
            
            if criterion is None:
                raise TypeError("For training you need to pass a loss function")
            
        # init state and action trackers
        self.samples = 0

        self.lastState = None
        self.lastAction = None

    def setTraining(self, value: bool):
        self.isTraining = value

    def predict(self, state: np.ndarray) -> np.ndarray:
        """predicts the best action to take in state <state> and returns this action"""
        
        # predict q values, no training yet -> no gradients needed
        with torch.no_grad():
            valuesPrediction = self.predictorNetwork(state)

        # pick action based on policy
        action = self.policy.pickFromValue(self.action_space, valuesPrediction)

        # store state action pair
        if self.isTraining:
            self.lastState = state
            self.lastAction = action

        return action
    
    def bufferLastAction(self, reward: float, newState: np.ndarray):
        """buffers the last state, action pair and its corresponding reward and resulting state"""

        # check if training is activated and at least one prediction has already been made
        if not self.isTraining:
            raise RuntimeError("Training is not activated")
        
        elif self.lastAction is None or self.lastState is None:
            raise RuntimeError("No previous prediction has been made")
            
        # append to buffer and increment its counter
        self.replayBuffer.push([self.lastState, self.lastAction, reward, newState])
        self.bufferedActions += 1

    def recalibrate(self):
        """overrides target network parameters with predictor networks weights"""
        self.targetNetwork.load_state_dict(self.predictorNetwork.state_dict())  # simply copy the state dict, more efficient than another deepcopy

    def train_model(self, batches: int = 1):

        # only start training if we have a useful amount of samples bufferd
        if self.bufferedActions < self.minBufferedActionsBeforeTraining:
            return
        
        # randomly sample <batches> batches
        for batch in range(batches):
            # cache length of the buffer and batch size
            bufferLength = min(self.bufferedActions, self.replayBufferSize)
            batch_size = min(self.batchSize, bufferLength)  # in case buffer is smaller than batch size

            ### randomly sample from replay buffer
            # generate <batches> random indeces to construct a randomly picked batch
            batch_indices = np.random.choice(bufferLength, batch_size, replace=False) 
            
            # construct batch from random indeces
            batch = [self.replayBuffer.buffer[i] for i in batch_indices]

            # create tensors for state action reward and next state
            states = torch.stack([sample[0] for sample in batch])
            actions = torch.tensor([sample[1] for sample in batch], dtype=torch.int64)
            rewards = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
            next_states = torch.stack([sample[3] for sample in batch])
            
            # Recalibrate if necessary
            if self.samples % self.recalibrationInterval == 0:
                self.recalibrate()
                
            # calculate current q val again because we need the computational graph for backpropagation
            current_q_values = self.predictorNetwork(states)

            # Get q values for the actions that were taken
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            ### Calculate target q values using the target network
            # Don't compute unnecessary gradients
            with torch.no_grad():  
                # Get max q value from target network for next states
                next_q_values = self.targetNetwork(next_states).max(1)[0]
                target_q_values = rewards + self.exponentialDecay * next_q_values
            
            ### loss
            # compute loss
            loss = self.criterion(current_q_values, target_q_values)
            
            ### optimization
            # reset gradients
            self.optimizer.zero_grad()

            # compute current gradients
            loss.backward()

            # gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.predictorNetwork.parameters(), max_norm=1.0)
            
            # optimization step
            self.optimizer.step()
            
            # Increment sample counter
            self.samples += batch_size
        
        # print(f"Training loss: {loss.item():.4f}")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, item):
        self.buffer.append(item)