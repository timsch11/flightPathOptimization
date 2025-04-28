from .policy import Policy
import numpy as np
import random
import torch


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.3, decay: float = 0.99, min_epsilon: float = 0.05):
        self.eps = epsilon
        self.currentEpsilon = epsilon * (1 / decay)
        self.decay = decay
        self.minEpsilon = min_epsilon

    def pickFromValue(self, actions, values):
        self.currentEpsilon *= self.decay
        if random.random() < max(self.currentEpsilon, self.minEpsilon):
            return actions[random.randint(0, len(actions) - 1)]
        else:
            return actions[torch.argmax(values).item()]


    def pickFromDistribution(self, actions, probabilities):
        pass