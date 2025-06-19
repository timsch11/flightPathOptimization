from .policy import Policy
import numpy as np
import random
import torch


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float = 0.3, decay: float = 0.99, min_epsilon: float = 0.05, decay_step_interval: int = 5, autoDecay: bool = True):
        self.eps = epsilon
        self.currentEpsilon = epsilon * (1 / decay)
        self.decay = decay
        self.minEpsilon = min_epsilon
        self.decayStepInterval = decay_step_interval
        self.autoDecay = autoDecay
        self.steps = 0

    def pickFromValue(self, actions, values):
        if self.autoDecay and self.steps % self.decayStepInterval == 0:
            self.currentEpsilon *= self.decay

        self.steps += 1
        
        if random.random() < max(self.currentEpsilon, self.minEpsilon):
            return actions[random.randint(0, len(actions) - 1)]
        else:
            return actions[torch.argmax(values).item()]
        
    def epsStep(self):
        self.currentEpsilon = max(self.currentEpsilon * self.decay, self.minEpsilon)


    def pickFromDistribution(self, actions, probabilities):
        if self.autoDecay and self.steps % self.decayStepInterval == 0:
            self.currentEpsilon *= self.decay

        self.steps += 1
        
        if random.random() < max(self.currentEpsilon, self.minEpsilon):
            return actions[random.randint(0, len(actions) - 1)]
        else:
            return actions[torch.argmax(probabilities).item()]
