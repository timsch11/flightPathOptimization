from .policy import Policy
import numpy as np
import torch


class GreedyPolicy(Policy):
    def __init__(self):
        pass

    def pickFromValue(self, actions, values):
        return actions[torch.argmax(values).item()]

    def pickFromDistribution(self, actions, probabilities):
        return actions[torch.argmax(probabilities).item()]
    