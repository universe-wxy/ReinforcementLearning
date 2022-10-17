from cmath import sqrt
import torch
import torch.distributions as D
import numpy as np

class agent:
    def __init__(self, K, const_flag):
        self.K = K
        self.const_flag = const_flag
        self.counts = torch.zeros(K)
        # metrics
        self.reward = torch.zeros(1)
        self.ba_count = 0
        self.rewards = []
        self.best_act = []
        self.timesteps = 0

    def get_action(self):
        raise NotImplementedError
    
    def update(self, a, r):
        raise NotImplementedError

    def update_metric(self):
        self.rewards.append(np.array(self.reward))
        self.best_act.append(np.array(self.ba_count/self.timesteps))
        
class EpsilonGreedy(agent):
    def __init__(self, K, const_flag):
        super(K,const_flag)
        
