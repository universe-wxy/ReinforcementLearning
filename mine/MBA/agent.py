from cmath import sqrt
import imp
import torch
import torch.distributions as D
import numpy as np

class agent:
    def __init__(self,K,const_flag):
        self.k=K
        self.const_flag=const_flag
        self.counts=torch.zeros(K)
        self.reward=torch.zeros(1)
        self.bestcount=0
        self.rewards=[]
        self.best_act=[]
        self.timesteps=0
    
    def get_action(self):
        raise NotImplementedError
    
    def update(self,a,r):
        raise NotImplementedError
    
    def update_metric(self):
        self.rewards.append(np.array(self.reward))
        self.best_act.append(np.array(self.bestcount/self.timesteps))
        
class EpsilonGreedy(agent):
    def __init__(self, K, const_flag,epsilon):
        super(K, const_flag)
        self.epsilon=epsilon
        self.Q=torch.zeros(K)
        self.counts=torch.zeros(K)
        
    def get_action(self):
        if torch.rand(1)<self.epsilon:
            return torch.randint(0,self.K,(1,))[0]
        else:
            return torch.max(self.Q,0)[1]
    
    def update(self, a, r):
        self.timesteps+=1
        self.reward+=r
        self.counts[a]+=1
        self.epsilon=self.epsilon*self.counts[a]/self.timesteps
        self.Q[a]+=(r-self.Q[a])/self.counts[a]
    
class UCB(agent):
    def __init__(self, K, const_flag):
        super(K, const_flag)
        self.u=torch.zeros(K)
        self.Q=torch.tensor(1,(K,))
        self.counts=torch.zeros(K)
        
    def get_action(self):
        return torch.max(self.Q+self.u)[1]
    
    def update(self, a, r):
        self.timesteps+=1
        self.u[a]=torch.sqrt(torch.log(-torch.tensor(self.timesteps))/2*self.counts[a]+1)
        self.reward+=r
        self.counts[a]+=1
        self.Q[a]+=(r-self.Q[a])/self.counts[a]