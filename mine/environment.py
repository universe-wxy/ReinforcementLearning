from pip import main
import torch 
import torch.distributions as D

class MAB:
    def __init__(self,K,const_flag,dist):
        self.K=K
        self.const_flag=const_flag
        self.dist=dist
        if dist=="Normal":
            self.q_a=torch.randn(self.K)
            self.probs=D.normal(self.q_a,1)
        elif dist=="Bernoulli":
            self.q_a=torch.rand(self.K)
            self.probs=D.Bernoulli(self.q_a)
        else:
            raise TypeError
        self.best_idx=torch.max(self.q_a,0)[1]
        if not self.const_flag:
            self.probs_delta=D.Normal(torch.zeros(self.K),0.1)
        print("生成了一个","定常" if self.const_flag else "实变","的{}臂赌博机".format(self.K))
        print("初始奖励的分布均值为:{}".format(self.q_a))
        
    def step(self,a):
        if not self.const_flag:
            if self.dist=="Bernoulli":
                raise TypeError
            self.q_a+=self.probs_delta.sample()
            self.best_idx=torch.max(self.q_a,0)[1]
        r=self.probs.sample()[a]
        if a== self.best_idx:
            return r,1
        else:
            return r,0