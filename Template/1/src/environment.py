import torch
import torch.distributions as D

class MAB:
    def __init__(self, K, const_flag, dist):
        self.K = K
        self.const_flag = const_flag
        self.dist = dist
        # dist
        if self.dist == "Normal":
            self.q_a = torch.randn(self.K) #size [self.K] std_normal N(0,1)
            self.probs = D.Normal(self.q_a,1)
        elif self.dist == "Bernoulli":
            self.q_a = torch.rand(self.K)
            self.probs = D.Bernoulli(self.q_a)
        else:
            raise TypeError
        # 判断是否是最优动作
        self.best_idx = torch.max(self.q_a,0)[1]
        # 
        if not self.const_flag:
            self.probs_delta = D.Normal(torch.zeros(self.K),0.1) 
        print("生成了一个","平稳" if self.const_flag else "非平稳","{K}臂赌博机".format(K=self.K),sep="")
        print("奖励概率分布R为均值由","标准正态" if self.dist == "Normal" else "伯努利","分布生成的正态分布",sep="")
        print("初始奖励分布均值为{loc}".format(loc=self.q_a))
        print("初始最优动作为{best_idx}".format(best_idx=self.best_idx))

    def step(self, a):
        if not self.const_flag:
            if self.dist == "Bernoulli":
                raise TypeError
            self.q_a += self.probs_delta.sample()
            self.best_idx = torch.max(self.q_a,0)[1]
        r = self.probs.sample()[a]
        if a == self.best_idx:
            return r,1
        else:
            return r,0