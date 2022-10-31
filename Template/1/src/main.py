from agent import EpsilonGreedy,UCB,ThompsonSampling
from environment import MAB
import argparse
import matplotlib.pyplot as plt
import torch

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_arm",default=10,type=int)
    parser.add_argument("--const_flag",default=False,type=bool)
    parser.add_argument("--num_steps",default=1000,type=int)
    parser.add_argument("--dist",default="Normal",type=str,help="Normal or Bernoulli")
    parser.add_argument("--seed",default=None,type=int)
    return parser.parse_args() 
        
def plot_metrics(agents,agent_names):
    plt.figure(1,figsize=(720,480))
    # rewards
    plt.subplot(1,2,1)
    for i,agent in enumerate(agents):
        time_list = range(len(agent.rewards))
        plt.plot(time_list,agent.rewards,label=agent_names[i])
    plt.xlabel('Time steps')
    plt.ylabel('rewards')
    plt.legend()
    # best act
    plt.subplot(1,2,2)
    for i,agent in enumerate(agents):
        time_list = range(len(agent.best_act))
        plt.plot(time_list,agent.best_act,label=agent_names[i])
    plt.xlabel('Time steps')
    plt.ylabel('best action percentage')
    plt.show()

def main(opt):
    env = MAB(opt.k_arm,opt.const_flag,opt.dist)

    # # 伯努利定常环境演示
    # agent_names = ("EpsilonGreedy", "UCB", "ThompsonSampling")
    # agents = (eval(agent_names[0])(opt.k_arm,opt.const_flag,0.01),\
    #         eval(agent_names[1])(opt.k_arm,opt.const_flag,0.1),\
    #         eval(agent_names[2])(opt.k_arm,opt.const_flag))

    # # 正态定常环境演示
    # agent_names = ("EpsilonGreedy", "UCB","ThompsonSamplingNormal")
    # agents = (eval(agent_names[0])(opt.k_arm,opt.const_flag,0.01),\
    #         eval(agent_names[1])(opt.k_arm,opt.const_flag,0.1),\
    #         eval(agent_names[2])(opt.k_arm,opt.const_flag))

    # 正态时变环境演示
    agent_names = ("EpsilonGreedy", "UCB","ThompsonSampling")
    agents = (eval(agent_names[0])(opt.k_arm,False,0.01),\
            eval(agent_names[1])(opt.k_arm,True,0.01),
            eval(agent_names[2])(opt.k_arm,opt.const_flag))
    
    def run(num_steps):
        for i,agent in enumerate(agents):
            for _ in range(num_steps):
                a = agent.get_action()
                r,ba = env.step(a)
                agent.update(a,r)
                if ba:
                    agent.bestcount += 1 
                agent.update_metric()
            print("{}:{}".format(agent_names[i],agent.rewards[-1]))
    run(opt.num_steps)
    plot_metrics(agents,agent_names)

if __name__ == "__main__":
    opt = parse_opt()
    if opt.seed:
        torch.manual_seed(opt.seed)
    main(opt)