import torch
import gym
import torch.optim as optim
import torch.multiprocessing as mp 
import argparse 
import numpy as np
from tensorboardX import SummaryWriter
import datetime

from worker import Worker
from network import Net_Simple


# NUM_PROCESS = mp.cpu_count()

# Utils
def get_date():
    return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

def run_i_worker(worker):
    return worker.run()

class A2CAgent:
    def __init__(self, gamma, env_list, lr, max_episode, NUM_PROCESS):
        self.gamma = gamma
        self.max_episode = max_episode
        self.global_actor_net = Net_Simple(outputs = 2)
        self.global_critic_net = Net_Simple(outputs = 1)
        self.global_actor_net.share_memory()
        self.global_critic_net.share_memory()

        self.global_actor_optim = optim.Adam(self.global_actor_net.parameters(), lr=lr)
        self.global_critic_optim = optim.Adam(self.global_critic_net.parameters(), lr=lr)

        self.workers = []

        for i in range(NUM_PROCESS):
            self.workers.append(Worker(i, self.gamma, env_list[i], \
                self.global_actor_net, self.global_critic_net, \
                self.global_actor_optim, self.global_critic_optim))

        self.writer = SummaryWriter(logdir='./log/log_{}'.format(get_date()))

    def train(self, ):
        pool = mp.Pool(processes = len(self.workers))
        for i in range(self.max_episode):
            reward = pool.map(run_i_worker, self.workers)
            print("Episode {}, Reward {}".format(i, np.mean(reward)))
            self.writer.add_scalar("reward", np.mean(reward), i)
        pool.close()
        self.writer.close()

if __name__ == "__main__":

    # Initialization
    parser = argparse.ArgumentParser(description='A2C Policy Gradient')
    parser.add_argument('-e', '--episode', type=int, help='number of episode you want to train', default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='discount coefficient', default = 0.95)
    parser.add_argument('-l', '--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('-p', '--process', type=int, help='number of process', default=4)
    args = vars(parser.parse_args())

    
    gamma = args['gamma']
    lr = args['lr']
    max_episode = args['episode']
    NUM_PROCESS = args['process']

    # env = gym.make("CartPole-v0")
    env = []
    for i in range(NUM_PROCESS):
        env.append(gym.make('CartPole-v0'))
    

    agent = A2CAgent(gamma, env, lr, max_episode, NUM_PROCESS)
    agent.train()