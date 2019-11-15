import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
import retro

import argparse
import time
from tensorboardX import SummaryWriter

import utils

class Net(nn.Module):
    def __init__(self, h, w, outputs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        convh = (conv2d_size_out(conv2d_size_out(h)))
        convw = (conv2d_size_out(conv2d_size_out(w)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return F.softmax(x, dim=1)


class ReinforcePolicyGradient():
    def __init__(self, algorithm, lr, gamma):
        self.policy_net = Net(20, 20, 3)
        
        self.algorithm = algorithm
        self.lr = lr
        self.gamma = gamma


    def get_action_index(self, observation, epsilon):
        if np.random.rand() < epsilon:
            action_to_take = np.random.randint(0,3)
            action_index = np.zeros((1,3))
            action_index[:,action_to_take] = 1
            return action_index
        else:
            action_index = self.policy_net(observation)
            return action_index.detach().cpu().numpy()

    def update_parameter(self, trajectory):
        optimizer = torch.optim.Adam(self.policy_net.parameters(), self.lr, betas=(0.9, 0.99))
        optimizer.zero_grad()
        observation, action_index, reward = zip(*trajectory)
        # print(torch.cat(observation, dim=0).shape, type(observation), observation[0].shape)
        observation = torch.cat(observation, dim=0)
        action_index = torch.LongTensor(action_index).squeeze(1)
        reward = torch.Tensor(reward)

        if self.algorithm == 'episodic':
            # Get discounted coefficient
            discounted_g = self.get_discount_g(reward, gamma = self.gamma)
            gamma_series = np.empty((discounted_g.shape))
            gamma_series[0] = 1
            gamma_series[1:] = self.gamma
            gamma_series = torch.from_numpy(np.cumprod(gamma_series))
            discounted_g = discounted_g * gamma_series
        elif self.algorithm == 'continuing':
            discounted_g = self.get_discount_g(reward, gamma = self.gamma)
        else:
            raise ValueError("Wrong input")

        predict_action_index = self.policy_net(observation) 
        prob = torch.stack([torch.ones((predict_action_index.shape)) / 3 * 2 - predict_action_index, predict_action_index], dim=-1)
        prob = F.relu(prob)
        m = torch.distributions.categorical.Categorical(prob)
        loss = m.log_prob(action_index)
        loss = torch.sum(loss, dim=1)
        loss = - loss * discounted_g 
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()

        return loss, torch.sum(reward)

    def get_discount_g(self, reward, gamma):
        g_value = [0]
        for i in reversed(reward):
            g_value.append(g_value[-1] * gamma + i)
        
        return torch.Tensor(list(reversed(g_value[:-1])))


    def save(self, filename='./parameter.pkl'):
        torch.save(self.policy_net.state_dict(), filename)



def collect_trajectory(env, agent, epsilon, render):
    observation = utils.get_observation(env.reset())
    trajectory = []
    while True:
        action_index = agent.get_action_index(observation, epsilon)
        # print(action_index)
        action_index[action_index > 1/3] = 1
        action_index[action_index <= 1/3] = 0
        action = np.zeros((12, )).astype(int)
        action[0], action[6], action[7] = list(action_index.squeeze())
        
        new_observation, reward, done, _info = env.step(list(action))
        trajectory.append((observation, action_index, reward))
        observation = utils.get_observation(new_observation)

        if render:
            env.render()
        if done:
            return trajectory, new_observation
        

if __name__ == "__main__":

    # Initialization
    parser = argparse.ArgumentParser(description='Reinforce Policy Gradient (Monte Carlo)')
    parser.add_argument('-a', '--algorithm', type=str, help='episodic task / continuing task', default='episodic')
    parser.add_argument('-e', '--episode', type=int, help='number of episode you want to train', default=50)
    parser.add_argument('-g', '--gamma', type=float, help='discount coefficient', default = 0.95)
    parser.add_argument('-r', '--render', type=bool, help='whether or not to render the game scene', default=False)
    parser.add_argument('-l', '--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('-ep', '--epsilon', type=float, help='action selection random factor',default=1)
    parser.add_argument('-ed', '--epsilondecay', type=float, help='epsilon decay', default=0.95)
    args = vars(parser.parse_args())

    # Hyper-parameter setup
    EPISODE = args['episode']
    GAMMA = args['gamma']
    RENDER = args['render']
    LR = args['lr']
    EPSILON = args['epsilon']
    EPSILON_DECAY = args['epsilondecay']

    # Training Setup
    algo = args['algorithm']
    agent = ReinforcePolicyGradient(algo, LR, GAMMA)
    env = retro.make(game='Airstriker-Genesis', state=retro.State.DEFAULT)
    writer = SummaryWriter('log/log_{}'.format(time.time()))

    for i in range(EPISODE):
        
        # Collect trajectory
        trajectory, _ = collect_trajectory(env, agent, EPSILON, RENDER)

        # Updating Models parameter
        loss, reward = agent.update_parameter(trajectory)

        # Logging
        writer.add_scalar('loss', loss.data, i)
        writer.add_scalar('reward', reward.data, i)
        print("Episode : {}, the loss is {}, the reward is {}".format(i, loss.data, reward.data))

        if EPSILON > 0.01:
            EPSILON *= EPSILON_DECAY
    # Termination
    _, final_state = collect_trajectory(env, agent, EPSILON, True)
    utils.save_screenshot(final_state)
    agent.save('./parameters.pkl')
    writer.close()