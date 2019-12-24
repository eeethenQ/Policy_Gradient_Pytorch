import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
import gym

import argparse
from tensorboardX import SummaryWriter

import utils

class PolicyNet_Pixel(nn.Module):
    def __init__(self, h, w, outputs):
        super(PolicyNet_Pixel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # convh = (conv2d_size_out(conv2d_size_out(h)))
        # convw = (conv2d_size_out(conv2d_size_out(w)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return torch.sigmoid(x) # if going to left or not

class PolicyNet_Simple(nn.Module):
    def __init__(self):
        super(PolicyNet_Simple, self).__init__()

        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 1)  # Prob of Left

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class ReinforcePolicyGradient():
    def __init__(self, algorithm, lr, gamma, use_pixel):
        if use_pixel:
            # the parameter means the (input image height, input image width, output)
            self.policy_net = PolicyNet_Pixel(30, 30, 1)
        else :
            self.policy_net = PolicyNet_Simple()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr, betas=(0.9, 0.99))
        
        self.algorithm = algorithm
        self.gamma = gamma


    def get_action(self, observation, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.rand()
            return np.ceil(action*2) - 1
        else:
            action = self.policy_net(observation).detach().cpu().numpy().squeeze()
            return np.ceil(action*2) - 1

    def update_parameter(self, trajectory):
        
        self.optimizer.zero_grad()
        observation, action, reward = zip(*trajectory)
        # print(torch.cat(observation, dim=0).shape, type(observation), observation[0].shape)
        observation = torch.cat(observation, dim=0)
        action = torch.FloatTensor(action).to(utils.device)
        reward = torch.Tensor(reward).to(utils.device)

        if self.algorithm == 'episodic':
            discounted_g = self.get_discount_g_episodic(reward, self.gamma)
        elif self.algorithm == 'continuing':
            discounted_g = self.get_discount_g_continuing(reward, gamma = self.gamma)
        else:
            raise ValueError("Wrong input")

        # Add Baseline
        discounted_g = (discounted_g - torch.mean(discounted_g)) / torch.std(discounted_g)

        for i in range(reward.shape[0]):
            prob = self.policy_net(observation[i].unsqueeze(0))
            m = torch.distributions.Bernoulli(prob)
            loss = -m.log_prob(action[i]) * discounted_g[i]
            loss.backward()
        self.optimizer.step()

        return loss, torch.sum(reward)

    def get_discount_g_continuing(self, reward, gamma):
        g_value = [0]
        for i in reversed(reward):
            g_value.append(g_value[-1] * gamma + i)
        return torch.Tensor(list(reversed(g_value))[:-1])

    def get_discount_g_episodic(self, reward, gamma):
        gamma_tensor = torch.tensor([gamma**i for i in range(reward.shape[0])])
        return self.get_discount_g_continuing(reward, gamma) * gamma_tensor


    def save(self, filename='./parameter.pkl'):
        torch.save(self.policy_net.state_dict(), filename)

def collect_trajectory_pixel(env, agent, epsilon):
    # Collect the trajectory data using pixel value observation
    env.reset()
    observation = utils.get_observation_for_pixel_cartpole(env)
    trajectory = []
    while True:
        action = agent.get_action(observation, epsilon)
        # print(observation.shape, action)
        new_observation, reward, done, _info = env.step(int(action))
        trajectory.append((observation, action, reward))
        observation = utils.get_observation_for_pixel_cartpole(env)
        if done:
            return trajectory
        
def collect_trajectory_simple(env, agent, epsilon, render):
    # Collect the trajectory data using 4 dim observation
    observation = env.reset()
    observation = torch.FloatTensor(observation).unsqueeze(0).to(utils.device)
    trajectory = []
    while True:
        action = agent.get_action(observation, epsilon)
        new_observation, reward, done, _info = env.step(int(action))
        trajectory.append((observation, action ,reward))
        observation = torch.FloatTensor(new_observation).unsqueeze(0)

        if render:
            env.render('rgb_array')
        if done:
            return trajectory

if __name__ == "__main__":

    # Initialization
    parser = argparse.ArgumentParser(description='Reinforce Policy Gradient (Monte Carlo)')
    parser.add_argument('-a', '--algorithm', type=str, help='episodic task / continuing task', default='continuing')
    parser.add_argument('-e', '--episode', type=int, help='number of episode you want to train', default=10000)
    parser.add_argument('-g', '--gamma', type=float, help='discount coefficient', default = 0.95)
    parser.add_argument('-l', '--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('-ep', '--epsilon', type=float, help='action selection random factor',default=1)
    parser.add_argument('-ed', '--epsilondecay', type=float, help='epsilon decay', default=0.95)
    parser.add_argument('-r', '--render', dest="RENDER", action='store_true', help='Flag to render when using simple observation')
    parser.add_argument('-p', '--usepixelvalue', dest='USE_PIXEL', action='store_true', help='Use Pixel Value')
    parser.add_argument('--info', type=str, help='extra information to label the log', default=None)
    parser.set_defaults(RENDER=False)
    parser.set_defaults(USE_PIXEL=False)
    args = vars(parser.parse_args())

    # Hyper-parameter setup
    EPISODE = args['episode']
    GAMMA = args['gamma']
    LR = args['lr']
    EPSILON = args['epsilon']
    EPSILON_DECAY = args['epsilondecay']
    USE_PIXEL = args['USE_PIXEL']
    RENDER = args['RENDER']
    info = args['info']

    # Training Setup
    algo = args['algorithm']
    agent = ReinforcePolicyGradient(algo, LR, GAMMA, USE_PIXEL)
    game_name = "CartPole-v0"
    env = gym.make(game_name)
    if info is None:
        writer = SummaryWriter('log/log_{}_{}'.format(game_name, utils.get_date()))
    else:
        writer = SummaryWriter('log/log_{}_{}_{}'.format(game_name, utils.get_date(), info))
    for i in range(EPISODE):
        
        # Collect trajectory
        if USE_PIXEL :
            trajectory = collect_trajectory_pixel(env, agent, EPSILON)
        else :
            trajectory = collect_trajectory_simple(env, agent, EPSILON, RENDER)

        # Updating Models parameter
        loss, reward = agent.update_parameter(trajectory)

        # Logging
        writer.add_scalar('loss', loss.data, i)
        writer.add_scalar('reward', reward.data, i)
        print("Episode : {}, the loss is {}, the reward is {}".format(i, loss.data[0][0], reward.data))

        if EPSILON > 0.01:
            EPSILON *= EPSILON_DECAY
    # Termination
    if USE_PIXEL:
        trajectory = collect_trajectory_pixel(env, agent, EPSILON)
        utils.save_screenshot(trajectory[-1][0].detach().cpu().numpy().squeeze(0))
    agent.save('./parameters.pkl')
    writer.close()
    env.close()