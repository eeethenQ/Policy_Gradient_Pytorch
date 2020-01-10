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

class Net_Pixel(nn.Module):
    def __init__(self, h, w, outputs):
        super(Net_Pixel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.out_dim = outputs

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # convh = (conv2d_size_out(conv2d_size_out(h)))
        # convw = (conv2d_size_out(conv2d_size_out(w)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, self.out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        if self.out_dim == 1:
            return x
        else:
            return F.softmax(x, dim=-1)

class Net_Simple(nn.Module):
    def __init__(self, outputs):
        super(Net_Simple, self).__init__()
        self.out_dim = outputs
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, self.out_dim)  # Prob of Left

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.out_dim == 1:
            x = self.fc3(x)
        else:
            x = F.softmax(self.fc3(x), dim=-1)
        return x

class A2CAgent():
    def __init__(self, lr, gamma, use_pixel):
        if use_pixel:
            self.actor_net = Net_Pixel(30, 30, 2)
            self.critic_net = Net_Pixel(30, 30, 1)
        else:
            self.actor_net = Net_Simple(outputs=2)
            self.critic_net = Net_Simple(outputs=1)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr, betas=(0.9, 0.99))
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr, betas=(0.9, 0.99))
        self.gamma = gamma


    def get_action(self, observation):
        
        probs = self.actor_net(observation)
        state_value = self.critic_net(observation)

        # categorical distribution
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        return action.data, m.log_prob(action), state_value

    def update_parameter(self, action_traj, reward):
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        actor_loss = []
        critic_loss = []

        discounted_g = self.get_discount_g(reward, gamma = self.gamma)
        discounted_g = (discounted_g - torch.mean(discounted_g)) / torch.std(discounted_g)

        for (log_prob, state_value), R in zip(action_traj, discounted_g):
            advantage = R - state_value.item()
            
            actor_loss.append(-log_prob * advantage)

            critic_loss.append(F.smooth_l1_loss(state_value, torch.tensor([[R]])))

        actor_loss = torch.stack(actor_loss).sum()
        critic_loss = torch.stack(critic_loss).sum()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step() 
        self.critic_optimizer.step()

        return actor_loss.data, critic_loss.data

    def get_discount_g(self, reward, gamma):
        g_value = [0]
        for i in reversed(reward):
            g_value.append(g_value[-1] * gamma + i)
        return torch.Tensor(list(reversed(g_value))[:-1])


    def save(self, filename='./parameter'):
        torch.save(self.actor_net.state_dict(), filename+"_actor.pkl")
        torch.save(self.critic_net.state_dict(), filename+"_critic.pkl")


        
def collect_trajectory(env, agent, render, use_pixel):
    
    if use_pixel:
        # Collect the trajectory data using pixel value observation
        env.reset()
        observation = utils.get_observation_for_pixel_cartpole(env)
    else:
        # Collect the trajectory data using 4 dim observation
        observation = env.reset()
        observation = torch.FloatTensor(observation).unsqueeze(0)
    
    action_traj = []
    rewards = []
    while True:
        action, log_prob, state_value = agent.get_action(observation)
        new_observation, reward, done, _info = env.step(int(action))
        action_traj.append((log_prob, state_value))
        rewards.append(reward)

        if use_pixel:
            observation = utils.get_observation_for_pixel_cartpole(env)
        else:
            observation = torch.FloatTensor(new_observation).unsqueeze(0)

        if render and use_pixel is False:
            env.render('rgb_array')
        if done:
            return action_traj, rewards

if __name__ == "__main__":

    # Initialization
    parser = argparse.ArgumentParser(description='A2C Policy Gradient')
    parser.add_argument('-e', '--episode', type=int, help='number of episode you want to train', default=10000)
    parser.add_argument('-g', '--gamma', type=float, help='discount coefficient', default = 0.95)
    parser.add_argument('-l', '--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('-r', '--render', dest="RENDER", action='store_true', help='Flag to render when using simple observation')
    parser.add_argument('--info', type=str, help='extra information to label the log', default="")
    parser.add_argument('-p', '--usepixelvalue', dest='USE_PIXEL', action='store_true', help='Use Pixel Value')
    parser.set_defaults(RENDER=False)
    parser.set_defaults(USE_PIXEL=False)
    args = vars(parser.parse_args())

    # Hyper-parameter setup
    EPISODE = args['episode']
    GAMMA = args['gamma']
    LR = args['lr']
    RENDER = args['RENDER']
    info = args['info']
    USE_PIXEL = args['USE_PIXEL']

    # Training Setup
    agent = A2CAgent(LR, GAMMA, USE_PIXEL)
    game_name = "CartPole-v0"
    env = gym.make(game_name)
    writer = SummaryWriter('log/log_{}_{}_{}'.format(game_name, utils.get_date(), info))

    for i in range(EPISODE):
        # Collect trajectory
        action_traj, reward = collect_trajectory(env, agent, RENDER, USE_PIXEL)

        # Updating Models parameter
        actor_loss, critic_loss = agent.update_parameter(action_traj, reward)

        # Logging
        writer.add_scalar('actor_loss', actor_loss, i)
        writer.add_scalar('critic_loss', critic_loss, i)
        writer.add_scalar('reward', sum(reward), i)
        print("Episode : {}, actor loss {}, critic loss {}, reward {}".format(i, actor_loss, critic_loss, sum(reward)))

    # Termination
    agent.save('./parameters')
    writer.close()
    env.close()