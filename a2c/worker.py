import torch
from torch.multiprocessing import Process
import torch.nn as nn
import torch.nn.functional as F

from network import Net_Simple, Net_Pixel



class Worker:
    def __init__(self, id, gamma, env, global_actor_net, global_critic_net, \
                global_actor_optim, global_critic_optim):
        # super(Worker, self).__init__()
        self.global_actor_net = global_actor_net
        self.global_critic_net = global_critic_net
        self.global_actor_optim = global_actor_optim
        self.global_critic_optim = global_critic_optim
        self.gamma = gamma
        self.env = env
        self.id = id

    def get_action(self, observation):
        
        probs = self.global_actor_net(observation)
        state_value = self.global_critic_net(observation)

        # categorical distribution
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        return action.data, m.log_prob(action), state_value

    def get_loss(self, action_traj, reward):
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

        return actor_loss, critic_loss

    def get_discount_g(self, reward, gamma):
        g_value = [0]
        for i in reversed(reward):
            g_value.append(g_value[-1] * gamma + i)
        return torch.Tensor(list(reversed(g_value))[:-1])


    def collect_trajectory(self, ):
        # Collect the trajectory data using 4 dim observation
        observation = self.env.reset()
        observation = torch.FloatTensor(observation).unsqueeze(0)

        action_traj = []
        rewards = []
        while True:
            action, log_prob, state_value = self.get_action(observation)
            new_observation, reward, done, _info = self.env.step(int(action))
            action_traj.append((log_prob, state_value))
            rewards.append(reward)

            observation = torch.FloatTensor(new_observation).unsqueeze(0)

            if done:
                return action_traj, rewards

    def update_global(self, action_traj, reward):
        actor_loss, critic_loss = self.get_loss(action_traj, reward)
        self.global_actor_optim.zero_grad()
        actor_loss.backward()
        self.global_actor_optim.step()

        self.global_critic_optim.zero_grad()
        critic_loss.backward()
        self.global_critic_optim.step()

    def run(self, ):
        action_traj, reward = self.collect_trajectory()
        self.update_global(action_traj, reward)
        # print("Worker: {}, Reward: {}".format(self.id, sum(reward)))
        return (sum(reward))






