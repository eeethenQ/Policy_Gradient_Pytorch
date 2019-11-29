# 1. Introduction

This repository is used as the implementations of policy gradient algorithms. 

The reference link is [here](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)

Things to do 

1. Actor-Critic
1. Off-policy Policy Gradient
1. A3C
1. A2C

# Implementation

## 2. Monte Carlo Policy Gradient

For REINFORCE, following files are implemented
1. `Reinforce_CartPole.py`: Implement the REINFORCE for game `CartPole-v0`. 

### 2.1. REINFORCE CartPole
To start the training, the simplest way is to run `python Reinforce_CartPole.py`

Type `python Reinforce_CartPole.py --help` to see other options of training.

This is the result running in my machine after 20000 epoch. The orange line is using directly the 4-dim observation (see [here](https://github.com/openai/gym/wiki/CartPole-v0)).

the blue line is using pixel value observation. Have tried different cropping technique(like cutting most of the blank area ), still not performing very well. Maybe try other algorithms

![REINFORCE_CART](./img/REINFORCE_CART.png)

## 3. Add based line to REINFORCE

# 4. Reference
[Reinforce Learning Book](http://incompleteideas.net/book/RLbook2018.pdf
)