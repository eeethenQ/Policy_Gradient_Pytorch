# 1. Introduction

This repository is used as the implementations of policy gradient algorithms. 

The reference link is [here](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)

Things to do 

1. Actor-Critic
1. Off-policy Policy Gradient
1. A3C
1. A2C

# 2. Implementation

## 2.1. Monte Carlo Policy Gradient

For REINFORCE, following files are implemented
1. `Reinforce_CartPole.py`: Implement the REINFORCE for game `CartPole-v0`. 
1. `Baseline_CartPole.py`: Implement the baseline REINFORCE for game `CartPole-v0`.

The observation used in `CartPole-v0` is 4-dim observation (see [here](https://github.com/openai/gym/wiki/CartPole-v0)) and pixel-value observation.

### 2.1.1. REINFORCE CartPole
To start the training, the simplest way is to run `python Reinforce_CartPole.py`

Type `python Reinforce_CartPole.py --help` to see other options of training.


### 2.1.2. Add baseline to REINFORCE

To start the training, the simplest way is to run `python Baseline_CartPole.py`

Type `python Baseline_CartPole.py --help` to see other options of training.

### 2.1.3. Result

* Red line: Baseline REINFORCE & Continuing Update
* Green line: Baseline REINFORCE & Episodic Update
* Blue line: REINFORCE & Continuing Update
* Orange line: REINFORCE & Episodic Update

![REINFORCE_RESULT](./img/REINFORCE_CART.png)

### 2.1.4. Discussion

Basically, the only difference between REINFORCE and baseline REINFORCE is to subtract the mean value from discounted reward summation. However, with this minor modification, the result has a significant improvement. Another way of using baseline is to use another state-value function to estimate the baseline (see [here](http://incompleteideas.net/book/RLbook2018.pdf) Page 330) 

Considering the fact that the CartPole is a continuing tasks without the episodes, the result also shows that the when using episodic Update and continuing Update, the one trained with continuing Update strongly outperformed the other one.

BTW, the difference of episodic update and continuing update is

* Episodic: ![](http://latex.codecogs.com/gif.latex?\mathbf{\theta}\leftarrow\mathbf{\theta}+\alpha\gamma^t(\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k)\nabla\ln\pi(A_t|S_t,\mathbf{\theta}))
* Continuing: ![](http://latex.codecogs.com/gif.latex?\mathbf{\theta}\leftarrow\mathbf{\theta}+\alpha(\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k)\nabla\ln\pi(A_t|S_t,\mathbf{\theta}))


## 2.2. Actor-Critic




# 3. Reference
[Reinforce Learning Book](http://incompleteideas.net/book/RLbook2018.pdf
)
