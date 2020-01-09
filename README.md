# 1. Introduction

This repository is used as the implementations of policy gradient algorithms. 

The reference link is [here](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)

Things to do 

1. A3C
1. A2C

# 2. Add game 2048 
CartPole is kind of easy to beat, thus add 2048 for some tests.

The api to use 2048 is `game.update(action)` in which action is UP(1), LEFT(2), DOWN(3), RIGHT(4).

To test, there already is a player who randomly selects action. run it with `python 2048_simplify.py`


# 3. Implementation

## 3.1. Monte Carlo Policy Gradient

For REINFORCE, following files are implemented
1. `Reinforce_CartPole.py`: Implement the REINFORCE for game `CartPole-v0`. 
1. `Baseline_CartPole.py`: Implement the baseline REINFORCE for game `CartPole-v0`.

The observation used in `CartPole-v0` is 4-dim observation (see [here](https://github.com/openai/gym/wiki/CartPole-v0)).

### 3.1.1. REINFORCE CartPole
To start the training, the simplest way is to run `python Reinforce_CartPole.py`

Type `python Reinforce_CartPole.py --help` to see other options of training.

For reference only, mine is `python Reinforce_CartPole.py -e 20000 -a [continuing/episodic] --info rein_cart`.

### 3.1.2. Add baseline to REINFORCE

To start the training, the simplest way is to run `python Baseline_CartPole.py`

Type `python Baseline_CartPole.py --help` to see other options of training.

For reference only, mine is `python Baseline_CartPole.py -e 20000 -a [continuing/episodic] --info base_cart`.

### 3.1.3. Result

* Red line: Baseline REINFORCE & Continuing Update
* Green line: Baseline REINFORCE & Episodic Update
* Blue line: REINFORCE & Continuing Update
* Orange line: REINFORCE & Episodic Update

![REINFORCE_RESULT](./img/REINFORCE_CART.png)

### 3.1.4. Discussion

Basically, the only difference between REINFORCE and baseline REINFORCE is to subtract the mean value from discounted reward summation. However, with this minor modification, the result has a significant improvement. Another way of using baseline is to use another state-value function to estimate the baseline (see [here](http://incompleteideas.net/book/RLbook2018.pdf) Page 330) 

Considering the fact that the CartPole is a continuing tasks without the episodes, the result also shows that the when using episodic Update and continuing Update, the one trained with continuing Update strongly outperformed the other one.

BTW, the difference of episodic update and continuing update is

* Episodic: 

![](http://latex.codecogs.com/gif.latex?\mathbf{\theta}\leftarrow\mathbf{\theta}+\alpha\gamma^t(\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k)\nabla\ln\pi(A_t|S_t,\mathbf{\theta}))

* Continuing: 

![](http://latex.codecogs.com/gif.latex?\mathbf{\theta}\leftarrow\mathbf{\theta}+\alpha(\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k)\nabla\ln\pi(A_t|S_t,\mathbf{\theta}))


## 3.2. Actor-Critic

### 3.2.1. Training

To start the training, the simplest way is to run `python AC_CartPole.py`

Type `python AC_CartPole.py --help` to see other options of training.

For reference only, mine is `python AC_CartPole.py -e 2000 --info ac_cart`.

### 3.2.2. Result
![AC_RESULT](./img/AC_CART.PNG)

### 3.2.3. Discussion

* Significantly faster than REINFORCE, take around 400 epochs to get satifying result. In comparison, baseline reinforce takes 7k epochs.

* Have tried pixel value observation, still not working well.


## 3.4. A3C

Asynchronous Advantage Actor-Critic

Workable now, gonna add functionality and more discussion later.


# 4. Reference
[Reinforce Learning Book](http://incompleteideas.net/book/RLbook2018.pdf)

[CS294 DRL UCB](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/)

[Pytorch Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) 