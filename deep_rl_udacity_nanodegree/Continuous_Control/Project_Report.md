# Project 2 - Continuos Control
## Follow a moving Target

### 1. Introduction

In this project we train policy-based reinforcement learning agent to learn the optimal policy in a model-free Reinforcement Learning setting using a Unity environment, in which a double-jointed arm can move to target locations. 
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1. 

Before training an agent that chooses actions randomly can be seen below:

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

### 2. Implementation

#### Learning Algorithm:
1. In this implementation we use Deep Deterministic Policy Gradient (DDPG) from  *Actor-Critic Method* as the learning algorithm.
2. Why this algorithm? Actor critic methods merge the positives from value based methods such as TD Learning (low-variance) and policy based methods such as REINFORCE where the 
  a. Actor is a neural network which updates the policy 
  b. Critic is another neural network which evaluates the policy being learned which is, in turn, used to train the Actor.

3. The DDPG algorithm differs from vanilla Actor-Critic method in way where the actor produces a deterministic policy instead of the usual stochastic policy and 
the critic evaluates the deterministic policy. The critic is updated using the TD-error and the actor is trained using the deterministic policy gradient algorithm.
4. For training multiple agents in parallel, we are using 20 agents and updating the network weights every 20 steps.
5. ***Fixed targets*** - Picked methodology from DQN implementation
6. ***Soft Updates*** - the target networks are updated using soft updates where during each update step, 0.01% of the local network weights are mixed with the target networks weights, i.e. 99.99% of the target network weights are retained and 0.01% of the local networks weights are added.
7. ***Experience Replay*** - we maintain a Replay Buffer of fixed size (say N). We run a few episodes and store each of the experiences in the buffer. After a fixed number of iterations, we sample a few experiences from this replay buffer and use that to calculate the loss and eventually update the parameters. This helps remove correlation between sequential experiences.
8. Based on experience of some peers it seems Leaky ReLu performs a little better in stabilizing the learning than ReLu.
9. ***Hyperparameter***

| Hyperparameter                  | Value |
| --------------------------------| ----- |
| Replay buffer size              | 1e6   |
| Batch size                      | 1024  |
| Gamma (discount factor)         | 0.99  |
| Tau                             | 1e-3  |
| Actor Learning rate             | 1e-4  |
| Critic Learning rate            | 3e-4  |
| Update interval                 | 20    |
| Update times per interval       | 10    |
| Number of episodes              | 500   |
| Max timesteps per episode       | 1000  |
| Leak for LeakyReLU              | 0.01  |


### 3. Results

The best performance was achieved by **DDPG** where the reward of +30 was achieved in  episodes. 
The plot of the rewards across episodes is shown below:

### 4. Further Improvements

- ***Prioritized Replay*** ([paper](https://arxiv.org/abs/1511.05952)) - expected to improved performance.

- Algorithms like ***PPO, TRPO, A3C, D4PG*** that have been discussed in the course could potentially lead to better results as well.

- The ***Q-prop algorithm*** combines both off-policy and on-policy learning; may be useful to try.

- ***Batch Normalization*** could be added to improve the stability of learning.

- Deep learning techniques like ***cyclical learning rates*** and warm restarts could be useful as well.
