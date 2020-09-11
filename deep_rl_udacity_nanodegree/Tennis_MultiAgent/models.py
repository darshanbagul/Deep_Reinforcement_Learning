import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """ Actor (policy) model class definition """
    def __init__(self, state_size, action_size, config, leak=0.01):
        """ Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1 (int)  : Number of nodes in first hidden layer
            fc2 (int)  : Number of nodes in second hidden layer
            leak: leaky relu parameter
            seed (int) : Random seed
        """
        super(Actor, self).__init__()
        self.config = config
        self.seed = torch.manual_seed(self.config.seed)
        self.leak = leak
        self.fc1 = nn.Linear(state_size, self.config.architecture[0])
        self.fc2 = nn.Linear(self.config.architecture[0], self.config.architecture[1])
        self.fc3 = nn.Linear(self.config.architecture[1], action_size)
        if self.config.batch_normalization:
            self.bn = nn.BatchNorm1d(state_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.config.batch_normalization:
            state = self.bn(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x =  torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, config, leak=0.01):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1 (int): Number of nodes in the first hidden layer
            fc2 (int): Number of nodes in the second hidden layer
            fc3 (int): Number of nodes in the third hidden layer
            leak: leaky relu parameter
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.config = config
        self.seed = torch.manual_seed(self.config.seed)
        self.leak = leak
        if self.config.batch_normalization:
            self.bn = nn.BatchNorm1d(state_size)
        
        self.fcs1 = nn.Linear(state_size, self.config.architecture[0])
        self.fc2 = nn.Linear(self.config.architecture[0]+action_size, self.config.architecture[1])
        self.fc3 = nn.Linear(self.config.architecture[1], 1)
        # self.fc4 = nn.Linear(fc3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.fcs1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        """ Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.config.batch_normalization:
            state = self.bn(state)
        x = F.leaky_relu(self.fcs1(state), negative_slope=self.leak)
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        # x = F.leaky_relu(self.fc3(x), negative_slope=self.leak)
        x =  self.fc3(x)
        return torch.sigmoid(x)