"""
@File    ：nn_model.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/2 16:59 
@Desc    ：The DNN network
"""

import torch
import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: The network's input dimension ---> feature
        :param output_dim: The network's input dimension ---> action
        """
        super(DQNModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # define a full-connection network and use RELU() as activate function
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        action = self.fc(state)
        return action   # action's Q value
