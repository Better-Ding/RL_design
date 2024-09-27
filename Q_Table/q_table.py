"""
@File    ：q_table.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/23 23:05 
@Desc    ：Q_table methods
"""
import numpy as np

from replay_buffer import ReplayBuffer
from surrogate import Surrogate
from agent import AGENT

def execute():
    # -------------------------------------------------------------------------
    # parameters setup
    replay_memory_capacity = 3000
    num_episodes = 5000
    target = 100
    print('Repaly memory capacity: {}'.format(replay_memory_capacity))
    print('training episodes: {}'.format(num_episodes))
    # -------------------------------------------------------------------------
    # prepare Gaussian Process Regression model for environment surrogate
    surrogate = Surrogate(target=target)

    replay_memory = ReplayBuffer(surrogate, capacity=replay_memory_capacity)
    agent = AGENT()


if __name__ == '__main__':
    execute()
