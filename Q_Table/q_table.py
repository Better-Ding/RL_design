"""
@File    ：q_table.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/23 23:05 
@Desc    ：Q_table methods
"""
from replay_buffer import ReplayBuffer
from surrogate import Surrogate


def execute():
    # -------------------------------------------------------------------------
    # parameters setup
    replay_memory_capacity = 5000
    training_epochs = 30000
    target = 100
    print('Repaly memory capacity: {}'.format(replay_memory_capacity))
    print('training epochs: {}'.format(training_epochs))
    # -------------------------------------------------------------------------
    # prepare Gaussian Process Regression model for environment surrogate
    surrogate = Surrogate(target=target)

    # -------------------------------------------------------------------------
    # instantiate DQN agent
    replay_memory = ReplayBuffer(surrogate, capacity=replay_memory_capacity)


if __name__ == '__main__':
    execute()
