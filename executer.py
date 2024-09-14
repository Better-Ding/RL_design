"""
@File    ：executer.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/3 11:04 
@Desc    ：Description of the file
"""
from dqn_agent import DQN_AGENT
from replay_buffer import ReplayBuffer
from surrogate import Surrogate


def execute():
    # -------------------------------------------------------------------------
    # parameters setup
    replay_memory_capacity = 3000
    training_epochs = 30000
    batch_size = 64
    target = 150
    print('Repaly memory capacity: {}'.format(replay_memory_capacity))
    print('training epochs: {}'.format(training_epochs))

    # -------------------------------------------------------------------------
    # prepare Gaussian Process Regression model for environment surrogate
    surrogate = Surrogate(target=target)

    # -------------------------------------------------------------------------
    # instantiate DQN agent
    replay_memory = ReplayBuffer(surrogate, capacity=replay_memory_capacity)
    agnt = DQN_AGENT(surrogate, replay_memory, batch_size=batch_size)

    # every (training_epochs // proposition_logs) per composition proposition.
    proposition_logs = 100
    proposition_log_res = list()

    # -------------------------------------------------------------------------
    # train DQN agent
    # for _ in range(proposition_logs):
    #     # set need_training explicitly
    #     # train DQN with desired epochs
    #     agnt.train(training_epochs=training_epochs // proposition_logs)


if __name__ == '__main__':
    execute()
