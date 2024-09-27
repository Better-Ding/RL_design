"""
@File    ：agent.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/25 14:06 
@Desc    ：Description of the file
"""
"""
@File    ：dqn_agent.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/2 16:31 
@Desc    To design a DQN architecture
"""
from typing import List

import math
import numpy as np
import torch
import random
from replay_buffer import ReplayBuffer
from state import State
from surrogate import Surrogate
from arguments import *
import torch.nn.functional as F


class AGENT:
    def __init__(self, surrogate: Surrogate, replay_memory: ReplayBuffer, batch_size=32, epsilon=0.1, gamma=0.80, T=10):
        self.gamma = gamma
        self.surrogate = surrogate
        self.replay_memory = replay_memory
        self.training_epoch = 0
        self.epsilon = epsilon
        self.T = T
        # decide when to update target model
        self.update_count = 0
        # for epsilon-greedy algorithm
        self.epsilon_end = 0.1
        self.epsilon_start = 0.8
        self.training_step = 0
        self.epsilon_decay_coef = 10000

    def select_action(self, current_state, epsilon=0.1):
        """
        Select actions according to epsilon-greedy algorithm
        :param current_state: State
        :param epsilon: to decide strategy
        :return: actions according to current state--> float
        """
        # if not epsilon:
        #     epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        #               math.exp(-1. * self.training_step / self.epsilon_decay_coef)
        if random.random() < epsilon:
            # select a random action with epsilon_tmp probability
            return current_state.generate_random_action()
        else:
            # greedy selection

            return self.greedy_select(current_state)

    def greedy_select(self, current_state):
        """
        Greedily choose from possible composition actions according to previous selected composition design actions
        :return: action (float)
        """
        with torch.no_grad():
            current_state_rl_feature = torch.tensor(current_state.get_feature()).float().to(device)
            q_values = self.__policy_network(current_state_rl_feature)
            action_index = torch.argmax(q_values).item()
            action = current_state.select_action_by_idx(action_index)
            return action

    def train(self, training_epochs, log_interval=1000):
        """
        Train the model and calculate the loss and Q value
        :param log_interval: to display the training status
        :param training_epochs
        :return: None
        """
        # if a pre-trainned model does not exists
        for epoch in range(training_epochs):
            self.training_epoch += 1
            # prepare a blank start state
            current_state = State(if_init=True)
            # prepare training state indicating parameters
            total_reward = 0
            for _ in range(3):
                '''
                    Prepare a new transition to shove into memory buffer.
                '''
                # self.__training_step += 1
                action = self.select_action(current_state)

                next_state = State(previous_state=current_state, action=action)
                trainsition = self.surrogate.pack_transition(current_state, action, next_state)
                self.replay_memory.push(trainsition)
                current_state = next_state
                '''
                    Calculate the loss and Q values
                '''
                loss, total_q = self.experience_replay()
            if self.training_epoch % 10 == 0:
                print('rl training epoch: {}, loss: {}, total_q: {}'.format(self.training_epoch, loss, total_q))
            # update TD difference network & policy network evaluation
            if self.training_epoch % self.T == 0:
                # memorize learned knowledge
                self.__target_network.load_state_dict(self.__policy_network.state_dict())

    # according to the count to get corresponding action index
    def get_action_index(self, action, count):
        if count == 0:
            idx = np.searchsorted(ACTIONS_HAMA, action)
        elif count == 1:
            idx = np.searchsorted(ACTIONS_GELMA, action)
        elif count == 2:
            idx = np.searchsorted(ACTIONS_SHEAR_RATE, action)
        return idx

    # def propose_next_experiment(self, epsilon: float = 0.0) -> List[float]:
    #     return self.evaluate_knowledge(epsilon)[-1][0]
    #
    # def evaluate_knowledge(self, epsilon: float = None):
    #     # prepare a blank kirigami structure
    #     current_state = State(if_init=True)
    #     action = None
    #     state_action_seq = list()
    #     # apply action sequence based on greedy policy
    #     for _ in range(3):
    #         # Change from greedy select -> epsilon select. --20220827
    #         # action = self.greedy_select(current_state)
    #         action = self.select_action(current_state=current_state, epsilon=epsilon)
    #         next_state = State(previous_state=current_state, action=action)
    #         state_action_seq.append([current_state.get_ex_content(), action])
    #         current_state = next_state
    #
    #     state_action_seq.append([current_state.get_ex_content(), None])
    #     return state_action_seq
