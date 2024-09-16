"""
@File    ：dqn_agent.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/2 16:31 
@Desc    To design a DQN architecture
"""
import numpy as np
import torch

from nn_model import DQNModel
from replay_buffer import ReplayBuffer
from state import State
from surrogate import Surrogate
from arguments import *


class DQN_AGENT:
    def __init__(self, surrogate: Surrogate, replay_memory: ReplayBuffer, batch_size=32, learning_rate=0.001,
                 epsilon=0.1, gamma=0.99):
        """
        :param surrogate: surrogate model ---> for cal reward
        :param replay_memory: experience replay buffer --> for random sample
        :param learning_rate: model learning rate
        :param epsilon: for epsilon-greedy algorithm
        :param batch_size: batch size
        :param gamma: the discount factor
        :return: None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        self.gamma = gamma
        self.surrogate = surrogate
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.__policy_network = DQNModel(INPUT_DIM, OUTPUT_DIM).to(self.device)
        self.__target_network = DQNModel(INPUT_DIM, OUTPUT_DIM).to(self.device)
        # During initialization, the parameters of the target network are equal to the parameters of the Q network
        for param, target_param in zip(self.__policy_network.parameters(), self.__target_network.parameters()):
            target_param.data.copy_(param)
        self.optimizer = torch.optim.Adam(self.__policy_network.parameters(), lr=learning_rate)
        # decide when to update target model
        self.update_count = 0

    def select_action(self, current_state, epsilon=0.0):
        """
        Select actions according to epsilon-greedy algorithm
        :param current_state: State
        :param epsilon: to decide strategy
        :return: actions according to current state--> float
        """
        if np.random.rand() < epsilon:
            return current_state.generate_random_action()
        else:
            # greedy selection
            return self.greedy_select(current_state)

    def greedy_select(self, current_state):
        """
        Greedily choose from possible composition actions according to previous selected composition design actions
        :return: action (float)
        """
        self.__policy_network.eval()
        with torch.no_grad():
            current_state_rl_feature = torch.tensor(current_state.get_feature()).float().to(device)
            q_values = self.__policy_network(current_state_rl_feature)
            action_index = torch.argmax(q_values).item()
            action = current_state.select_action_by_idx(action_index)
            return action

    def train(self, training_epochs):
        """
        Train the model and calculate the loss and Q value
        :param training_epochs
        :return:
        """
        # if a pre-trainned model does not exists
        for _ in range(training_epochs):
            # self.__training_epoch += 1
            # evaluation mode for target network
            self.__target_network.eval()
            # training mode for policy network
            self.__policy_network.train()
            # prepare a blank start state
            current_state = State(if_init=True)
            # prepare training state indicating parameters
            loss, total_q = None, None
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
                # loss, total_q = self.experience_replay()

    def experience_replay(self):
        state, action, reward, next_state = self.replay_memory.sample(self.batch_size)
        # sample_batch = self.replay_memory.sample(self.__sample_batch_size)
        # print(state)
        # print("=========================")
        # print(action)
        return 1, 2
