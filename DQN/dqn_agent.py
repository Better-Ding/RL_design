"""
@File    ：dqn_agent.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/2 16:31 
@Desc    To design a DQN architecture
"""
import pickle
from typing import List

import math
import numpy as np
import torch
import random

from matplotlib import pyplot as plt

from nn_model import DQNModel
from replay_buffer import ReplayBuffer
from state import State
from surrogate import Surrogate
from arguments import *
import torch.nn.functional as F


class DQN_AGENT:
    def __init__(self, surrogate: Surrogate, replay_memory: ReplayBuffer, batch_size=32, learning_rate=0.0005,
                 epsilon=0.1, gamma=0.80, T=10):
        """
        :param surrogate: surrogate model ---> for cal reward
        :param replay_memory: experience replay buffer --> for random sample
        :param learning_rate: model learning rate
        :param epsilon: for epsilon-greedy algorithm
        :param batch_size: batch size
        :param gamma: the discount factor
        :param T: the update frequence for target network
        :return: None
        """
        self.gamma = gamma
        self.surrogate = surrogate
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_epoch = 0
        self.epsilon = epsilon
        self.T = T
        self.__policy_network = DQNModel(INPUT_DIM, OUTPUT_DIM).to(device)
        self.__target_network = DQNModel(INPUT_DIM, OUTPUT_DIM).to(device)
        # During initialization, the parameters of the target network are equal to the parameters of the Q network
        for param, target_param in zip(self.__policy_network.parameters(), self.__target_network.parameters()):
            target_param.data.copy_(param)
        self.optimizer = torch.optim.Adam(self.__policy_network.parameters(), lr=learning_rate)
        # decide when to update target model
        self.update_count = 0
        # for epsilon-greedy algorithm
        self.epsilon_end = 0.1
        self.epsilon_start = 0.8
        self.training_step = 0
        self.epsilon_decay_coef = 10000
        self.__training_indicators = list()

    def select_action(self, current_state, epsilon=0.1):
        """
        Select actions according to epsilon-greedy algorithm
        :param current_state: State
        :param epsilon: to decide strategy
        :return: actions according to current state--> float
        """
        if not epsilon:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.training_step / self.epsilon_decay_coef)
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
        self.__policy_network.eval()
        with torch.no_grad():
            current_state_rl_feature = torch.tensor(current_state.get_feature()).float().to(device)
            q_values = self.__policy_network(current_state_rl_feature)
            action_index = torch.argmax(q_values).item()
            action = current_state.select_action_by_idx(action_index)
            return action

    def train(self, training_epochs, log_interval=300):
        """
        Train the model and calculate the loss and Q value
        :param log_interval: to display the training status
        :param training_epochs
        :return: None
        """
        # if a pre-trainned model does not exists
        for epoch in range(training_epochs):
            self.training_epoch += 1
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
                self.training_step += 1
                action = self.select_action(current_state)
                next_state = State(previous_state=current_state, action=action)
                trainsition = self.surrogate.pack_transition(current_state, action, next_state)
                self.replay_memory.push(trainsition)
                current_state = next_state
                '''
                    Calculate the loss and Q values
                '''
                loss, total_q = self.experience_replay()
            if self.training_epoch % log_interval == 0:
                print('\n rl training epoch: {}, loss: {}, total_q: {}'.format(self.training_epoch, loss, total_q))

            self.__training_indicators.append(TrainingIndicator(self.training_epoch, loss, total_q))

            # update TD difference network & policy network evaluation
            if self.training_epoch % self.T == 0:
                # memorize learned knowledge
                self.__target_network.load_state_dict(self.__policy_network.state_dict())

    def experience_replay(self):
        """
        calculate policy network and Target network Q value and get loss
        :return: loss and q values
        """
        states, actions, delayed_rewards, next_states = self.replay_memory.sample(self.batch_size)

        states_counts = [state.get_episode_count() for state in states]
        action_idx = torch.tensor(
            [self.get_action_index(actions[i], states_counts[i]) for i in range(len(actions))]).view(-1, 1).to(device)
        state_feature_batch = torch.FloatTensor([state.get_feature() for state in states]).to(device)
        delayed_reward_batch = torch.FloatTensor(
            [[torch.tensor(delayed_reward).unsqueeze(dim=0).unsqueeze(dim=0)] for delayed_reward in
             delayed_rewards]).to(device)
        next_state_feature_batch = torch.FloatTensor([state.get_feature() for state in next_states]).to(device)
        # current result --- policy network
        curr_q = self.__policy_network(state_feature_batch).gather(1, action_idx)
        # next result --- target network
        non_final_mask = torch.tensor([not state.is_end_state() for state in next_states]).float().unsqueeze(1).to(
            device)
        next_q = self.__target_network(next_state_feature_batch)
        next_q = next_q * non_final_mask
        max_next_q = torch.max(next_q, 1)[0].view(-1, 1)

        # compute expected state-action values
        expected_q = max_next_q * self.gamma + delayed_reward_batch
        # compute loss
        # loss = F.smooth_l1_loss(curr_q, expected_q)
        loss = F.mse_loss(curr_q, expected_q)
        self.optimizer.zero_grad()
        # DQN network optimization
        loss.backward()
        for param in self.__policy_network.parameters():
            param.grad.data.clamp_(min=-1., max=1.)
        self.optimizer.step()

        # assemble criterions and return
        total_q = None
        with torch.no_grad():
            total_q = torch.mean(curr_q).detach()

        return loss.item(), total_q.item()

    # according to the count to get corresponding action index
    def get_action_index(self, action, count):
        if count == 0:
            idx = np.searchsorted(ACTIONS_HAMA, action)
        elif count == 1:
            idx = np.searchsorted(ACTIONS_GELMA, action)
        elif count == 2:
            idx = np.searchsorted(ACTIONS_SHEAR_RATE, action)
        return idx

    def propose_next_experiment(self, epsilon: float = 0.0) -> List[float]:
        return self.evaluate_knowledge(epsilon)[-1][0]

    def save_training_indicators(self, training_indicator_path: str = DQL_TRAINING_INDICATOR_PATH):
        with open(training_indicator_path, 'wb') as f:
            pickle.dump(self.__training_indicators, f)

    def evaluate_knowledge(self, epsilon: float = None):
        # prepare a blank kirigami structure
        current_state = State(if_init=True)
        action = None
        state_action_seq = list()
        # apply action sequence based on greedy policy
        for _ in range(3):
            # Change from greedy select -> epsilon select. --20220827
            # action = self.greedy_select(current_state)
            action = self.select_action(current_state=current_state, epsilon=epsilon)
            next_state = State(previous_state=current_state, action=action)
            state_action_seq.append([current_state.get_ex_content(), action])
            current_state = next_state

        state_action_seq.append([current_state.get_ex_content(), None])
        return state_action_seq

    def save_knowledge(self, knowledge_save_path: str = DQL_AGENT_PATH):
        torch.save(self.__policy_network.state_dict(), knowledge_save_path)