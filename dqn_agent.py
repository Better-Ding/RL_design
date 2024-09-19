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
import torch.nn.functional as F


class DQN_AGENT:
    def __init__(self, surrogate: Surrogate, replay_memory: ReplayBuffer, batch_size=32, learning_rate=0.001,
                 epsilon=0.1, gamma=0.99, T =10):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        self.gamma = gamma
        self.surrogate = surrogate
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_epoch = 0
        self.epsilon = epsilon
        self.T = 10
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

    def train(self, training_epochs, log_interval = 1000):
        """
        Train the model and calculate the loss and Q value
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
            if (self.training_epoch) % log_interval == 0:
                print('rl training epoch: {}, loss: {}, total_q: {}'.format(self.training_epoch, loss, total_q))
            # update TD difference network & policy network evaluation
            if self.training_epoch % self.T == 0:
                # memorize learned knowledge
                self.__target_network.load_state_dict(self.__policy_network.state_dict())

    def experience_replay(self):
        
        states, actions, delayed_rewards, next_states = self.replay_memory.sample(self.batch_size)
        
        states_counts = [state.get_episode_count() for state in states]
        action_idx = torch.tensor([self.get_action_index(actions[i], states_counts[i]) for i in range(len(actions))]).view(-1,1).to(device)
        state_feature_batch = torch.FloatTensor([state.get_feature() for state in states]).to(device)
        delayed_reward_batch = torch.FloatTensor([[torch.tensor(delayed_reward).unsqueeze(dim = 0).unsqueeze(dim = 0)] for delayed_reward in delayed_rewards]).to(device)
        next_state_feature_batch = torch.FloatTensor([state.get_feature() for state in next_states]).to(device)
        # current result --- policy network
        curr_q = self.__policy_network(state_feature_batch).gather(1, action_idx)
        # next result --- target network
    
        non_final_mask = torch.tensor([not state.is_end_state() for state in next_states]).float().unsqueeze(1).to(device)

        next_q = self.__target_network(next_state_feature_batch)
        next_q = next_q * non_final_mask
        max_next_q = torch.max(next_q, 1)[0].view(-1,1)
        # compute expected state-action values
        expected_q = max_next_q * self.gamma + delayed_reward_batch
        # compute loss
        loss = F.smooth_l1_loss(curr_q, expected_q)

        self.optimizer.zero_grad()
        # DQN network optimization
        loss.backward()
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
    
