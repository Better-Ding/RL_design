"""
@File    ：replay_buffer.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/2 16:49 
@Desc    ：experience replay buffer
"""
import os
import pickle
import random

from state import State
from surrogate import Surrogate
from arguments import *

replay_memory_path = 'replay_memory_buffer.pk'


class ReplayBuffer:
    def __init__(self, surrogate: Surrogate, capacity: int):
        self.capacity = capacity
        self.buffer = list()
        self.__replay_memory_path = replay_memory_path
        self.surrogate = surrogate
        self.__index_adder = 0
        self.init_memory_buffer()

    def push(self, transition: Transition):
        """
            Push a transition into the buffer. if full, replaced
            @input:     transition
            @output:    None
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # ring buffer (self.buffer[self.__init_trans_count, ~]) for low fidelity samples
            self.buffer[self.__index_adder] = transition
            self.__index_adder = (self.__index_adder + 1) % self.capacity

    def sample(self, batch_size):
        """
            Randomly sample a list of Transition from buffer.
            @input:     batch size
            @output:    a list of Transitions
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return state, action, reward, next_state

    # save current memory into memory_save_path
    def save_current_memory(self, memory_save_path: str = None) -> None:
        if not memory_save_path:
            memory_save_path = self.__replay_memory_path
        with open(memory_save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def init_memory_buffer(self, resume_memory_buffer=True):
        """
            Init replay_memory_buffer with few experiences.
            :param resume_memory_buffer: if True load previous buffer, else recreate new one
            :return: None
        """
        if os.path.exists(self.__replay_memory_path) and resume_memory_buffer:
            with open(self.__replay_memory_path, 'rb') as f:
                self.buffer = pickle.load(f)
        else:
            tmp_count = 0
            # generate random transitions
            while tmp_count < self.capacity:
                # init a blank state
                current_state = State(if_init=True)
                for _ in range(3):
                    random_action = current_state.generate_random_action()
                    next_state = State(previous_state=current_state, action=random_action)
                    self.push(self.surrogate.pack_transition(current_state, random_action, next_state))
                    current_state = next_state
                    tmp_count += 1
            self.save_current_memory()

    def __len__(self):
        return len(self.buffer)
