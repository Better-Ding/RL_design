"""
@File    ：state.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/4 10:54 
@Desc    ： Environment state class.
"""
import random
from copy import deepcopy
from arguments import *


class State:
    def __init__(self, if_init: bool = False, previous_state=None, action: float = None, episode_len=3):
        """
           self.content --> 2 materials HAMA(%),GelMA(%), 1 factor ShearRate (/s)
           self.episode_len --> 3
        """
        if if_init:
            # [HAMA,GelMA,ShearRate]
            self.content = [0, 0, 0]
            self.episode_len = episode_len
            self.episode_count = 0
        else:
            self.content = deepcopy(previous_state.get_ex_content())
            self.episode_len = episode_len
            self.episode_count = previous_state.get_episode_count() + 1
            substitution_index = self.episode_count
            # print(substitution_index)
            self.content[substitution_index - 1] = action

    def get_episode_len(self) -> int:
        return self.episode_len

    def get_episode_count(self) -> int:
        return self.episode_count

    def get_ex_content(self):
        return self.content

    # len(feature) corresponds to flattened dimensions in DqlModel.
    def get_feature(self):
        """
        feature: 2 content + 1 episode count
        """
        feature = deepcopy(self.content)
        feature.append(self.episode_count)
        return feature
    
    # check if end state
    def is_end_state(self):
        return self.episode_count == self.episode_len - 1

    def generate_random_action(self):
        """
            Generate one random action that can be applied to this state

            @output:  episode_count = 0 ---> HAMA ---> 0.25 - 1.5
                      episode_count = 1 ---> GelMA ----> 5 - 17.5
                      episode_count = 2 ---> Shear Rate(after mapping) ----> 0 - 25
        """
        rand_idx = random.randint(0, ACTIONS_COUNT-1)
        if self.episode_count == 0:
            action = ACTIONS_HAMA[rand_idx]
        elif self.episode_count == 1:
            action = ACTIONS_GELMA[rand_idx]
        elif self.episode_count == 2:
            action = ACTIONS_SHEAR_RATE[rand_idx]
        else:
            raise Exception("Wrong episode count")
        return action

    def select_action_by_idx(self, idx):
        """
        select action by index provided by Policy Network
        :param idx: action index
        :return: action
        """
        if self.episode_count == 0:
            action = ACTIONS_HAMA[idx]
        elif self.episode_count == 1:
            action = ACTIONS_GELMA[idx]
        elif self.episode_count == 2:
            action = ACTIONS_SHEAR_RATE[idx]
        else:
            raise Exception("Wrong episode count")
        return action