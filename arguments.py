"""
@File    ：arguments.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/4 16:45 
@Desc    ：General constant definition and data structure definition for DQN algorithm
"""
from collections import namedtuple

import numpy as np
import torch

# General data structure definition
Transition = namedtuple('Transition', ('current_state', 'action', 'delayed_reward', 'next_state'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
state.py
"""

# HAMA ---> 0.25 - 1.5
# GelMA ---> 5 - 20
# ShearRate(after mapping) ---> 1 - 25
HAMA_MIN, HAMA_MAX = 0.5, 1
GELMA_MIN, GELMA_MAX = 7.5, 15
SHEARRATE_MIN, SHEARRATE_MAX = 1, 25
ACTIONS_COUNT = 126

# possible actions
ACTIONS_HAMA = np.round(np.linspace(HAMA_MIN, HAMA_MAX, ACTIONS_COUNT), decimals=4)
ACTIONS_GELMA = np.round(np.linspace(GELMA_MIN, GELMA_MAX, ACTIONS_COUNT), decimals=4)
ACTIONS_SHEAR_RATE = np.round(np.linspace(SHEARRATE_MIN, SHEARRATE_MAX, ACTIONS_COUNT), decimals=4)
# print(ACTIONS_HAMA)
# print(ACTIONS_GELMA)
# print(ACTIONS_SHEAR_RATE)


"""
nn_model.py 
"""
INPUT_DIM = 4
OUTPUT_DIM = ACTIONS_COUNT
