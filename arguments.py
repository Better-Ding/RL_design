"""
@File    ：arguments.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/4 16:45 
@Desc    ：General constant definition and data structure definition for DQN algorithm
"""
from collections import namedtuple

# General data structure definition
Transition = namedtuple('Transition', ('current_state', 'action', 'delayed_reward', 'next_state'))

"""
nn_model.py 
"""
INPUT_DIM = 4
OUTPUT_DIM = 1

