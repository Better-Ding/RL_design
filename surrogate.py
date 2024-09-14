"""
@File    ：surrogate.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/3 11:09 
@Desc    ：Use Gaussian Process Regressor as environment surrogate
            1. Train the model
                Input: HAMA (%) + GelMA (%) + ShearRate(/s)
                Label: Viscosity
"""
import pickle
from collections import namedtuple
from typing import List
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from arguments import *
from state import State

FEATURE_VISCOSITY_PATH = 'data/data-modi-vistr.xlsx'
GP_MODEL_PATH = './models/gp_model.pk'


class Surrogate:
    def __init__(self, target):
        # prepare buffers
        self.read_data()
        # instantiate GPR model
        gpr_kernel = ConstantKernel() * RBF() + WhiteKernel()
        self.__gp_model = GaussianProcessRegressor(kernel=gpr_kernel, n_restarts_optimizer=10)
        # target value to control the pred value not exceed 150
        self.target = target
        # train a gpr model with initial exp_points
        self.train_gp_model()

    # read feature_viscosity data 
    def read_data(self, path=FEATURE_VISCOSITY_PATH):
        data = pd.read_excel(path)
        self.feature_viscosity_buffer = data.to_numpy()

    # save Surrogate.__gp_model -> gp_model_path
    def save_gp_model(self, gp_model_path=GP_MODEL_PATH):
        with open(gp_model_path, 'wb') as f:
            pickle.dump(self.__gp_model, f)

    # Train gpr model
    def train_gp_model(self):
        x = [item[0:3].tolist() for item in self.feature_viscosity_buffer]
        y = [item[7] for item in self.feature_viscosity_buffer]
        self.__gp_model.fit(x, y)

    def predict(self, features: List[float]) -> float:
        """
            @input:     HAMA (%) + GelMA (%) + ShearRate(/s)
            @output:    expected abs viscosity of the hydrogel
        """
        features = np.atleast_2d(features)
        # pred_val：模型的预测值。
        # sigma：预测的不确定性（标准差）
        pred_val, sigma = self.__gp_model.predict(features, return_std=True)  # return [[predicted_value]]
        return pred_val[0], sigma[0]

    '''
        Pack (s, a, r, s') tuple

        @input:     s, a, s'
        @output:    Transition(s, a, r, s')

        @Note:      r: (mean + k * std)
    '''
    def pack_transition(self, current_state: State, action: float, next_state: State) -> Transition:

        next_state_pred_mean, next_state_pred_std = self.predict(next_state.get_ex_content())
        current_state_pred_mean, current_state_pred_std = self.predict(current_state.get_ex_content())
        k = 0
        delayed_reward = (
            (next_state_pred_mean - k * next_state_pred_std) -
            (current_state_pred_mean - k * current_state_pred_std)
        )
        # if exceed 150, reduce the reward
        if next_state_pred_mean > self.target:
            delayed_reward -= (next_state_pred_mean - self.target)

        return Transition(current_state, action, delayed_reward, next_state)
