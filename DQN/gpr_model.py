"""
@File    ：gpr_model.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/3 17:00 
@Desc    ：To evaluate the effect of gpr model
"""
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

FEATURE_VISCOSITY_PATH = '../data/data-modi-vistr.xlsx'
data = pd.read_excel(FEATURE_VISCOSITY_PATH)
feature_viscosity_buffer = data.to_numpy()
x = [item[0:3].tolist() for item in feature_viscosity_buffer]
y = [item[7] for item in feature_viscosity_buffer]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=50)
# Define the kernel
kernel = ConstantKernel() * RBF() + WhiteKernel()
gpr_model = GaussianProcessRegressor(kernel=kernel)
# gpr_model.fit(x_train, y_train)
# # Predict on the test set
# gpr_pred, gpr_std = gpr_model.predict(x_test, return_std=True)
# print(len(gpr_std))
# # Evaluate the model on the test set
# gpr_r2 = r2_score(y_test, gpr_pred)
# print(f'Test Set R2 Score: {gpr_r2}')
# gpr_mse = mean_squared_error(y_test, gpr_pred)
# print(f'Test Set Mean Squared Error: {gpr_mse}')

# gpr_model = RandomForestRegressor()

gpr_model.fit(x, y)
rf_pred_1 = gpr_model.predict([[0.5, 7.5, 1]])
rf_pred_2 = gpr_model.predict([[0.972, 7.5, 1]])
print(rf_pred_1)
print(rf_pred_2)


# plt.scatter(y_test, gpr_pred, alpha=0.5)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
# plt.title('GPR model')
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.show()
