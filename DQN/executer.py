"""
@File    ：executer.py
@Author  ：Dingding Chen <dingding.chen18@student.xjtlu.edu.cn>
@Date    ：2024/9/3 11:04 
@Desc    ：Description of the file
"""
from DQN.dqn_agent import DQN_AGENT
from replay_buffer import ReplayBuffer
from surrogate import Surrogate
import sys
from tqdm import *
from arguments import GP_MODEL_PATH, DQL_AGENT_PATH
from datetime import datetime


def execute():
    # -------------------------------------------------------------------------
    # parameters setup
    replay_memory_capacity = 3000
    training_epochs = 30000
    batch_size = 64
    target = 2
    print('Repaly memory capacity: {}'.format(replay_memory_capacity))
    print('training epochs: {}'.format(training_epochs))

    # -------------------------------------------------------------------------
    # prepare Gaussian Process Regression model for environment surrogate
    surrogate = Surrogate(target=target)

    # -------------------------------------------------------------------------
    # instantiate DQN agent
    replay_memory = ReplayBuffer(surrogate, capacity=replay_memory_capacity)
    agnt = DQN_AGENT(surrogate, replay_memory, batch_size=batch_size)

    # every (training_epochs // proposition_logs) per composition proposition.
    proposition_logs = 100
    proposition_log_res = list()

    # -------------------------------------------------------------------------
    # train DQN agent
    for _ in tqdm(range(proposition_logs), file=sys.stdout):
        # set need_training explicitly
        # train DQN with desired epochs
        agnt.train(training_epochs=training_epochs // proposition_logs)

        # Knowledge evaluation
        print('Knowledge evaluation:')
        proposed_composition = agnt.propose_next_experiment()
        pred_enthalpy = surrogate.predict(proposed_composition)
        print('Proposed experiments [HAMA, GELMA, Shear_Rate]: {} with predicted viscosity of {}'. \
              format(proposed_composition, 10 ** pred_enthalpy[0]))
        print('----------------------------------------------------------------------------------------')
        proposition_log_res.append((proposed_composition, pred_enthalpy))
    print("Finish training:")
    # proposition_log_res = proposition_log_res.sort(key=lambda x: x[1][0], reverse=True)
    proposition_log_res = sorted(proposition_log_res, key=lambda x: x[1][0], reverse=True)
    top_ten = proposition_log_res[:10]
    now = datetime.now()

    with open('proposed compositions-{}-{}-{}.txt'.format(now.year, now.month, now.day), 'wt') as f:
        for proposed_composition, pred_enthalpy in top_ten:
            f.write('Proposed composition [Ti, Ni, Cu, Hf, Zr]: {} with predicted enthalpy of {}\n'. \
                    format(proposed_composition, pred_enthalpy))

    # save trained DQN model
    agnt.save_knowledge(DQL_AGENT_PATH)
    agnt.save_training_indicators()
    surrogate.save_gp_model(GP_MODEL_PATH)


if __name__ == '__main__':
    execute()
