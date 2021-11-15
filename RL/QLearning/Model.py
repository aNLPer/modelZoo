"""
简单学习强化学习Q-Learning
"""
import numpy as np
import pandas as pd
import time

np.random.seed(2) # reproducible

N_STATE = 6 # 状态
ACTIONS = ["left", "right"] # 可选的行为
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # 学习率
LAMDA = 0.9  # 衰减率
MAX_EPISODE = 13  #
FRESH_TIME = 0.6  # AGENT 决策时间间隔


def init_table(n_state, actions):
    """
    初始化q-table
    :param n_state:
    :param actions:
    :return:
    """
    table = pd.DataFrame(np.zeros((n_state, len(actions))), columns=actions)
    print(table)
    return table

init_table(N_STATE, ACTIONS)


def choose_action():
    pass