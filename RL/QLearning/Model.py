"""
简单学习强化学习Q-Learning
"""
import numpy as np
import pandas as pd
import time

#np.random.seed(2) # reproducible

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




def choose_action(q_table, state):
    """
    根据当前state从q_table中选择action
    :return:
    """
    state_actions = q_table.iloc[state,:]
    if np.random.uniform()> EPSILON or (state_actions.all()==0): # 随机选取
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

q_table = init_table(N_STATE, ACTIONS)
print(choose_action(q_table, 3))

def get_env_feedback(S, A):
    if A == "right":
        if S == N_STATE-2:
            S_ = "terminal"
            R = 1
        else:
            S_ = S+1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S-1
    return S_, R

def update_env(S, episode, step_counter):
    env_list = []*(N_STATE-1) + ["T"]
    if S == "terminal":
        interaction = f"episode {episode}: total_steps = {step_counter}"
        print("\r{}".format(interaction), end="")
        time.sleep(2)
        print("\r                       ", end="")
    else:
        env_list[S]="o"
        interaction = ''.join(env_list)
        print("\r{}".format(interaction), end="")
        time.sleep(FRESH_TIME)

