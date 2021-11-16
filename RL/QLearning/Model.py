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
LAMDA = 0.99  # 衰减率
MAX_EPISODE = 13  #
FRESH_TIME = 0.1  # AGENT 决策时间间隔


def init_table(n_state, actions):
    """
    初始化q-table
    :param n_state:
    :param actions:
    :return:
    """
    return pd.DataFrame(np.zeros((n_state, len(actions))), columns=actions)



def choose_action(q_table, state):
    """
    根据当前state从q_table中选择action
    :return:
    """
    state_actions = q_table.iloc[state,:]
    if np.random.uniform()> EPSILON or (state_actions.all()==0): # 随机选取
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    """
    根据当前S、A 更新S并得到R
    :param S: 当前状态
    :param A: 动作
    :return: 新的状态和奖励
    """
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
    env_list = ["-"]*(N_STATE-1) + ["T"]
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

def rl():
    q_table = init_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODE):
        step_counter = 0
        # 初始状态
        S = 0
        is_terminal = False
        update_env(S, episode, step_counter)
        while not is_terminal:
            A = choose_action(q_table, S)  # 根据当前状态选择ACTION
            S_, R = get_env_feedback(S, A)  # 根据ACTION和S（状态）获取奖励、更新状态
            q_predict = q_table.loc[S,A]
            if S_ != "terminal":
                q_target = R + LAMDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminal = True
            q_table.loc[S,A] += ALPHA * (q_target-q_predict)
            S = S_

            update_env(S, episode, step_counter+1)
            step_counter+=1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print(q_table)
