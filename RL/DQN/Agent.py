import torch
import torch.nn as nn



class Agent():
    """
    Agent 可以观察环境状态、评判状态下每个动作的值、选择最优的动作与环境交互
    """
    def __init__(self):
        """
        初始化动作
        """
        pass

    def observe(self):
        """
        观察环境状态
        :return:
        """
        pass

    def value(self):
        """
        评判状态下每个动作的值
        :return:
        """
        pass

    def act(self):
        """
        选取最优的动作
        :return:
        """
        pass


class DQN(nn.Module):
    """
    相比于Q-Learning在内存中维护一个场景-行为决策表（Q表）
    DQN使用神经网络来代替Q表
    """
    def __init__(self, in_channels=4, num_actions=5):
        super(DQN, self).__init__
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        
    pass


print(torch.__version__)