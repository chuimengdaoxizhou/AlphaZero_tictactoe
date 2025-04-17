import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# 定义井字棋游戏类
class TicTacToe:
    def __init__(self):
        # 初始化棋盘的行数和列数，井字棋固定为3x3的棋盘
        self.row_count = 3
        self.column_count = 3
        # 计算总的可用动作数量（9个格子）
        self.action_size = self.row_count * self.column_count

    # 返回井字棋游戏的名称
    def __repr__(self):
        return "TicTacToe"
        
    # 获取游戏的初始状态，返回一个3x3的零矩阵，表示棋盘为空
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    # 给定当前状态、动作和玩家，返回更新后的棋盘状态
    def get_next_state(self, state, action, player):
        # 计算当前动作对应的行和列
        row = action // self.column_count
        column = action % self.column_count
        # 将指定位置标记为当前玩家（1表示玩家1，-1表示玩家2）
        state[row, column] = player
        return state

    # 获取当前棋盘状态下所有有效的动作（即空位）
    def get_valid_moves(self, state):
        # 将棋盘展平成一维数组，空位置用0表示
        return (state.reshape(-1) == 0).astype(np.uint8)

    # 检查当前棋盘状态下是否有玩家获胜
    def check_win(self, state, action):
        # 如果没有动作（action为None），则返回False
        if action == None:
            return False
        
        # 根据动作计算所在的行和列
        row = action // self.column_count
        column = action % self.column_count
        # 获取当前玩家的标记（1或-1）
        player = state[row, column]

        # 检查是否有横向、纵向或对角线上的三个连续标记相同的格子
        return (
            # 横向检查：当前行的所有元素之和是否等于玩家标记 * 列数
            np.sum(state[row, :]) == player * self.column_count
            # 纵向检查：当前列的所有元素之和是否等于玩家标记 * 行数
            or np.sum(state[:, column]) == player * self.row_count
            # 主对角线检查：主对角线上的所有元素之和是否等于玩家标记 * 行数
            or np.sum(np.diag(state)) == player * self.row_count
            # 副对角线检查：副对角线上的所有元素之和是否等于玩家标记 * 行数
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    # 检查游戏是否结束并返回游戏的结果
    def get_value_and_terminated(self, state, action):
        # 如果当前动作导致某一方获胜，返回胜利标记和值1
        if self.check_win(state, action):
            return 1, True  # 1表示玩家获胜，游戏终止
        # 如果棋盘已满且没有胜者，返回平局值0，游戏终止
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True  # 0表示平局，游戏终止
        # 否则游戏继续
        return 0, False

    # 获取对手玩家的标记
    def get_opponent(self, player):
        return -player  # 如果当前玩家是1，则对手是-1，反之亦然

    # 获取对手玩家的胜利值
    def get_opponent_value(self, value):
        return -value  # 如果当前玩家获胜值是1，则对手获胜值为-1，反之亦然

    # 根据当前玩家的视角转换棋盘（将当前玩家标记转换为正值，方便计算）
    def change_perspective(self, state, player):
        return state * player  # 将棋盘的所有格子的标记乘以当前玩家的标记，玩家标记为1时不变，标记为-1时转换

    # 获取棋盘的编码状态，将棋盘转换为三个通道的状态：
    # 1. 玩家1的标记位置为True
    # 2. 空位的位置为True
    # 3. 玩家2的标记位置为True
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        return encoded_state  # 返回一个三通道的编码状态矩阵

# 定义神经网络模型
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        
        # 初始卷积层：将输入的3通道图像（例如棋盘状态）转换为num_hidden通道
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),  # 卷积层，输入通道为3，输出通道为num_hidden
            nn.BatchNorm2d(num_hidden),  # 批归一化，减少内部协方差偏移
            nn.ReLU()  # 激活函数，增加非线性
        )

        # 主干网络：由多个ResBlock组成
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]  # 创建num_resBlocks个残差块
        )

        # 策略头：用于输出动作分布
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),  # 卷积层，输入num_hidden通道，输出32通道
            nn.BatchNorm2d(32),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Flatten(),  # 扁平化，将多维张量转为一维
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)  # 全连接层，输出动作数量
        )

        # 价值头：用于输出棋局的评分（估计的值）
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),  # 卷积层，输入num_hidden通道，输出3通道
            nn.BatchNorm2d(3),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Flatten(),  # 扁平化
            nn.Linear(3 * game.row_count * game.column_count, 1),  # 全连接层，输出1个值（棋局评分）
            nn.Tanh()  # 使用Tanh将输出映射到[-1, 1]之间
        )

        self.to(device)  # 将模型发送到指定的设备（GPU/CPU）

    def forward(self, x):
        # 前向传播：输入x经过各层处理
        x = self.startBlock(x)  # 先经过初始卷积块
        for resBlock in self.backBone:  # 依次通过每个残差块
            x = resBlock(x)
        policy = self.policyHead(x)  # 计算策略（动作分布）
        value = self.valueHead(x)  # 计算价值（棋局评分）
        return policy, value  # 返回策略和价值

# 定义残差块（ResBlock）
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        # 残差块包含两层卷积，卷积后进行批归一化，并加上残差连接
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)  # 第1层卷积
        self.bn1 = nn.BatchNorm2d(num_hidden)  # 第1层卷积后的批归一化
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)  # 第2层卷积
        self.bn2 = nn.BatchNorm2d(num_hidden)  # 第2层卷积后的批归一化

    def forward(self, x):
        residual = x  # 保存输入x，用于残差连接
        # 经过第一层卷积、批归一化和ReLU激活函数
        x = F.relu(self.bn1(self.conv1(x)))  
        # 经过第二层卷积和批归一化
        x = self.bn2(self.conv2(x))
        x += residual  # 加上残差连接
        x = F.relu(x)  # 最后通过ReLU激活函数
        return x  # 返回经过残差连接和激活后的结果
