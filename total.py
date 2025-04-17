import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.manual_seed(0)

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

# 定义MCTS树的节点
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        """
        初始化一个MCTS树的节点

        :param game: 游戏实例
        :param args: 参数字典，包含MCTS的超参数
        :param state: 当前节点的游戏状态
        :param parent: 父节点，默认为None
        :param action_taken: 父节点采取的动作，默认为None
        :param prior: 先验概率，默认为0
        :param visit_count: 访问次数，默认为0
        """
        self.game = game  # 游戏实例
        self.args = args  # 参数字典
        self.state = state  # 当前节点的游戏状态
        self.parent = parent  # 父节点
        self.action_taken = action_taken  # 父节点采取的动作
        self.prior = prior  # 先验概率

        self.children = []  # 子节点列表

        self.visit_count = visit_count  # 访问次数
        self.value_sum = 0  # 价值总和，用于回溯时计算平均值

    def is_fully_expanded(self):
        """
        判断当前节点是否已经完全展开，即是否有子节点
        :return: 如果有子节点返回True，否则返回False
        """
        return len(self.children) > 0

    def select(self):
        """
        选择当前节点的最佳子节点，使用UCB（上置信界）选择策略
        :return: 最优的子节点
        """
        best_child = None
        best_ucb = -np.inf  # 初始化最优UCB为负无穷

        for child in self.children:
            ucb = self.get_ucb(child)  # 计算每个子节点的UCB值
            if ucb > best_ucb:  # 更新最优子节点
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        """
        计算一个子节点的UCB值
        UCB = Q + C * sqrt(N) / (n + 1) * P
        其中Q为子节点的平均值，C为探索因子，N为当前节点的访问次数，n为子节点的访问次数，P为先验概率

        :param child: 子节点
        :return: 子节点的UCB值
        """
        if child.visit_count == 0:
            q_value = 0  # 如果子节点未被访问，则Q值为0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2  # Q值调整为-1到1之间

        # 计算UCB值
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) /
                            (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        """
        展开当前节点，生成所有有效动作的子节点
        :param policy: 当前状态下每个动作的概率分布
        :return: 最后一个扩展的子节点
        """
        for action, prob in enumerate(policy):  # action为动作索引，prob为概率
            if prob > 0:  # 只对概率大于0的动作进行扩展
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)  # 根据动作更新状态
                child_state = self.game.change_perspective(child_state, player=-1)  # 反转视角（换对手）
        
                child = Node(self.game, self.args, child_state, self, action, prob)  # 创建新子节点
                self.children.append(child)  # 将新子节点添加到子节点列表

        # 返回最后一个扩展的子节点，后续不关心返回值，这个函数是将有效动作的子节点添加到当前节点的children列表中        
        return child  

    def backpropagate(self, value):
        """
        回溯更新当前节点到根节点的路径上的所有节点的访问次数和价值
        :param value: 当前节点的价值
        """
        self.value_sum += value  # 更新节点的价值总和
        self.visit_count += 1  # 增加节点的访问次数

        value = self.game.get_opponent_value(value)  # 获取对手的价值（对手的价值是相反的）
        if self.parent is not None:
            self.parent.backpropagate(value)  # 如果有父节点，则继续回溯更新父节点

# 定义MCTS（蒙特卡洛树搜索）类
class MCTS:
    def __init__(self, game, args, model):
        """
        初始化MCTS（蒙特卡洛树搜索）实例

        :param game: 游戏实例
        :param args: 参数字典
        :param model: 神经网络模型，用于获取策略和价值
        """
        self.game = game  # 游戏实例
        self.args = args  # 参数字典
        self.model = model  # 神经网络模型

    @torch.no_grad()
    def search(self, state):
        """
        进行MCTS搜索，返回当前状态下每个动作的概率分布

        :param state: 当前游戏状态
        :return: 当前状态下每个动作的概率分布
        """
        # 定义根节点
        root = Node(self.game, self.args, state, visit_count=1)
        
        # 使用神经网络预测当前状态下的策略和价值
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()  # 计算策略的概率分布
        # 添加Dirichlet噪声，以保证探索性
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)  # 获取当前状态下的有效动作
        policy *= valid_moves  # 只保留有效动作的概率
        policy /= np.sum(policy)  # 归一化策略

        # 扩展根节点
        root.expand(policy)
        
        # 进行多次搜索
        for search in range(self.args['num_searches']):
            node = root
            # 选择阶段：根据UCB选择子节点
            while node.is_fully_expanded():
                node = node.select()

                # 评估当前节点
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)  # 获取对手的价值

                if not is_terminal:  # 如果当前节点不是终止状态
                    # 使用模型计算策略和价值
                    policy, value = self.model(
                        torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                    )
                    policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                    valid_moves = self.game.get_valid_moves(node.state)
                    policy *= valid_moves  # 只保留有效动作的概率
                    policy /= np.sum(policy)  # 归一化策略

                    value = value.item()  # 转换为Python标量

                    # 扩展当前节点
                    node.expand(policy)

                # 回溯更新节点的访问次数和价值
                node.backpropagate(value)

        # 计算每个动作的概率分布
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count  # 根据访问次数计算每个动作的概率
        action_probs /= np.sum(action_probs)  # 归一化概率分布
        return action_probs  # 返回动作概率分布

# 定义AlphaZero类
class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        # 初始化AlphaZero类
        self.model = model  # AlphaZero模型
        self.optimizer = optimizer  # 优化器
        # # 当验证损失停止下降时，减少学习率（factor 是学习率的衰减因子）
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.1)
        self.game = game  # 游戏环境
        self.args = args  # 超参数设置
        self.mcts = MCTS(game, args, model)  # 初始化蒙特卡罗树搜索（MCTS）

    def selfPlay(self):
        # 自对弈过程
        memory = []  # 用于存储游戏记录
        player = 1  # 当前玩家，1表示玩家一
        state = self.game.get_initial_state()  # 获取游戏初始状态

        while True:
            # 改变视角，使得当前玩家总是以自己的角度看游戏
            neutral_state = self.game.change_perspective(state, player)
            # 使用MCTS搜索得到当前状态下的动作概率分布
            action_probs = self.mcts.search(neutral_state)

            # 记录当前状态、动作概率和当前玩家
            memory.append((neutral_state, action_probs, player))
            # 对动作概率进行温度调节，使得探索变得更加平衡
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)  # 归一化

            # 根据温度调节后的动作概率随机选择一个动作
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            # 更新状态
            state = self.game.get_next_state(state, action, player)

            # 获取当前状态的价值以及是否为终局
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                # 如果游戏结束，计算并返回每一步的奖励
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    # 计算每一步的最终奖励
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),  # 游戏状态编码
                        hist_action_probs,  # 动作概率
                        hist_outcome  # 该步骤的最终奖励
                    ))
                return returnMemory  # 返回游戏过程中的状态、动作概率和最终奖励
            # 更换玩家
            player = self.game.get_opponent(player)

    def train(self, memory):
        # 训练模型
        random.shuffle(memory)  # 打乱记忆库
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            # 从记忆库中按批次取样
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            # 转换为NumPy数组并进行类型转换
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1,1)
            
            # 将数据转换为Tensor并移至计算设备
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.long, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            # 获取模型的输出
            out_policy, out_value = self.model(state)

            # **在这里，将 policy_targets 转换为 float32 类型**
            policy_targets = policy_targets.float()  # 显式转换为 float32

            # 计算损失
            policy_loss = F.cross_entropy(out_policy, policy_targets)  # 策略损失
            value_loss = F.mse_loss(out_value, value_targets)  # 价值损失
            loss = policy_loss + value_loss  # 总损失

            # 优化模型
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            self.scheduler.step(loss)   # 优化器学习率调度
            
    def learn(self):
        # 训练循环
        for iteration in range(self.args['num_iterations']):
            memory = []  # 存储每次自对弈的记忆

            # 进行自对弈并积累记忆
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()  # 设置模型为训练模式
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)  # 进行训练

            # 保存模型和优化器的状态 D:\Progream\python\program\AlphaZero_tictactoe\model\tictactoe_model
            torch.save(self.model.state_dict(), 
                       f"D:\Progream\python\program\AlphaZero_tictactoe\model\\tictactoe_model\model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), 
                       f"D:\Progream\python\program\AlphaZero_tictactoe\model\\tictactoe_model\optimizer_{iteration}_{self.game}.pt")


# 初始化井字棋游戏环境
tictactoe = TicTacToe()

# 判断是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化ResNet模型，输入参数包括游戏环境、网络深度（4层）、每层的通道数（64）和设备（CPU或GPU）
model = ResNet(tictactoe, 4, 64, device)

# 使用Adam优化器，学习率设置为0.001，权重衰减（L2正则化）为0.0001
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)

# 配置训练的超参数
args = {
    'C' : 2,  # 游戏的玩家数量（井字棋为2个玩家）
    'num_searches': 10,  # 每次MCTS搜索的次数
    'num_iterations': 2,  # 总共的训练迭代次数
    'num_selfPlay_iterations': 100,  # 每次学习过程中进行的自对弈次数
    'num_epochs': 4,  # 每次学习中进行的训练轮数
    'batch_size': 64,  # 每批训练数据的大小
    'temperature': 1.25,  # 动作概率分布的温度，用于调节探索的多样性
    'dirichlet_epsilon': 0.25,  # Dirichlet噪声的参数，控制探索程度
    'dirichlet_alpha': 0.3  # Dirichlet噪声的强度
}

# 创建AlphaZero实例，并开始学习
alphaZero = AlphaZero(model, optimizer, tictactoe, args)

# 启动学习过程
alphaZero.learn()


# 初始化井字棋游戏环境
tictactoe = TicTacToe()

# 判断是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取游戏的初始状态
state = tictactoe.get_initial_state()

# 进行几次操作，模拟玩家在游戏中的轮流动作（2, 4, 6, 8位置）
state = tictactoe.get_next_state(state, 2, -1)  # 玩家-1在位置2下子
state = tictactoe.get_next_state(state, 4, -1)  # 玩家-1在位置4下子
state = tictactoe.get_next_state(state, 6, 1)   # 玩家1在位置6下子
state = tictactoe.get_next_state(state, 8, 1)   # 玩家1在位置8下子

# 打印当前游戏状态
print("Current Game State:")
print(state)

# 获取当前状态的编码表示（通常是经过某种转换处理的游戏状态）
encoded_state = tictactoe.get_encoded_state(state)

# 打印编码后的状态
print("Encoded State:")
print(encoded_state)

# 将编码后的状态转为PyTorch张量并调整维度，适配模型输入
tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

# 初始化ResNet模型，设置网络深度、通道数和设备（CPU或GPU）
model = ResNet(tictactoe, 4, 64, device=device)

# D:\Progream\python\program\AlphaZero_tictactoe\model\tictactoe_model
# 加载预训练模型（如果有的话）
model.load_state_dict(torch.load(
    f'D:\Progream\python\program\AlphaZero_tictactoe\model\\tictactoe_model\model_9_TicTacToe.pt', 
    map_location=device
))

# 将模型设置为评估模式，关闭dropout等训练时的特性
model.eval()

# 使用模型进行前向传播，得到策略和价值的预测
policy, value = model(tensor_state)

# 提取价值（scalar值）并将其转化为Python数据类型
value = value.item()

# 使用softmax函数计算动作概率分布（策略），并将结果转为NumPy数组
policy = torch.softmax(policy, dim=1).squeeze(0).detach().cpu().numpy()

# 打印价值和动作概率分布
print("Value (Game Evaluation):", value)
print("Action Probabilities (Policy):")
print(policy)

# 使用matplotlib绘制条形图，显示各个动作的概率
plt.bar(range(tictactoe.action_size), policy)
plt.xlabel('Action')
plt.ylabel('Probability')
plt.title('Action Probabilities for TicTacToe')
plt.show()
