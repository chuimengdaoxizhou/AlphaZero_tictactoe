import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange
from mtcs import Node, MCTS


# 定义AlphaZero类
class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        # 初始化AlphaZero类
        self.model = model  # AlphaZero模型
        self.optimizer = optimizer  # 优化器
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
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            # 获取模型的输出
            out_policy, out_value = self.model(state)

            # 计算损失
            policy_loss = F.cross_entropy(out_policy, policy_targets)  # 策略损失
            value_loss = F.mse_loss(out_value, value_targets)  # 价值损失
            loss = policy_loss + value_loss  # 总损失

            # 优化模型
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            
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

            # 保存模型和优化器的状态
            torch.save(self.model.state_dict(), 
                       f"D:\Progream\python\program\MTCS\model\{self.game}_model\model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), 
                       f"D:\Progream\python\program\MTCS\model\{self.game}_model\optimizer_{iteration}_{self.game}.pt")

