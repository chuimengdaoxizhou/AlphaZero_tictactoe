import math
import numpy as np
import torch


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
        for action, prob in enumerate(policy):
            if prob > 0:  # 只对概率大于0的动作进行扩展
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)  # 根据动作更新状态
                child_state = self.game.change_perspective(child_state, player=-1)  # 反转视角（换对手）
        
                child = Node(self.game, self.args, child_state, self, action, prob)  # 创建新子节点
                self.children.append(child)  # 将新子节点添加到子节点列表
                
        return child  # 返回最后一个扩展的子节点

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