from alphazero import *
from tictactoe_resnet import *

# 初始化井字棋游戏环境
game = TicTacToe()

# 判断是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化ResNet模型，输入参数包括游戏环境、网络深度（4层）、每层的通道数（64）和设备（CPU或GPU）
model = ResNet(game, 4, 64, device)

# 使用Adam优化器，学习率设置为0.001，权重衰减（L2正则化）为0.0001
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)

# 配置训练的超参数
args = {
    'C' : 2,  # 游戏的玩家数量（井字棋为2个玩家）
    'num_searches': 60,  # 每次MCTS搜索的次数
    'num_iterations': 2,  # 总共的训练迭代次数
    'num_selfPlay_iterations': 500,  # 每次学习过程中进行的自对弈次数
    'num_epochs': 4,  # 每次学习中进行的训练轮数
    'batch_size': 64,  # 每批训练数据的大小
    'temperature': 1.25,  # 动作概率分布的温度，用于调节探索的多样性
    'dirichlet_epsilon': 0.25,  # Dirichlet噪声的参数，控制探索程度
    'dirichlet_alpha': 0.3  # Dirichlet噪声的强度
}

# 创建AlphaZero实例，并开始学习
alphaZero = AlphaZero(model, optimizer, game, args)

# 启动学习过程
alphaZero.learn()
