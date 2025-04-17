import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class InputEmbedding2(nn.Module):
    def __init__(self, patch_size=16, n_channels=3, device='cpu', latent_size=768):
        super(InputEmbedding2, self).__init__()
        self.latent_size = latent_size  # 潜在空间的大小
        self.patch_size = patch_size  # 图像块的大小
        self.n_channels = n_channels  # 图像的通道数（例如RGB为3）
        self.device = device  # 设备（CUDA or CPU）
        self.input_size = self.patch_size * self.patch_size * self.n_channels  # 每个小块的尺寸

        # 定义线性投影层，将每个图像块映射到潜在空间
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

        # 随机初始化 [class] token，该 token 会被加到线性投影结果的最前面
        self.class_token = nn.Parameter(torch.randn(1, self.latent_size)).to(self.device)

        # 定义位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.latent_size)).to(self.device)

    def forward(self, input_data):
        # input_data should now be (Time_steps, Channels, Height, Width)


        time_steps = input_data.shape[0]
        frame_embeddings = [] # List to store embeddings for each frame

        for t in range(time_steps):
            frame = input_data[t] # Extract a single frame (Channels, Height, Width)


            frame = frame.to(self.device)  # 将输入数据移到指定设备
            # 获取输入图像的尺寸
            c, h, w = frame.shape  # 只取通道数、高度、宽度

            # 计算需要的填充量，确保图像的尺寸能够被patch_size整除
            pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

            # 对图像进行填充
            if pad_h > 0 or pad_w > 0:
                frame = F.pad(frame, (0, pad_w, 0, pad_h), mode='constant', value=0)

            # 重新获取填充后的尺寸
            c, h, w = frame.shape

            # 将图像数据分成小块
            patches = einops.rearrange(
                frame, 'c (h h1) (w w1) -> (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)

            # 将每个图像块通过线性投影映射到潜在空间
            linear_projection = self.linearProjection(patches).to(self.device)
            n, _ = linear_projection.shape  # n: 小块数量

            # 将 [class] token 拼接到原始线性投影的最前面
            linear_projection = torch.cat((self.class_token, linear_projection), dim=0)

            # 扩展位置编码的维度，适配当前小块数目
            pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b n d', n=n + 1)

            # 将位置编码加到线性投影上
            linear_projection += pos_embed.squeeze(0)  # 去掉多余的批量维度

            frame_embeddings.append(linear_projection) # Append embedding for current frame

        # Stack frame embeddings along the time dimension (dim=0)
        frame_embeddings_sequence = torch.stack(frame_embeddings, dim=0)

        return frame_embeddings_sequence


class InputEmbedding(nn.Module):
    def __init__(self, patch_size=16, n_channels=3, device='cpu', latent_size=768):
        super(InputEmbedding, self).__init__()
        self.latent_size = latent_size  # 潜在空间的大小
        self.patch_size = patch_size  # 图像块的大小
        self.n_channels = n_channels  # 图像的通道数（例如RGB为3）
        self.device = device  # 设备（CUDA or CPU）
        self.input_size = self.patch_size * self.patch_size * self.n_channels  # 每个小块的尺寸

        # 定义线性投影层，将每个图像块映射到潜在空间
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

        # 随机初始化 [class] token，该 token 会被加到线性投影结果的最前面
        self.class_token = nn.Parameter(torch.randn(1, self.latent_size)).to(self.device)

        # 定义位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.latent_size)).to(self.device)

    def forward(self, input_data):
        print(f"Input shape: {input_data.shape}")
        input_data = input_data.squeeze(0)
        input_data = input_data.to(self.device)  # 将输入数据移到指定设备
        print(f"Input shape translated: {input_data.shape}")
        # 获取输入图像的尺寸
        c, h, w = input_data.shape  # 只取通道数、高度、宽度

        # 计算需要的填充量，确保图像的尺寸能够被patch_size整除
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size

        # 对图像进行填充
        if pad_h > 0 or pad_w > 0:
            input_data = F.pad(input_data, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # 重新获取填充后的尺寸
        c, h, w = input_data.shape

        # 将图像数据分成小块
        patches = einops.rearrange(
            input_data, 'c (h h1) (w w1) -> (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)

        # 将每个图像块通过线性投影映射到潜在空间
        linear_projection = self.linearProjection(patches).to(self.device)
        n, _ = linear_projection.shape  # n: 小块数量

        # 将 [class] token 拼接到原始线性投影的最前面
        linear_projection = torch.cat((self.class_token, linear_projection), dim=0)

        # 扩展位置编码的维度，适配当前小块数目
        pos_embed = einops.repeat(self.pos_embedding, 'b 1 d -> b n d', n=n + 1)

        # 将位置编码加到线性投影上
        linear_projection += pos_embed.squeeze(0)  # 去掉多余的批量维度

        return linear_projection


class VitTransformer(nn.Module):
    def __init__(self, N, patch_size=16, n_channels=3, latent_size=768, device='cuda', num_classes=1000, dropout=0.1):
        super(VitTransformer, self).__init__()
        self.N = N  # 编码器层数（由N参数指定）
        self.latent_size = latent_size  # 潜在空间大小
        self.device = device  # 设备
        self.num_classes = num_classes  # 类别数
        self.dropout = dropout  # dropout比例
        self.patch_size = patch_size # 图像块大小
        self.n_channels = n_channels  # 图像通道数
        # 创建输入嵌入层
        self.embedding = InputEmbedding(self.patch_size, self.n_channels, device=self.device, latent_size=self.latent_size)

        # 创建由N个编码器块组成的堆叠
        self.encStack = nn.ModuleList([EncoderBlock(latent_size=self.latent_size) for _ in range(self.N)])

        # 分类头：用于将最终的 [class] token 映射到类别空间
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size, self.num_classes),
        )

    def forward(self, test_input):
        # 输入图像进行嵌入（图像分块 + 线性投影 + 位置编码）
        enc_output = self.embedding(test_input)

        # 将嵌入后的图像通过所有编码器层
        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)

        # 提取 [class] token 的输出（通常位于第一个位置）
        cls_token_embedding = enc_output[0]

        # 返回最终的分类结果
        return self.MLP_head(cls_token_embedding)

class PolicyNetwork(nn.Module):
    def __init__(self, N, patch_size=16, n_channels=3, latent_size=768, device='cpu', num_actions=10, dropout=0.1):
        super(PolicyNetwork, self).__init__()
        self.N = N  # 编码器层数（由N参数指定）
        self.latent_size = latent_size  # 潜在空间大小
        self.device = device  # 设备
        self.num_actions = num_actions  # 动作空间大小
        self.dropout = dropout  # dropout比例
        self.patch_size = patch_size  # 图像块大小
        self.n_channels = n_channels  # 图像通道数

        # 创建ViT的输入嵌入层
        self.embedding = InputEmbedding(self.patch_size, self.n_channels, device=self.device, latent_size=self.latent_size)

        # 创建由N个编码器块组成的堆叠
        self.encStack = nn.ModuleList([EncoderBlock(latent_size=self.latent_size) for _ in range(self.N)])

        # 分类头：输出每个动作的概率
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size, self.num_actions),
            nn.Softmax(dim=-1)  # 动作概率分布
        )

    def forward(self, state_input):
        # 输入图像进行嵌入（图像分块 + 线性投影 + 位置编码）
        enc_output = self.embedding(state_input)

        # 将嵌入后的图像通过所有编码器层
        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)

        # 提取 [class] token 的输出（通常位于第一个位置）
        cls_token_embedding = enc_output[0]

        # 返回每个动作的概率分布
        return self.MLP_head(cls_token_embedding)

class ValueNetwork(nn.Module):
    def __init__(self, N, patch_size=16, n_channels=3, latent_size=768, device='cpu', dropout=0.1):
        super(ValueNetwork, self).__init__()
        self.N = N  # 编码器层数（由N参数指定）
        self.latent_size = latent_size  # 潜在空间大小
        self.device = device  # 设备
        self.dropout = dropout  # dropout比例
        self.patch_size = patch_size  # 图像块大小
        self.n_channels = n_channels  # 图像通道数

        # 创建ViT的输入嵌入层
        self.embedding = InputEmbedding(self.patch_size, self.n_channels, device=self.device, latent_size=self.latent_size)

        # 创建由N个编码器块组成的堆叠
        self.encStack = nn.ModuleList([EncoderBlock(latent_size=self.latent_size) for _ in range(self.N)])

        # 价值头：输出状态的价值（一个标量）
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size, 1)  # 状态价值输出
        )

    def forward(self, state_input):
        # 输入图像进行嵌入（图像分块 + 线性投影 + 位置编码）
        enc_output = self.embedding(state_input)

        # 将嵌入后的图像通过所有编码器层
        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)

        # 提取 [class] token 的输出（通常位于第一个位置）
        cls_token_embedding = enc_output[0]

        # 返回状态的价值
        return self.MLP_head(cls_token_embedding)

class VitTransformerForGo_PolicyPerPatch(nn.Module): # 修改类名以区分
    def __init__(self, N, patch_size=1, n_channels=3,
                 latent_size=768, device='cuda', dropout=0.1): # patch_size=1
        super(VitTransformerForGo_PolicyPerPatch, self).__init__()
        self.N = N
        self.latent_size = latent_size
        self.device = device
        self.dropout = dropout
        self.patch_size = patch_size # 图像块大小为 1
        self.n_channels = n_channels
        self.embedding = InputEmbedding2(self.patch_size, self.n_channels, device=self.device, latent_size=self.latent_size)
        self.encStack = nn.ModuleList([EncoderBlock(latent_size=self.latent_size) for _ in range(self.N)])
        self.train_flag = False
        # Value Head (保持不变，使用 cls_token)
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, 1),
            nn.Tanh()
        )

        # Policy Head (基于 19x19 图像块 tokens)
        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, 1), # 将每个 token 映射到 1 个标量值
        )
        self.softmax = nn.Softmax(dim=-1) #  Softmax 激活函数

    def forward(self, test_input):

        enc_output = self.embedding(test_input)  # Output shape should be (1, 226, latent_size)
        if self.train_flag:
            print(f"enc_output.shape {enc_output.shape}")
        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)
        if self.train_flag:
            print(f"enc_output_changed.shape {enc_output.shape}")
        # Assuming enc_output is now [1, 226, latent_size]
        cls_token_embedding = enc_output[:, 0, :]  # Shape: [1, latent_size] - Select CLS token for batch 0
        if self.train_flag:
            print(f"cls_token_embedding.shape {cls_token_embedding.shape}")
        patch_token_embeddings = enc_output[:, 1:, :]  # Shape: [1, 225, latent_size] - Select patch tokens for batch 0
        if self.train_flag:
            print(f"patch_token_embeddings.shape {patch_token_embeddings.shape}")
        value = self.value_head(cls_token_embedding)  # Value Head 输出, 形状 (1, 1) or (1,)
        if self.train_flag:
            print(f"value.shape {value.shape}")

        # Policy Head 处理 15*15 patch tokens
        policy_logits_flattened = self.policy_head(
            patch_token_embeddings.squeeze(0))  # Shape: [225, 1] - Remove batch dim for policy head
        if self.train_flag:
            print(f"policy_logits_flattened.shape {policy_logits_flattened.shape}")
        policy_logits = policy_logits_flattened.reshape(15, 15)  # Reshape to 15*15
        if self.train_flag:
            print(f"policy_logits.shape {policy_logits.shape}")
        policy = self.softmax(
            policy_logits.flatten(0).unsqueeze(0))  # Flatten to (361,), apply softmax, and unsqueeze to (1, 361)
        if self.train_flag:
            print(f"policy.shape {policy.shape}")

        return policy, value  # 返回 Policy (形状 (1, 361)) 和 Value (形状 (1, ))

# 以下是为了测试而简化的 EncoderBlock 实现
class EncoderBlock(nn.Module):
    def __init__(self, latent_size=768):
        super(EncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=latent_size, num_heads=8)
        self.ffn = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )
        self.layer_norm1 = nn.LayerNorm(latent_size)
        self.layer_norm2 = nn.LayerNorm(latent_size)

    def forward(self, x):
        # 自注意力层
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)  # 残差连接
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)  # 残差连接
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class GoViT(nn.Module):
    def __init__(self, num_history=4, img_size=19, patch_size=3, in_channels=3, device='cpu',
                 embed_dim=768, num_layers=8, num_head=12, mlp_ratio=4.0, dropout=0.1):
        """
        初始化GoViT模型
        参数：
        - num_history: 历史步数
        - img_size: 棋盘大小
        - patch_size: patch大小
        - in_channels: 每个patch的通道数
        - embed_dim: Transformer嵌入维度
        - num_layers: Transformer编码器层数
        - num_heads: 注意力头数
        - mlp_ratio: MLP隐藏层维度与嵌入维度比率
        - dropout: dropout概率
        """
        super(GoViT, self).__init__()
        self.num_history = num_history
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device

        # 计算扩展后的棋盘大小，使其能被 patch_size 整除
        self.padded_size = ((img_size + patch_size - 1) // patch_size) * patch_size  # 向上取整到 patch_size 的倍数
        self.num_patches_per_img = (self.padded_size // patch_size) ** 2  # 每个图像的 patches 数量
        self.total_patches = num_history * self.num_patches_per_img  # 总 patches 数量

        # 嵌入层
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer 编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_head,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True
            ),
            num_layers=self.num_layers
        )

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, img_size * img_size),  # 输出仍然是 19*19，因为策略是针对原始棋盘
            # nn.Softmax(dim=-1)
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Tanh()
        )

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        """
        前向传播
        :param x: tensor, 形状为 (batch_size, num_history, in_channels, img_size, img_size)
        :return:
        - policy: 策略头输出，形状为 (batch_size, img_size * img_size)
        - value: 价值头输出，形状 (batch_size, 1)
        """
        batch_size = x.size(0)

        # 扩展棋盘大小到可以被 patch_size 整除，例如从 19x19 到 21x21
        pad_h = self.padded_size - self.img_size  # 需要填充的高度
        pad_w = self.padded_size - self.img_size  # 需要填充的宽度
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)  # 在 h 和 w 维度上填充

        # 重塑输入：将 num_history 合并到 batch_size
        # 输入形状：(batch_size, num_history, in_channels, padded_size, padded_size)
        # 输出形状：(batch_size * num_history, in_channels, padded_size, padded_size)
        x = x.view(batch_size * self.num_history, 3, self.padded_size, self.padded_size)

        # 提取 patches 并嵌入
        x = self.patch_embed(x)  # 输出形状：(batch_size * num_history, embed_dim, num_patches_per_img)
        x = x.flatten(2).transpose(1, 2)  # 重塑为 (batch_size * num_history, num_patches_per_img, embed_dim)

        # 重塑回 batch 维度
        x = x.reshape(batch_size, self.total_patches, self.embed_dim)

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过 Transformer 编码器
        x = self.transformer(x)

        # 提取 [CLS] token 的输出用于价值头
        cls_output = x[:, 0]
        value = self.value_head(cls_output)

        # 提取 patches 的输出用于策略头
        patches_output = x[:, 1:]
        patches_output = patches_output.mean(dim=1)  # 平均池化
        policy = self.policy_head(patches_output)

        return policy, value

# 输入图像的参数
batch_size = 2
channels = 3  # 例如 RGB 图像
height = 3
width = 3
patch_size = 16
latent_size = 768
N = 4  # 编码器层数
num_classes = 9  # 假设有10个类别
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 定义policy网络
policy_network = PolicyNetwork(N=4, 
                               patch_size=3, 
                               n_channels=3, 
                               latent_size=40, 
                               device=device, 
                               num_actions=9
                               )

# 定义value网络
value_network = ValueNetwork(N=4, 
                             patch_size=3,
                             n_channels=3, 
                             latent_size=40, 
                             device=device
                             )



import torch
import torch.nn as nn
import torch.nn.functional as F
import einops # 确保已安装 pip install einops

# 修改后的 InputEmbedding2，支持批处理
class InputEmbedding2_Batched(nn.Module):
    # patch_size=1 适用于围棋/五子棋，每个点一个 patch
    # 增加了 img_size 参数
    def __init__(self, patch_size=1, n_channels=3, device='cpu', latent_size=768, img_size=15):
        super().__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.device = device
        self.img_size = img_size # 假设是方形棋盘
        # 每个 patch 的输入大小 (对于 patch_size=1, 就是 n_channels)
        self.input_size_per_patch = self.patch_size * self.patch_size * self.n_channels
        # 计算 patch 的数量
        if img_size % patch_size != 0:
             raise ValueError("图像尺寸必须能被 patch_size 整除")
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # 线性投影层
        self.linearProjection = nn.Linear(self.input_size_per_patch, self.latent_size)

        # CLS token，增加一个批处理维度占位符
        self.class_token = nn.Parameter(torch.randn(1, 1, self.latent_size))

        # 位置编码 (针对空间维度 + CLS token)
        # 形状: (1, num_patches + 1, latent_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.latent_size))
        self.pos_embedding.to(self.device)
        self.class_token.to(self.device)


    def forward(self, input_data):
        # input_data 形状: (batch_size, seq_len, C, H, W)
        # 检查输入维度是否符合预期
        if input_data.ndim != 5:
             raise ValueError(f"Expected input_data to have 5 dimensions (B, T, C, H, W), but got {input_data.ndim}")

        batch_size, seq_len, C, H, W = input_data.shape
        input_data = input_data.to(self.device)

        # 检查图像尺寸是否匹配
        if H != self.img_size or W != self.img_size:
             raise ValueError(f"Input height/width ({H}x{W}) don't match expected img_size ({self.img_size})")

        # 将 batch 和 sequence 维度合并，以便进行批处理
        # 形状变为: (batch_size * seq_len, C, H, W)
        x = input_data.view(batch_size * seq_len, C, H, W)

        # --- (可选) 填充逻辑 ---
        # 对于 patch_size=1，且 H=W=15，通常不需要填充
        # 如果需要填充，逻辑应在这里，作用于 x
        # pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        # pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        # if pad_h > 0 or pad_w > 0:
        #    x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        # H, W = x.shape[-2:] # 更新 H, W

        # 使用 einops 将图像块重排
        # 输入 x 形状: (B*T, C, H, W)
        # 输出 patches 形状: (B*T, num_patches, patch_h * patch_w * C)
        patches = einops.rearrange(
            x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_size, p2=self.patch_size
        )
        # patches 形状: (batch_size * seq_len, num_patches, input_size_per_patch)

        # 线性投影
        # 输出 tokens 形状: (B*T, num_patches, latent_size)
        tokens = self.linearProjection(patches)

        # 准备 CLS token，扩展到当前的有效 batch size (B*T)
        # class_token 形状: (1, 1, latent_size) -> (B*T, 1, latent_size)
        cls_tokens = self.class_token.expand(batch_size * seq_len, -1, -1)

        # 将 CLS token 拼接到序列最前面
        # 输出 tokens 形状: (B*T, num_patches + 1, latent_size)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        # 添加位置编码 (利用 PyTorch 的广播机制)
        # self.pos_embedding 形状: (1, num_patches + 1, latent_size)
        tokens = tokens + self.pos_embedding

        # 返回处理后的 tokens，注意，batch 和 time 维度仍然是合并的
        # 输出形状: (batch_size * seq_len, num_patches + 1, latent_size)
        return tokens

class VitTransformerForGo_PolicyPerPatch(nn.Module):
    # 增加了 game_size 和 seq_len 参数
    def __init__(self, N=4, patch_size=1, n_channels=3, game_size=15, num_history=5, # seq_len = num_history + 1
                 latent_size=768, device='cpu', dropout=0.1):
        super().__init__()
        self.N = N
        self.game_size = game_size
        self.num_patches = game_size * game_size
        self.action_size = self.num_patches
        self.latent_size = latent_size
        self.device = device
        self.dropout_rate = dropout
        self.seq_len = num_history + 1 # 存储序列长度 (num_history + 1)

        # 使用支持批处理的 Embedding 类
        self.embedding = InputEmbedding2_Batched(patch_size, n_channels, device, latent_size, game_size)

        # Encoder 栈
        # 确保 EncoderBlock 能接收 dropout 参数 (如果需要)
        self.encStack = nn.ModuleList([EncoderBlock(latent_size=self.latent_size) for _ in range(self.N)])

        # Value Head (保持不变)
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, 1),
            nn.Tanh()
        )

        # Policy Head (保持不变)
        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, 1),
        )

    def forward(self, input_tensor):
        # input_tensor 形状: (batch_size, seq_len, C, H, W)
        batch_size = input_tensor.shape[0]
        # 检查输入序列长度是否匹配
        if input_tensor.shape[1] != self.seq_len:
             raise ValueError(f"Input sequence length ({input_tensor.shape[1]}) doesn't match model expected seq_len ({self.seq_len})")


        # 1. 获取嵌入
        # embedding 输出形状: (batch_size * seq_len, num_patches + 1, latent_size)
        x = self.embedding(input_tensor)

        # 2. 通过 Transformer Encoder 栈处理
        # 输入/输出形状: (batch_size * seq_len, num_patches + 1, latent_size)
        for enc_layer in self.encStack:
            x = enc_layer(x)

        # 3. 将 batch 和 time 维度重新分开
        # 形状: (batch_size, seq_len, num_patches + 1, latent_size)
        x = x.view(batch_size, self.seq_len, self.num_patches + 1, self.latent_size)

        # --- 4. 从序列中选择用于 Policy/Value Head 的特征 ---
        #    策略：只使用最后一个时间步 (索引 -1) 的特征，代表当前状态经过历史信息处理后的结果
        last_step_features = x[:, -1, :, :] # 形状: (batch_size, num_patches + 1, latent_size)

        # 从最后一个时间步的特征中分离 CLS token 和 patch tokens
        cls_token_last = last_step_features[:, 0, :]      # 形状: (batch_size, latent_size)
        patch_tokens_last = last_step_features[:, 1:, :]  # 形状: (batch_size, num_patches, latent_size)

        # --- 5. 计算 Value ---
        # 使用最后一个时间步的 CLS token
        value = self.value_head(cls_token_last) # 形状: (batch_size, 1)

        # --- 6. 计算 Policy ---
        # 使用最后一个时间步的 patch tokens
        policy_logits_per_patch = self.policy_head(patch_tokens_last) # 形状: (batch_size, num_patches, 1)
        policy_logits = policy_logits_per_patch.squeeze(-1) # 形状: (batch_size, num_patches)

        # 返回 policy logits 和 value
        return policy_logits, value



class CNN(torch.nn.Module):
    def __init__(self, game_size=15, patch_size=3, num_history=2, embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.):
        super().__init__()
        # 初始化参数：棋盘大小、补丁大小、历史步骤数、嵌入维度、深度、头数、MLP 比率
        self.num_history = num_history  # 历史步数
        self.game_size = game_size  # 棋盘大小（默认15x15）
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用 GPU 或 CPU

        # 假设输入通道数为 3（来自 get_encoded_state 方法）
        self.input_channels = 3
        self.seq_len = num_history + 1  # 序列长度为历史步数加 1（当前步）

        # 这里使用的是一个简单的卷积网络（ConvNet）加全连接层结构
        # 实际的 GoViT 应该处理序列数据，下面的代码需要替换为真实的 Vision Transformer 实现
        self.conv1 = torch.nn.Conv2d(self.input_channels * self.seq_len, embed_dim, kernel_size=3, padding=1)  # 卷积层
        self.fc_policy = torch.nn.Linear(embed_dim * game_size * game_size, game_size * game_size)  # 策略输出层
        self.fc_value = torch.nn.Linear(embed_dim * game_size * game_size, 1)  # 价值输出层
        self.train_flag = True  # 用于与原始代码的一致性

    def forward(self, x):
        # 前向传播：输入 x 的形状为 (batch, seq_len, channels, height, width)
        batch_size = x.shape[0]

        # 重新调整输入形状，将历史序列视为通道（这个操作需要根据 ViT 实际处理方式调整）
        # 对于真正的 Vision Transformer（ViT），序列维度的处理方式会有所不同
        x = x.view(batch_size, self.seq_len * self.input_channels, self.game_size, self.game_size)

        # 通过卷积层处理输入
        x = F.relu(self.conv1(x))  # 使用 ReLU 激活函数
        x = x.view(batch_size, -1)  # 将输出展平为一维

        # 通过全连接层得到策略和价值预测
        policy = self.fc_policy(x)  # 策略预测：每个位置的概率分布
        value = torch.tanh(self.fc_value(x))  # 价值预测：使用 Tanh 激活函数，输出值在 (-1, 1) 之间

        return policy, value  # 返回策略和价值预测
