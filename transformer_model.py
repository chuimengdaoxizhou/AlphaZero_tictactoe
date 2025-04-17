import torch
import torch.nn as nn
import math

# 定义层归一化（Layer Normalization）类
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps  # 防止除以零的极小值
        self.alpha = nn.Parameter(torch.ones(features))  # alpha 是可学习的参数，形状为 (features,)
        self.bias = nn.Parameter(torch.zeros(features))   # bias 是可学习的参数，形状为 (features,)

    def forward(self, x):
        # 输入 x 的形状是 (batch, seq_len, hidden_size)
        mean = x.mean(dim = -1, keepdim = True)  # 计算均值，结果形状为 (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True)    # 计算标准差，结果形状为 (batch, seq_len, 1)
        # eps 是为了防止 std 太小导致除零错误
        return self.alpha * (x - mean) / (std + self.eps) + self.bias  # 归一化后加上偏置


# 定义前馈网络模块（FeedForward Block）
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一层线性变换，将输入映射到更高的维度
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二层线性变换，将输出映射回原始维度
        self.dropout = nn.Dropout(dropout)  # Dropout 防止过拟合

    def forward(self, x):
        x = self.linear1(x)  # 第一层线性变换
        x = self.relu(x)  # 应用 ReLU 激活函数
        x = self.linear2(x)  # 第二层线性变换
        return self.dropout(x)  # 返回通过 Dropout 的结果


# 定义输入嵌入（Input Embedding）模块
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # 嵌入维度
        self.vocab_size = vocab_size  # 词汇表大小
        self.embedding = nn.Embedding(vocab_size, d_model)  # 定义嵌入层，词汇大小为 vocab_size，嵌入维度为 d_model

    def forward(self, x):
        # 输入 x 的形状是 (batch, seq_len)，表示词索引
        # 将嵌入结果乘以 sqrt(d_model) 来缩放嵌入值
        return self.embedding(x) * math.sqrt(self.d_model)  # 输出形状 (batch, seq_len, d_model)
    

# 定义位置编码（Positional Encoding）模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # 嵌入维度
        self.seq_len = seq_len  # 序列长度
        self.dropout = nn.Dropout(dropout)  # Dropout层

        # 创建形状为 (seq_len, d_model) 的位置编码矩阵
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # 生成形状为 (seq_len, 1) 的位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算归一化系数
        pe[:, 0::2] = torch.sin(position * div_term)  # 对偶数位置应用 sin 函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 对奇数位置应用 cos 函数
        pe = pe.unsqueeze(0)  # 添加 batch 维度，形状为 (1, seq_len, d_model)
        self.register_buffer('pe', pe)  # 将位置编码作为缓冲区保存

    def forward(self, x):
        # 将输入 x 和位置编码相加
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # 输出形状 (batch, seq_len, d_model)
        return self.dropout(x)


# 定义残差连接模块
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.norm = LayerNormalization(features)  # 归一化层

    def forward(self, x, sublayer):
        # 先对 x 进行归一化，再通过子层（sublayer），最后加上原始的 x （残差连接）
        return x + self.dropout(sublayer(self.norm(x)))  # 输出形状与输入相同


# 定义多头自注意力（Multi-Head Attention）模块
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # 嵌入维度
        self.h = h  # 注意力头的数量
        assert d_model % h == 0, "d_model is not divisible by h"  # 确保 d_model 可以被 h 整除

        self.d_k = d_model // h  # 每个头的维度
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Q 权重矩阵
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # K 权重矩阵
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # V 权重矩阵
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # 输出权重矩阵
        self.dropout = nn.Dropout(dropout)  # Dropout层

    # 计算注意力
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # 每个头的维度
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力分数，形状 (batch, h, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)  # 应用 mask
        attention_scores = attention_scores.softmax(dim=-1)  # 归一化
        if dropout is not None:
            attention_scores = dropout(attention_scores)  # 应用 Dropout
        return (attention_scores @ value), attention_scores  # 返回加权值和注意力分数

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # 分别将 Q、K、V 转换为 (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)  # 计算注意力
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # 合并头部，形状 (batch, seq_len, d_model)
        return self.w_o(x)  # 输出形状 (batch, seq_len, d_model)


# 编码器块（Encoder Block）
class EncoderBlock(nn.Module):
    def __init__(self, 
                 features: int,  # 模型维度
                 self_attention_block: MultiHeadAttentionBlock,  # 自注意力层
                 feed_forward_block: FeedForwardBlock,  # 前馈网络层
                 dropout: float  # Dropout 概率
                 ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block  # 自注意力层
        self.feed_forward_block = feed_forward_block  # 前馈网络层
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout), # 两个残差连接
                                                   ResidualConnection(features, dropout)  # 两个残差连接    
                                                   ])

    def forward(self, x, mask):
        # 自注意力块和前馈网络块都使用残差连接
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))  # 自注意力
        return self.residual_connections[1](x, self.feed_forward_block)  # 前馈网络
    

# 定义编码器（Encoder）模块
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # 编码器中各层的模块列表
        self.norm = LayerNormalization(features)  # 层归一化

    def forward(self, x, mask):
        for layer in self.layers:  # 遍历每一层
            x = layer(x, mask)  # 对输入 x 应用每一层的运算
        return self.norm(x)  # 返回经过层归一化后的结果


# 定义解码器块（Decoder Block）模块
class DecoderBlock(nn.Module):
    def __init__(self, features: int, # 模型维度
                 self_attention_block: MultiHeadAttentionBlock,  # 自注意力层
                 cross_attention_block: MultiHeadAttentionBlock, # 跨注意力层
                 feed_forward_block: FeedForwardBlock, # 前馈网络层
                 dropout: float # Dropout 概率
                 ) -> None:
        
        super().__init__()
        self.self_attention_block = self_attention_block  # 自注意力层
        self.cross_attention_block = cross_attention_block  # 跨注意力层
        self.feed_forward_block = feed_forward_block  # 前馈网络层
        # 3个残差连接：自注意力、跨注意力和前馈网络
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 对输入 x 应用自注意力层
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))  
        # 对输入 x 应用跨注意力层，使用编码器的输出作为键和值
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))  
        # 对输入 x 应用前馈网络
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x  # 返回最终的解码器输出

# 定义解码器（Decoder）模块
class Decoder(nn.Module):
    def __init__(self, features: int,  # 模型维度
                 layers: nn.ModuleList  # 解码器中各层的模块列表
                 ) -> None:
        super().__init__()
        self.layers = layers  # 解码器中各层的模块列表
        self.norm = LayerNormalization(features)  # 层归一化

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:  # 遍历每一层
            x = layer(x, encoder_output, src_mask, tgt_mask)  # 应用每一层
        return self.norm(x)  # 返回经过层归一化后的结果


# 定义投影层（Projection Layer）
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # 使用线性变换将 d_model 映射到 vocab_size

    def forward(self, x) -> None:
        # 输入 x 的形状为 (batch, seq_len, d_model)
        # 输出的形状为 (batch, seq_len, vocab_size)，表示每个位置的词汇分布
        return self.proj(x)  # 将 d_model 映射到词汇表大小


# 定义 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,    # 编码器
                 decoder: Decoder,          # 解码器
                 src_embed: InputEmbeddings,    # 输入嵌入层
                 tgt_embed: InputEmbeddings,    # 输出嵌入层
                 src_pos: PositionalEncoding,   # 输入位置编码 
                 tgt_pos: PositionalEncoding,   # 输出位置编码 
                 projection_layer: ProjectionLayer   # 投影层
                 ) -> None:
        
        super().__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = src_embed  # 输入嵌入层
        self.tgt_embed = tgt_embed  # 输出嵌入层
        self.src_pos = src_pos  # 输入位置编码层
        self.tgt_pos = tgt_pos  # 输出位置编码层
        self.projection_layer = projection_layer  # 投影层，用于将解码器输出映射为词汇表大小

    def encode(self, src, src_mask):
        # 输入 src 的形状为 (batch, src_seq_len)，表示源语言的词索引
        # src_mask 的形状为 (batch, src_seq_len)，表示源语言的mask
        src = self.src_embed(src)  # 获取源语言的嵌入表示，输出形状 (batch, src_seq_len, d_model)
        src = self.src_pos(src)  # 加上源语言的位置信息，输出形状 (batch, src_seq_len, d_model)
        return self.encoder(src, src_mask)  # 经过编码器，输出形状 (batch, src_seq_len, d_model)

    def decode(self, 
                 encoder_output: torch.Tensor, # 编码器输出
                 src_mask: torch.Tensor,   # 源语言mask
                 tgt: torch.Tensor,   # 目标语言词索引
                 tgt_mask: torch.Tensor   # 目标语言mask
                 ):
        # 输入 tgt 的形状为 (batch, tgt_seq_len)，表示目标语言的词索引
        # tgt_mask 的形状为 (batch, tgt_seq_len)，表示目标语言的mask
        tgt = self.tgt_embed(tgt)  # 获取目标语言的嵌入表示，输出形状 (batch, tgt_seq_len, d_model)
        tgt = self.tgt_pos(tgt)  # 加上目标语言的位置信息，输出形状 (batch, tgt_seq_len, d_model)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # 经过解码器，输出形状 (batch, tgt_seq_len, d_model)

    def project(self, x):
        # 输入 x 的形状为 (batch, seq_len, d_model)
        # 输出形状为 (batch, seq_len, vocab_size)，表示每个位置的词汇分布
        return self.projection_layer(x)  # 投影到词汇表大小


# 构建 Transformer 模型的函数
def build_transformer(
        src_vocab_size: int, # 源语言词汇表大小
        tgt_vocab_size: int, # 目标语言词汇表大小
        src_seq_len: int,    # 源语言序列长度
        tgt_seq_len: int,    # 目标语言序列长度
        d_model: int=512,    # 模型维度
        N: int=6,            # 编码器和解码器堆叠的层数
        h: int=8,            # 注意力头的数量
        dropout: float=0.1,  # Dropout 概率
        d_ff: int=2048       # 前馈网络的内部层维度
        ) -> Transformer: 
    
    # 创建输入和输出嵌入层
    src_embed = InputEmbeddings(d_model, src_vocab_size)  # 输入语言的嵌入层
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)  # 输出语言的嵌入层

    # 创建位置编码层
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)  # 输入语言的位置信息编码
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)  # 输出语言的位置信息编码
    
    # 创建编码器块（Encoder Blocks）
    encoder_blocks = []
    for _ in range(N):  # 生成 N 个编码器块
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # 自注意力层
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # 前馈网络层
        # 创建编码器块
        encoder_block = EncoderBlock(d_model, # 模型维度
                                     encoder_self_attention_block, # 自注意力层
                                     feed_forward_block, dropout   # 前馈网络层
                                     )  
        # 添加到编码器块列表
        encoder_blocks.append(encoder_block)

    # 创建解码器块（Decoder Blocks）
    decoder_blocks = []
    for _ in range(N):  # 生成 N 个解码器块
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # 自注意力层
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # 跨注意力层
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # 前馈网络层
        # 创建解码器块
        decoder_block = DecoderBlock(d_model, 
                                     decoder_cross_attention_block, 
                                     feed_forward_block, dropout
                                     )  
        # 添加到解码器块列表
        decoder_blocks.append(decoder_block)
    
    # 创建编码器和解码器
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))  # 编码器
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))  # 解码器
    
    # 创建投影层
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)  # 投影层，将解码器输出映射到词汇表大小
    
    # 创建 Transformer 模型
    transformer = Transformer(encoder,   # 编码器
                               decoder,  # 解码器
                               src_embed,   # 输入嵌入层
                               tgt_embed,   # 输出嵌入层
                               src_pos,  # 输入位置编码
                               tgt_pos,  # 输出位置编码
                               projection_layer  # 投影层
                        )
    
    # 初始化模型参数
    for p in transformer.parameters():
        if p.dim() > 1:  # 只初始化权重参数
            nn.init.xavier_uniform_(p)  # 使用 Xavier 均匀分布初始化
    
    return transformer

# 1. `EncoderOnlyModel` 类定义了模型的前向传播过程
class EncoderOnlyModel(nn.Module):
    def __init__(self, encoder, src_embed, src_pos):
        super().__init__()
        self.encoder = encoder  # 编码器部分
        self.src_embed = src_embed  # 输入嵌入层
        self.src_pos = src_pos  # 位置编码层

    def forward(self, src, src_mask):
        # 先通过输入嵌入层和位置编码层处理输入
        src = self.src_embed(src)
        src = self.src_pos(src)
        # 然后通过编码器生成输出
        return self.encoder(src, src_mask)
    
# 2. `build_encoder_only_model` 是一个工厂函数，返回 `EncoderOnlyModel` 实例
def build_encoder_only_model(
        src_vocab_size: int, # 源语言词汇表大小
        src_seq_len: int,    # 源语言序列长度 
        d_model: int=512,    # 模型维度
        N: int=6,         # 编码器堆叠的层数
        h: int=8,         # 注意力头的数量
        dropout: float=0.1,  # Dropout 概率
        d_ff: int=2048       # 前馈网络的内部层维度
    ) -> EncoderOnlyModel:
    
    # 创建编码器所需的组件
    src_embed = InputEmbeddings(d_model, src_vocab_size)  # 输入嵌入层
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)  # 位置编码层
    encoder_layers = nn.ModuleList([
        EncoderBlock(
            features=d_model,
            self_attention_block=MultiHeadAttentionBlock(d_model, h, dropout),
            feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout),
            dropout=dropout
        ) for _ in range(N)
    ])
    
    # 创建并返回 EncoderOnlyModel 实例
    encoder = Encoder(features=d_model, layers=encoder_layers)
    return EncoderOnlyModel(encoder, src_embed, src_pos)

