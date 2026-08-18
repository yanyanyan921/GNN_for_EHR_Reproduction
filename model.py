import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)


# 创建 N 个相同的层（模块）的深拷贝副本，并将它们放入一个 ModuleList 中
def clones(module, N):
    """
    如果直接用 Python 列表 [copy.deepcopy(param) for _ in range(N)]，PyTorch 不会自动注册这些参数，导致：
    model.parameters() 会漏掉它们
    model.to(device) 不会自动转移它们
    用 nn.ParameterList 可以避免这个问题。
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 创建 N 个相同参数的独立副本，并将它们放入一个 ParameterList 中
def clone_params(param, N):
    """
    param：一个 PyTorch 的 nn.Parameter 对象（通常是可训练的张量，如权重矩阵、偏置等），是个类实例/张量
    示例：
    param = nn.Parameter(torch.randn(3, 3))，nn.Parameter(...)将其包装成可训练的参数
    torch.rand 是 PyTorch 中用于生成随机张量的函数，它从 均匀分布 中采样，生成的值在 [0, 1) 区间内
    tensor([[-1.5908,  0.9245, -0.0667],
            [-0.1830,  0.1338, -0.0925],
            [-0.8582, -0.7299,  0.7864]], requires_grad=True)
    """
    return nn.ParameterList([copy.deepcopy(param) for _ in range(N)])


# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        在 PyTorch 中，所有自定义层都必须继承自 nn.Module
        features：特征维度
        原始层归一化的公式是 output = γ * (x - μ) / sqrt(σ² + ε) + β
        ε (eps)   = 小常数，防止除零 (self.eps)
        """
        # 在 self 对象的继承链中，从 LayerNorm 的父类开始查找方法
        super(LayerNorm, self).__init__()  # 调用父类 nn.Module 的构造函数（必须的初始化步骤）
        self.a_2 = nn.Parameter(torch.ones(features))  # a_2（缩放参数），即γ，控制归一化后的 “范围”
        self.b_2 = nn.Parameter(torch.zeros(features))  # b_2（平移参数），即β，控制归一化后的 “偏移”
        self.eps = eps

    def forward(self, x):
        """
        假如 x 输入形状: torch.Size([2, 10, 512])，则输出形状: torch.Size([2, 10, 512])，features=512

        因为序列长度可变，批次大小不稳定，BatchNorm 不适用
        LayerNorm 对每个样本独立计算，更稳定

        缩放参数 a_2 和平移参数 b_2 的作用是：在保证训练稳定性的前提下，让网络自己决定归一化后的数据应该具有什么样的分布，从而不丢失表达能力
        """
        """
        x = tensor([
            [   # 第1个batch
                [ 0.0869, -1.5495,  0.7818,  0.6079],  # ← 长度为4，被归一化的单位
                [-0.8282,  0.3942,  0.8590,  0.7582]   # ← 长度为4，被归一化的单位
            ],
            [   # 第2个batch
                [-0.9661,  0.3623, -0.2381,  0.4326],  # ← 长度为4，被归一化的单位
                [-1.1702,  0.3027, -0.3016, -0.8585]   # ← 长度为4，被归一化的单位
            ]
        ])
        """
        mean = x.mean(-1, keepdim=True)  # 沿着最后一个维度（features 维度）计算均值，并保持维度不变（便于广播）。
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 自定义图神经网络层（GraphLayer）的实现
class GraphLayer(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, num_of_nodes,
                 num_of_heads, dropout, alpha, concat):
        super(GraphLayer, self).__init__()
        # concat：是否拼接多头输出（True）还是平均（False）
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.num_of_nodes = num_of_nodes
        self.num_of_heads = num_of_heads
        # 全连接层，将输入特征映射到隐藏维度，每个节点的特征向量被看作一个列向量，形状为(in_features, 1)
        self.W = clones(nn.Linear(in_features, hidden_features), num_of_heads)
        # 创建多头注意力参数，在 GAT 中，计算注意力分数时需要拼接源节点和目标节点的特征，
        # 即 [source_features || target_features]，形状是(2 * hidden_features, 1)
        self.a = clone_params(nn.Parameter(torch.rand(size=(1, 2 * hidden_features)), requires_grad=True), num_of_heads)
        self.ffn = nn.Sequential(  # 标准 Transformer 的 FFN：FFN(x) = ReLU(x * W1 + b1) * W2 + b2，这里是一个简化版
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
        if not concat:  # 根据 concat 参数动态创建的输出投影层，concat=False（平均模式）：多头输出先取平均
            self.V = nn.Linear(hidden_features, out_features)
        else:  # 多头输出拼接在一起，维度是 num_of_heads * hidden_features
            self.V = nn.Linear(num_of_heads * hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if concat:  # “二维视图”原则：永远把高维张量压成 (N, features)
            self.norm = LayerNorm(num_of_heads * hidden_features)  # 改了一下，由hidden_features改为num_of_heads * hidden_features
        else:
            self.norm = LayerNorm(hidden_features)

    # 让每个注意力头的线性变换层都有良好的初始值，避免梯度消失或爆炸
    def initialize(self):
        """
        Xavier   正态	    激活函数是 Sigmoid/Tanh 的网络，保持前向传播的方差稳定
        He       初始化	    激活函数是 ReLU 的网络，考虑 ReLU 的截断效应
        标准正态	 无特殊要求	可能导致梯度消失/爆炸
        """
        for i in range(len(self.W)):
            nn.init.xavier_normal_(self.W[i].weight.data)
        for i in range(len(self.a)):
            nn.init.xavier_normal_(self.a[i].data)
        if not self.concat:
            nn.init.xavier_normal_(self.V.weight.data)
            # nn.init.xavier_normal_(self.out_layer.weight.data)  这行代码有问题

    # 图注意力网络（GAT）中单个注意力头的核心计算逻辑
    def attention(self, linear, a, N, data, edge):
        """
        linear：当前头的线性变换层（self.W[i]），将输入特征映射到隐藏维度
        a：当前头的注意力参数（self.a[i]），用于计算节点对之间的注意力分数
        N：节点总数
        data：节点特征矩阵，形状 [N, in_features]
        edge：边列表，形状 [2, E]（第一行是源节点，第二行是目标节点）
        """
        """
        linear(data)：将节点特征从 [N, in_features] 映射到 [N, hidden_features]
        .unsqueeze(0)：增加一个维度，形状变为 [1, N, hidden_features]（方便后续广播和拼接）
        
        torch.isnan(data)：返回一个与 data 形状相同的布尔张量。每个位置：如果是 NaN，则为 True；否则为 False。
        .any()：检查布尔张量中是否有任何一个 True。只要有至少一个 NaN，.any() 就返回 True。
        assert：如果表达式为 False，触发 AssertionError，程序停止。如果为 True，程序继续执行。
        """
        data = linear(data).unsqueeze(0)  # data形状：[1,N,hidden_features]
        assert not torch.isnan(data).any()
        """
        h是提取边的两端节点特征后的结果，其中E 是选中边的数量：
        edge[0, :]：所有边的源节点索引
        edge[1, :]：所有边的目标节点索引
        data[:, edge[0, :], :]：提取源节点的特征，形状 [1,E, hidden_features]
        data[:, edge[1, :], :]：提取目标节点的特征，形状 [1, E, hidden_features]
        torch.cat(..., dim=0)：沿第0维拼接，形状 [2, E, hidden_features]
        data = data.squeeze(0)：去掉第0维，恢复为 [N, hidden_features]
        此时 h 的维度：
        h[0, :, :]：所有源节点的特征 [E, hidden_features]
        h[1, :, :]：所有目标节点的特征 [E, hidden_features]
        """
        h = torch.cat((data[:, edge[0, :], :], data[:, edge[1, :], :]), dim=0)  # h形状：[2, E, hidden_features]
        data = data.squeeze(0)  # data形状：[N,hidden_features]
        assert not torch.isnan(h).any()
        """
        拼接源节点和目标节点特征，：
        h[0, :, :]：源节点特征 [E, hidden_features]
        h[1, :, :]：目标节点特征 [E, hidden_features]
        torch.cat(..., dim=1)：沿特征维度拼接，得到 [E, 2 * hidden_features]
        .transpose(0, 1)：转置为 [2 * hidden_features, E]
        """
        # edge_h是同一条边两端节点的拼接结果，形状：[2 * hidden_features, E]，每列对应一条边，内容是 [源节点特征 || 目标节点特征]（拼接后的列向量）
        edge_h = torch.cat((h[0, :, :], h[1, :, :]), dim=1).transpose(0, 1)
        """
        计算注意力分数：
        a.mm(edge_h)：(1, 2*hidden_features) × (2*hidden_features, E) = (1, E)，得到每条边的原始注意力分数
        .squeeze()：去掉第0维，变为 (E,)，squeeze() 默认会去掉所有大小为 1 的维度
        / np.sqrt(self.hidden_features)：缩放（防止数值过大），理论上应该除以 sqrt(2 * hidden_features)。
        但由于 a 是可学习的参数，模型会自动调整，所以实际代码中普遍使用 sqrt(hidden_features)，这是 GAT 实现的标准做法。
        """
        # edge_e显示的是每条边的一个权重值。形状：(E,)
        edge_e = torch.exp(self.leakyrelu(a.mm(edge_h).squeeze()) / np.sqrt(self.hidden_features))  # 我把self.hidden_features * self.num_of_heads修改成self.hidden_features
        assert not torch.isnan(edge_e).any()
        """
        这行代码是在构建一个 PyTorch 的稀疏张量（Sparse COO Tensor），用来表示图结构中的注意力权重矩阵
        torch.sparse_coo_tensor 是 PyTorch 中用于创建稀疏张量（Sparse Tensor）的函数，采用 COO（Coordinate List，坐标列表） 格式
        torch.Size 本质上就是一个 tuple，可以当作序列（列表/元组）来使用
        
        示例：
        data = torch.tensor([
           [0.1, 0.2, 0.3],   # 节点 0
           [0.4, 0.5, 0.6],   # 节点 1
           [0.7, 0.8, 0.9],   # 节点 2
           [1.0, 1.1, 1.2]    # 节点 3
        ])
        edge_e:
        tensor([0.9925, 0.9896, 0.9867], grad_fn=<ExpBackward0>)
        
        edge_e表示得到的注意力权重矩阵：
        tensor(indices=tensor([[0, 1, 2],       edge是作为位置索引，indices表示非零元素的坐标，形状 [N, nnz]（N 为维度数）
                               [1, 2, 3]]),
               values=tensor([0.9925, 0.9896, 0.9867]),
               size=(4, 4), nnz=3, layout=torch.sparse_coo,
               grad_fn=<SparseCooTensorWithDimsAndTensorsBackward0>)

        转换成稠密矩阵，形状为torch.Size([4, 4])：
        tensor([[0.0000, 0.9925, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.9896, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.9867],
                [0.0000, 0.0000, 0.0000, 0.0000]], grad_fn=<ToDenseBackward0>)
        """
        # edge_e表示得到的注意力权重矩阵，形状为torch.Size([N, N])
        edge_e = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N]))
        """
        torch.ones(size=(N, 1)).to(device)：全1向量 [N, 1]
        torch.sparse.mm(edge_e, ones)：稀疏矩阵 × 向量，计算每个节点的入度总和（即所有邻居的注意力权重之和），形状 [N, 1]
        
        torch.mm：只接受稠密张量
	    torch.sparse.mm：第一个参数必须是稀疏张量，第二个参数可以是稠密张量
	    
	    因为 edge_e 是稀疏张量，它可能在 GPU 上。为了执行 torch.sparse.mm(edge_e, ones)，两个张量必须在同一个设备上
	    如果 edge_e 在 GPU，而 ones 在 CPU，PyTorch 会报错
        """
        # e_rowsum表示的是每个节点的所有邻居的注意力权重之和，形状为[N, 1]
        e_rowsum = torch.sparse.mm(edge_e, torch.ones(size=(N, 1)).to(device))
        row_check = (e_rowsum == 0)
        e_rowsum[row_check] = 1  # 用于后续归一化：每个节点的所有邻居的注意力权重之和作为分母
        # .nonzero() 返回所有 True 元素的坐标（索引），形状为 [M, 2]，其中 M 是 True 的个数，第 0 列是行索引，第 1 列是列索引
        zero_idx = row_check.nonzero()[:, 0]
        """
        zero_idx.repeat(2, 1)表示在第 0 维（行）上重复 2 次，在第 1 维（列）上重复 1 次。
        示例：
        zero_idx:tensor([[1, 3, 5]])
        zero_idx.repeat(2, 1):
        tensor([[1, 3, 5],
                [1, 3, 5]])
        这表示添加三条自环边：
        节点 1 → 节点 1
        节点 3 → 节点 3
        节点 5 → 节点 5
        torch.sparse_coo_tensor(...)：创建自环的稀疏矩阵，权重为1
        .add(...)：将自环添加到原注意力矩阵中，确保所有节点都能被更新（即使没有邻居，也能保留自身信息）
        """
        # edge_e表示添加自环后的注意力权重矩阵，形状为torch.Size([N, N])
        edge_e = edge_e.add(  # 我把torch.sparse.FloatTensor换成torch.sparse_coo_tensor，需要最后一维的维数相等
            torch.sparse_coo_tensor(zero_idx.repeat(2, 1), torch.ones(len(zero_idx)).to(device), torch.Size([N, N])))
        # h_prime：每个节点的聚合特征
        h_prime = torch.sparse.mm(edge_e, data)  # h_prime的形状：稀疏矩阵 × 特征矩阵 = [N, N] × [N, hidden_features] = [N, hidden_features]
        assert not torch.isnan(h_prime).any()
        """
        h_prime.div_(e_rowsum) 的作用是把加权和转换为加权平均，让每个节点的特征值不受邻居数量的影响，从而让模型更关注特征本身而不是节点度数
        
        # 标准 GAT
        attention = softmax(eij)      # 每行和为 1
        h_prime = attention @ data    # 已经是加权平均
 
        # 本论文的代码
        attention = exp(leakyrelu(...))     # 未归一化
        h_prime = attention @ data          # 加权和
        h_prime = h_prime / sum(attention)  # 转为加权平均
        """
        # h_prime，每个节点的特征变为邻居特征的加权平均，形状 [N, hidden_features]
        h_prime.div_(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return h_prime

   # 单个图层里多个注意力头的结果组合
    def forward(self, edge, data=None):
        N = self.num_of_nodes
        if self.concat:
            # 在特征维度（dim=1，按列拼接）上拼接所有头的输出，得到形状为 [N, num_of_heads * hidden_features] 的拼接张量
            # zip(self.W, self.a) 是 Python 内置函数 zip() 的用法，它的作用是将两个列表（或可迭代对象）按位置配对，生成一个元组的迭代器
            h_prime = torch.cat([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=1)
        else:
            """
            torch.stack(..., dim=0)：将所有头的输出堆叠成形状为[num_of_heads, N, hidden_features]的张量
            .mean(dim=0)：沿头的维度（dim = 0）取平均，得到形状为[N, hidden_features]的平均表示
            
            示例：
            stacked = torch.stack([head1, head2, head3], dim=0)，stacked.shape = [3, 2, 2]  (头的数量, 节点数, 特征维度)
            即stacked=tensor([[[0.5, 0.2],
                               [0.8, 0.3]],
            
                              [[0.1, 0.9],
                               [0.4, 0.6]],
             
                              [[0.7, 0.4],
                               [0.2, 0.5]]])
                     
            .mean(dim=0)是指沿着第0维（头的维度）取平均，效果为：
            stacked.mean(dim=0)=tensor([[0.4333, 0.5000],       # 节点0的平均特征 (0.5+0.1+0.7)/3, (0.2+0.9+0.4)/3
                                        [0.4667, 0.4667]])      # 节点1的平均特征 (0.8+0.4+0.2)/3, (0.3+0.6+0.5)/3

            dim 指向哪个轴，就把哪个轴压缩掉，该轴上的所有值合并成一个平均值。
            示例：
                t = torch.tensor([
                    # 批次0
                    [[1, 2, 3, 4],
                     [5, 6, 7, 8],                形状: (2, 3, 4)
                     [9, 10, 11, 12]],
    
                    # 批次1
                    [[13, 14, 15, 16],
                     [17, 18, 19, 20],
                     [21, 22, 23, 24]]
                ], dtype=torch.float32)
            mean(dim=0)：沿批次维度求平均（跨批次），对每个位置的元素，在批次维度上求平均。
            mean(dim=1)：沿行维度求平均（跨行），保留批次和列，对每一行求平均，效果为：
            t.mean(dim=1)=tensor([[ 5,  6,  7,  8],
                                  [17, 18, 19, 20]])                 形状: (2, 4)，去掉行维度
            mean(dim=2)：沿列维度求平均（跨列），保留批次和行，对每一列求平均，效果为：
            stacked.mean(dim=2)=tensor([[ 2.5,  6.5, 10.5],
                                        [14.5, 18.5, 22.5]])         形状: (2, 3)，去掉列维度
            """
            h_prime = torch.stack([self.attention(l, a, N, data, edge) for l, a in zip(self.W, self.a)], dim=0).mean(dim=0)
        h_prime = self.dropout(h_prime)  # 在训练过程（每次前向传播）中，以一定概率随机将部分神经元输出置为 0，从而防止模型过拟合，提升泛化能力
        if self.concat:
            # 先对 h_prime 进行层归一化（self.norm），然后应用ELU激活函数，输出形状为[N, num_of_heads * hidden_features]
            return F.elu(self.norm(h_prime))
        else:
            # 先对 h_prime 进行层归一化，然后应用 ReLU 激活，最后通过一个线性层 self.V 将特征映射到 out_features 维度，输出形状为[N, out_features]
            return self.V(F.relu(self.norm(h_prime)))


# 变分图神经网络（Variational GNN）模型，用于处理 EHR 数据，主要包含图注意力层、变分正则化和分类预测功能
class VariationalGNN(nn.Module):

    def __init__(self, in_features, out_features, num_of_nodes, n_heads, n_layers,
                 dropout, alpha, variational=True, none_graph_features=0, concat=False):  # 默认不拼接，我把concat=True改成concat=False
        """
        n_layers：图编码器的层数（堆叠的 GraphLayer 数量）
        alpha：LeakyReLU的负斜率
        variational：是否启用变分正则化（默认True）
        none_graph_features：非图特征个数（如年龄、性别等额外特征）
        """
        super(VariationalGNN, self).__init__()
        self.variational = variational
        # 为 padding_idx = 0预留一个额外索引。在 nn.Embedding中，padding_idx = 0表示索引0的词向量为零向量，用于填充或占位
        # 此处 none_graph_features 指的是非图特征的节点数
        # 我把 num_of_nodes+1-none_graph_features 改成 num_of_nodes+2-none_graph_features，我自己添加了目标节点，目标节点也需要嵌入向量，注意力计算的时候需要用节点的特征向量
        self.num_of_nodes=num_of_nodes+2-none_graph_features
        # padding_idx=0 的实际意义：节点编号从 0 开始，通常 0 被预留为 padding，不参与训练
        self.embed = nn.Embedding(self.num_of_nodes, in_features, padding_idx=0)  # 创建一个嵌入层
        self.in_att = clones(
            GraphLayer(in_features, in_features, in_features, self.num_of_nodes,
                       n_heads, dropout, alpha, concat=True), n_layers)
        self.out_features = out_features
        self.out_att = GraphLayer(in_features, in_features, out_features, self.num_of_nodes,
                                  n_heads, dropout, alpha, concat)  # 我把concat=False改成concat
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        """
        如果启用变分正则化，这个线性层用于从节点表示中生成均值（μ）和对数方差（logσ）。输出维度是out_features * 2，前一半是均值，后一半是方差。
        self.parameterize 将每个节点的表示映射为其对应的高斯分布参数，然后通过采样得到 z，作为解码器的输入。让节点表示不再是固定值，而是服从一个分布。
        """
        self.parameterize = nn.Linear(out_features, out_features * 2)
        # nn.Sequential 接受的是一个有序的模块列表，每个元素必须是一个 nn.Module 的子类实例
        self.out_layer = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, 1))  # 将最终的特征向量（长度为 out_features）映射到一个单一的数值（标量），作为模型的输出
        self.none_graph_features = none_graph_features
        if none_graph_features > 0:
            self.features_ffn = nn.Sequential(
                nn.Linear(none_graph_features, out_features//2),
                nn.ReLU(),
                nn.Dropout(dropout))
            """
            假设两个患者的图节点表示分别为：
            torch.tensor([
                          [0.1, 0.2, 0.3, 0.4],   # 患者1的图特征
                          [0.5, 0.6, 0.7, 0.8]    # 患者2的图特征
                         ], dtype=torch.float32)  
            假设两个患者的非图特征（年龄、性别、种族）经过 features_ffn 编码后为：
            torch.tensor([
                          [1.0, 2.0],   # 患者1的非图特征编码（out_features//2 = 2）
                          [3.0, 4.0]    # 患者2的非图特征编码
                         ], dtype=torch.float32) 
            执行拼接：
            tensor([
                    [0.1, 0.2, 0.3, 0.4, 1.0, 2.0],   # 患者1：4个图特征 + 2个非图特征
                    [0.5, 0.6, 0.7, 0.8, 3.0, 4.0]    # 患者2：4个图特征 + 2个非图特征
                   ])
            """
            # 更新 self.out_layer：将图节点表示（out_features）和非图特征（out_features//2）拼接后，再通过全连接层输出。
            self.out_layer = nn.Sequential(
                nn.Linear(out_features + out_features//2, out_features),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_features, 1))
        for i in range(n_layers):
            self.in_att[i].initialize()

    # 将每个患者存在的医疗概念（节点）转换为全连接图的边（edges），并返回输入图和输出图的边索引，其实就是创建edge
    def data_to_edges(self, data):
        # data：一个形状为[num_nodes]的张量，表示哪些医疗概念（节点）被激活 / 存在。示例如下：
        # 节点索引: 0    1      2     3     4      5
        # data：  1      0     1     1     0      1
        # 状态: [True, False, True, True, False, True]
        data = data.bool()
        length = data.size(0)  # 我把 length = data.size()[0]改成了 length = data.size(0)，取 data 张量在第一个维度上的大小，即 num_nodes
        nonzero = data.nonzero()  # nonzero：形状为 [M, 1]，记录所有值为 True（即被激活的节点）的索引，M为符合要求的节点个数，没有列索引
        """
        nonzero.size()=torch.Size([3, 1])，torch.Size([3, 1])是一个 torch.Size 对象，但它支持列表式操作，可以像列表一样索引和迭代
        
        如果没有任何节点激活，返回一个“虚拟自环”边
        0 是在 1 到 length 范围之外的节点索引，它是一个占位符，代表“没有激活节点时的虚拟输入节点
        length + 1 是输出图独有的目标节点索引
        
          项目	            输入图	               输出图
        节点范围	         1 ~ length	            1 ~ length + 1
        自环索引	         0（范围外占位）	        length + 1（真实目标节点）
          原因	         在输入节点范围之外	    在输出节点范围之内
        """
        if nonzero.size(0) == 0:  # 我把 nonzero.size()[0]改成 nonzero.size(0)，LongTensor 即 torch.int64，是索引操作的标准选择。
            return torch.LongTensor([[0], [0]]), torch.LongTensor([[length + 1], [length + 1]])
        if self.training:
            # torch.rand 生成的是 [0, 1) 区间上的均匀分布
            mask = torch.rand(nonzero.size(0))  # 我把 nonzero.size()[0]改成 nonzero.size(0)
            mask = mask > 0.1  # 这里对齐论文设定在训练阶段，随机丢弃 10% 节点，我把 mask > 0.05 改成 mask > 0.1
            nonzero = nonzero[mask]
            if nonzero.size(0) == 0:  # 我把 nonzero.size()[0]改成 nonzero.size(0)
                return torch.LongTensor([[0], [0]]), torch.LongTensor([[length + 1], [length + 1]])
        nonzero = nonzero.transpose(0, 1) + 1  # 将形状变为 [1, M]，并将索引 +1（把 0 位留作 padding 节点），映射到图中的节点 1 ~ length
        lengths = nonzero.size(1)  # 我把 nonzero.size()[1]改成 nonzero.size(1)，这是实际激活节点的数量 M
        """
        构建输入图（全连接）：
        nonzero.repeat(1, lengths)：把源节点重复 M 次，形成所有可能的边的起点。
        .contiguous()确保张量在内存中是连续存储的，这是调用.view() 前的必要操作
        nonzero.repeat(lengths, 1).transpose(0, 1).contiguous().view((1, lengths ** 2))：把目标节点也重复并转置，形成所有可能的终点。
        拼接得到 input_edges：形状 [2, M * M]，代表从激活节点指向所有激活节点的全连接边。
        
        示例如下：
        nonzero = torch.tensor([[1, 3]])  
        nonzero.repeat(1, lengths)=tensor([[1, 3, 1, 3]])    # 每个节点都可成为其他节点的源节点，所以复制M份
        nonzero.repeat(lengths, 1)：
        tensor([[1, 3],          .transpose(0, 1)后——>          tensor([[1, 1],
                [1, 3]])                                                [3, 3]])
        .contiguous().view((1, lengths ** 2))=tensor([[1, 1, 3, 3]])    # 每个节点都可成为其他节点的终节点，固定终结点，集中给每一轮的源节点分配同一终节点    
        input_edges：
        tensor([[1, 3, 1, 3],
                [1, 1, 3, 3]])
        """
        input_edges = torch.cat((nonzero.repeat(1, lengths),
                                 nonzero.repeat(lengths, 1).transpose(0, 1)
                                 .contiguous().view((1, lengths ** 2))), dim=0)
        # 构建输出图
        nonzero = torch.cat((nonzero, torch.LongTensor([[length + 1]]).to(device)), dim=1)  # 实际节点索引比 data 中的节点索引大1
        lengths = nonzero.size(1)  # 我把 nonzero.size()[1]改成 nonzero.size(1)，加了一个目标节点
        output_edges = torch.cat((nonzero.repeat(1, lengths),
                                  nonzero.repeat(lengths, 1).transpose(0, 1)
                                  .contiguous().view((1, lengths ** 2))), dim=0)
        return input_edges.to(device), output_edges.to(device)

    # 这是变分自编码器（VAE）中重参数化技巧（ReparameterizationTrick） 的标准实现
    def reparameterise(self, mu, logvar):
        """
        mu：均值（Mean）
        logvar：对数方差（Log Variance）
        """
        if self.training:
            # mul(0.5)是逐元素乘以0.5，相当于logvar * 0.5，数学意义：0.5 * log(σ²) = log(σ)
            # .exp_()是原地指数运算，计算e ^ (输入)，将对数转换为原始值，_后缀表示原地操作，直接修改张量，不创建新副本
            std = logvar.mul(0.5).exp_()  # 计算标准差
            """
            std 是一个张量，std.data 返回它的底层数据张量，此处的作用只是获取 std 的数据类型和设备信息（CPU/GPU）
            std.data.new(size) 创建一个新的空张量，它具有与 std 相同的：数据类型（dtype）/设备（device，即 CPU 或 GPU）
            .normal_()是一个原地方法，用标准正态分布（均值为 0，方差为 1）的随机数填充张量
            """
            eps = std.data.new(std.size()).normal_()  # 生成标准正态噪声
            # eps.mul(std)：噪声 × 标准差 → ε * σ
            # .add_(mu)：加上均值 → μ + ε * σ
            # 这就是重参数化的核心公式：z = μ + ε * σ，其中ε~ N(0, 1)
            return eps.mul(std).add_(mu)
        # 直接采样 z = N(mu, exp(logvar)不行，采样操作不可导，无法反向传播
        else:  # 评估时返回均值（mu）是为了获得确定性的、最可能的重建结果
            return mu

    # 这是整个模型的核心，它接收一个患者的特征向量，通过编码器 - 解码器图神经网络提取信息，最终返回预测结果和KL散度损失
    def encoder_decoder(self, data):
        N = self.num_of_nodes
        input_edges, output_edges = self.data_to_edges(data)
        """
        torch.arange(N).long()：生成[0, 1, 2, ..., N - 1]的索引
        h_prime 形状为 [N, in_features]，就是节点特征矩阵，这时候嵌入还带上了索引0
        最开始所有节点的嵌入都是一视同仁地通过 self.embed 生成的，与节点是否激活无关，data 是布尔掩码
        nn.Embedding 是全局共享的参数表。每个医学概念（如"糖尿病"）在表中只有一行，所有患者共用。训练过程中，这一行会从随机初始值逐渐学习到有意义的语义表示。
        """
        h_prime = self.embed(torch.arange(N).long().to(device))
        # 依次通过 n_layers 个图注意力层，每一层都使用 input_edges 作为图结构，节点表示逐层更新，捕捉高阶邻居信息
        for attn in self.in_att:
            h_prime = attn(input_edges, h_prime)
        if self.variational:  # 如果启用变分正则化，将节点表示映射到均值和方差空间
            """
            view 执行过程:
            ①先计算出-1=N
            原始张量 h_prime（形状 [2, 6]）：
            行0: [-0.4329, -0.3409,  0.3759,  0.4377,  0.0909,  0.2249]
            行1: [-0.3492,  0.1240,  1.3735, -0.1260, -0.9876,  0.6611]
            ②展平成一维数组
            [-0.4329, -0.3409,  0.3759,  0.4377,  0.0909,  0.2249,
             -0.3492,  0.1240,  1.3735, -0.1260, -0.9876,  0.6611]
            ③按目标形状 [2, 2, 3] 逐行填充，目标形状是 4 个样本，每个样本是 [2, 3] 的矩阵
            样本0（对应原行0）：
            取前 6 个数：[-0.4329, -0.3409, 0.3759] 和 [0.4377, 0.0909, 0.2249]
            结果：[[-0.4329, -0.3409, 0.3759], [0.4377, 0.0909, 0.2249]]
            样本1（对应原行1）：
            取接下来的 6 个数：[-0.3492, 0.1240, 1.3735] 和 [-0.1260, -0.9876, 0.6611]
            """
            h_prime = self.parameterize(h_prime).view(-1, 2, self.out_features)  # h_prime 形状为 [N, 2, out_features]
            h_prime = self.dropout(h_prime)
            mu = h_prime[:, 0, :]  # 均值：形状 [N, out_features]
            logvar = h_prime[:, 1, :]  # 对数方差：形状 [N, out_features]
            h_prime = self.reparameterise(mu, logvar)
            """
            以下是我自己修改的部分，data原始是没有考虑 padding=0的，因此里面的节点索引必须+1才能对齐嵌入层的索引
            索引 0 的嵌入向量始终为零向量，并且在训练过程中梯度不更新，它的存在只是为了技术占位（让图结构完整），不携带任何语义信息
            所以我把嵌入的特征矩阵第一行去掉，data只显示在节点位置上该节点是否激活，节点索引从0开始，所以和去掉第一行的mu刚好匹配
            最后一行也需要去掉，不然和data的维度不匹配，目标节点不参与KL散度计算，所以不需要正则化
            正则化的作用是防止特征节点的表示变得太相似，从而保留每个医疗概念的独特性。在没有预定义知识图谱的情况下，自注意力机制容易导致节点表示"坍塌"成紧密聚类
            """
            # 原始代码，获取观察节点的 mu 和 logvar（计算KL散度用）
            # mu = mu[data, :]
            # logvar = logvar[data, :]
            # 修改后的代码
            data=data.bool()
            mu = mu[1:-1,:][data, :]
            logvar = logvar[1:-1,:][data, :]
            # 以上是修改后的代码
        h_prime = self.out_att(output_edges, h_prime)
        if self.variational:
            """
            以3个节点为例:
            t=tensor([[0.0000, 0.0000],
                      [1.1487, 0.1487],
                      [8.3891, 8.3891]])
                      
            torch.sum(t,dim=1):
            tensor([ 0.0000,  1.2974, 16.7781])
                        ↑        ↑       ↑
                      节点1     节点2    节点3
                      
            h_prime[-1]：最后一个节点（目标节点）的表示，形状 [out_features]，经过 out_layer 后变成标量预测值
            学到的分布 q(z|x) = N(μ, exp(logvar))与标准正态分布 p(z) = N(0, 1)之间的 KL 散度
            平均KL = 总KL和 / mu.size(0)  # 除以节点个数
            """
            # Python 中，多个返回值用逗号分隔，会自动打包成一个元组
            # 我把 mu.size()[0]改成 mu.size(0)
            return h_prime[-1], 0.5*torch.sum(logvar.exp()-logvar-1+mu.pow(2))/mu.size(0)
        else:
            return h_prime[-1], torch.tensor(0.0).to(device)

    def forward(self, data):
        """
        data 是一个 batch 的数据，这里的一行指的是一个患者，形状：[batch_size, num_features]，每个样本是一个 0/1 向量
        每个样本中的 0/1 代表着该医疗概念节点患者有没有包括，即有没有被激活。data.size(0) 获取第0维的大小 = batch 中有多少个样本

        data = [
                #  前3个是非图特征(连续值)    后7个是图特征(0/1)
                #  年龄  性别  种族        糖尿病 高血压   肺炎  心衰  哮喘  肾衰  流感
                 [ 65,   1,    0,       1,     0,     1,    0,   0,   0,   0 ],   # 患者1
                 [ 45,   0,    1,       0,     1,     0,    0,   1,   0,   0 ]    # 患者2
               ]
        患者1: features_ffn([65, 1, 0]) → [128] 向量 (128维)
        患者2: features_ffn([45, 0, 1]) → [128] 向量
        拼接的是：目标节点的向量（图网络的输出，代表"患者画像"）+非图特征的向量（经过 FFN 处理后的结果）
        """
        batch_size = data.size(0)  # 我把 data.size()[0]改成 data.size(0)
        "在 eICU 数据中，第一个特征（即患者之前是否曾入院）不包含在图中"
        if self.none_graph_features == 0:
            outputs = [self.encoder_decoder(data[i, :]) for i in range(batch_size)]
            """
            torch.stack 会增加维度，默认是在 第0维（dim=0） 上堆叠，torch.sum 默认是对所有元素求和
            torch.cat 默认在 dim=0 维度上拼接，一维张量
            torch.stack 的作用是：把列表变成张量，让 torch.sum 能处理，torch.sum() 不标明维度默认把所有元素求和
            
            示例:
            [out[0] for out in outputs]:
            [tensor([ 0.2000, -0.5000,  0.3000,  0.8000]), tensor([ 0.1000,  0.3000, -0.4000,  0.6000])]
             │        torch.stack           │
             ↓                              ↓
            tensor([[ 0.2000, -0.5000,  0.3000,  0.8000],      # shape=torch.Size([2, 4])   
                    [ 0.1000,  0.3000, -0.4000,  0.6000]])  
                    
            [out[1] for out in outputs]:
            [tensor(0.5000), tensor(0.3000), tensor(0.7000)]
             │        torch.stack           │
             ↓                              ↓
            tensor([0.5000, 0.3000, 0.7000])      # shape=torch.Size([3])
             │        torch.sum           │
             ↓                            ↓
                         1.5
            """
            return self.out_layer(F.relu(torch.stack([out[0] for out in outputs]))), \
                   torch.sum(torch.stack([out[1] for out in outputs]))
        else:
            # ([none_graph_features], (target_node , KL散度)) 存为一个元组
            outputs = [(data[i, :self.none_graph_features],
                        self.encoder_decoder(data[i, self.none_graph_features:])) for i in range(batch_size)]
            """
            out = [65, 1, 0]
             │   torch.FloatTensor   │     
             ↓                       ↓
            tensor([[65.,  1.,  0.]])     # shape=torch.Size([1, 3])
            torch.FloatTensor([out[0]])：这里需要的是 [1, 3] 形状（batch维度 + 特征维度），而不是 [3]
            
            torch.cat()默认在第0维（dim=0）上拼接，并不会增加新维度
            ①拼接 1D 向量，在唯一维度上拼接，示例如下：
            a=torch.tensor([1, 2, 3])   # 形状 [3]
            b=torch.tensor([4, 5, 6])   # 形状 [3]
            torch.cat([a, b])=tensor([1, 2, 3, 4, 5, 6])
            ②拼接 1D 向量（dim=0，默认），示例如下：
            a = torch.tensor([[1, 2], [3, 4]])  # 形状 [2, 2]
            b = torch.tensor([[5, 6], [7, 8]])  # 形状 [2, 2]
            torch.cat([a, b])：
                  tensor([[1, 2],
                          [3, 4],      # 形状=torch.Size([4, 2])
                          [5, 6],
                          [7, 8]])
                          
            torch.sum(...,dim=-1)的用法：
            x2 = torch.tensor([[0.5, 0.3, 0.7],
                               [0.2, 0.6, 0.1]])
             │   torch.sum(x2, dim=-1)   │     
             ↓                           ↓
               tensor([1.5000, 0.9000])  ← 向量！对列求和
            
            """
            # 我把 torch.FloatTensor([out[0]])改成了 torch.FloatTensor(out[0])
            # 我把 torch.sum(torch.stack([out[1][1] for out in outputs]), dim=-1)中的 dim=-1 删掉了
            # 上述修改是为了使torch.FloatTensor(out[0])的形状变为[none_graph_features]（即 1维，长度等于非图特征的数量）,可以横向拼接
            return self.out_layer(F.relu(
                torch.stack([torch.cat((self.features_ffn(torch.FloatTensor(out[0]).to(device)), out[1][0]))
                             for out in outputs]))), \
                             torch.sum(torch.stack([out[1][1] for out in outputs]))
