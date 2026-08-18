import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import Dataset

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)


# 训练函数，负责 单个 batch 的模型训练
def train(data, model, optim, criterion, lbd, max_clip_norm=5):
    """
    lbd	KL 散度的权重（论文中的 λ）
    max_clip_norm=5	梯度裁剪的最大范数
    """
    # model 是 VariationalGNN 的实例，而 VariationalGNN 继承自 torch.nn.Module
    # model.train() 是 torch.nn.Module 类的方法，子类完全继承了父类的所有方法
    model.train()
    """
    data = [
            [特征0, 特征1, ..., 特征99, 标签],   # 样本0
            [特征0, 特征1, ..., 特征99, 标签],   # 样本1
            ...
           ]
    data是一个批次的数据，类型为Tensor
    """
    input = data[:, :-1].to(device)
    label = data[:, -1].float().to(device)
    #model.train()  # 这个删掉
    optim.zero_grad()  # 清空上一轮 batch 的梯度，PyTorch 默认梯度会累积，不清零会导致梯度叠加
    # logits：形状 [batch_size, 1]，是输出层的原始分数（未经过 sigmoid），kld：所有样本的 KL 散度总和（标量）
    logits, kld = model(input)
    logits = logits.squeeze(-1)  # .squeeze(-1) 将形状 [batch_size, 1]——>[batch_size]，和 label 形状一致
    kld = kld.sum()  # kld 可能已经是标量（所有样本的 KL 总和），kld.sum() 确保它是标量
    bce = criterion(logits, label)  # criterion = BCEWithLogitsLoss
    loss = bce + lbd * kld  # 总损失 = BCE损失 + λ × KL散度，传统 KL 散度 ≥ 0
    """
    torch.nn.utils.clip_grad_norm_：梯度裁剪函数（下划线表示原地操作，会直接修改梯度）
    model.parameters()：模型的所有可训练参数
    max_clip_norm：梯度范数的最大允许值
    
    示例：
    参数1的梯度  grad1 = torch.tensor([3.0, 4.0])
    参数2的梯度  grad2 = torch.tensor([6.0, 8.0])
    计算总范数：
    total_norm = sqrt(3² + 4² + 6² + 8²)
               = sqrt(9 + 16 + 36 + 64)
               = sqrt(125)
               = 11.18
    因为 max_clip_norm = 5，所以 total_norm (11.18) > 5 → 需要裁剪！
    scale = 5 / 11.18 = 0.447   
    裁剪后的梯度：
          grad1 = grad1 * 0.447 = [1.34, 1.79]
          grad2 = grad2 * 0.447 = [2.68, 3.58]     
          
    model.parameters() 会自动收集模型中所有需要训练的参数，所有参数都是通过 nn.Parameter() 或 nn.Module 的子层定义的，PyTorch 会自动追踪它们      
    """
    # 原代码
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
    # loss.backward()
    # 我把反向传播计算梯度和裁剪梯度的代码顺序换了一下，先反向传播计算梯度再裁剪梯度
    loss.backward()  # 先反向传播计算梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)  # 再裁剪梯度
    # 以上是修改的代码
    optim.step()  # 根据梯度更新模型参数
    # .item() 把 Tensor 转为 Python 标量
    return loss.item(), kld.item(), bce.item()


# 评估函数，用于在验证集或测试集上评估模型性能，计算 AUPRC
def evaluate(model, data_iter, length):
    # length：数据集总样本数（用于预分配数组）
    model.eval()  # 切换到评估模式，关闭 Dropout（不再随机丢弃）
    y_pred = np.zeros(length)
    y_true = np.zeros(length)
    y_prob = np.zeros(length)  # # 预测概率（0~1）
    pointer = 0  # 当前填充位置
    for data in data_iter:
        input = data[:, :-1].to(device)
        label = data[:, -1]
        batch_size = len(label)  # 一维张量可以用 len()
        probability, _ = model(input)  # probability 是 logits（未归一化的分数），形状 [batch_size, 1]
        # .detach()：让 probability 不再参与反向传播，到 probability 为止，切断了计算图的连接，让后面的操作（如 sigmoid、比较大小）不会影响梯度计算
        probability = torch.sigmoid(probability.squeeze(-1).detach())
        predicted = probability > 0.5  # 布尔型
        y_true[pointer: pointer + batch_size] = label.numpy()  # label 在 CPU 上，直接 .numpy()
        y_pred[pointer: pointer + batch_size] = predicted.cpu().numpy()  # predicted可能在 GPU，需要先 .cpu() 再 .numpy()
        y_prob[pointer: pointer + batch_size] = probability.cpu().numpy()
        pointer += batch_size
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)  # 不同阈值下的精确率、召回率、阈值列表
    # 计算 PR 曲线下的面积，等价于 average_precision_score；AUPRC 适用于不平衡数据集（正例很少），越高越好
    return auc(recall, precision), (y_pred, y_prob, y_true)


# 这个类继承自 PyTorch 的 Dataset 基类，将特征数据和标签封装在一起，方便后续使用 DataLoader 进行批量加载和迭代
class EHRData(Dataset):
    def __init__(self, data, cla):
        """
        data：特征数据（通常是 NumPy 数组、CSR 稀疏矩阵或 PyTorch Tensor）。
        cla：类别标签（通常是 0/1 数组，对应二分类任务）。
        """
        self.data = data
        self.cla = cla

    def __len__(self):
        return len(self.cla)

    def __getitem__(self, idx):
        # 返回 (稀疏矩阵的一行, 标签)
        return self.data[idx], self.cla[idx]


# collate_fn 把 Dataset 吐出来的零散样本（特征 + 标签），拼装成 DataLoader 可以整齐输出的 Tensor 批次，如[特征..., 标签]
def collate_fn(data):
    """
    data = [
            (csr_matrix([0, 0, 1, ...]), 1),  # 样本0，稀疏矩阵的一行
            (csr_matrix([1, 0, 0, ...]), 0),  # 样本1
            (csr_matrix([0, 1, 0, ...]), 1)   # 样本2
           ]
    """
    data_list = []
    for datum in data:
        """
        np.hstack：水平堆叠（按列拼接）	np.concatenate：通用拼接（可指定轴）
        np.hstack((..., ...))：将特征向量和标签水平拼接成一个更长的向量
        datum[0].toarray()：将稀疏矩阵转为稠密 NumPy 数组，如[[0. 0. 0. ... 0. 0. 0.]]        <class 'numpy.ndarray'>
        
                                    
        test_csr.pkl的内容是一个元组 (features, labels)：
             features(特征矩阵) = （稀疏矩阵）<5043x10591 sparse matrix>    shape=(5043, 10591)
             labels(标签数组)   =  array([[0.], [0.], ...])               shape=(5043, 1)
             
        稀疏矩阵的第一行就是原矩阵第一行的稀疏表示（只存非零元素，零元素全部省略）
        示例：
        dense = np.array([
                          [1, 0, 0, 2, 0, 3, 0, 0, 0, 4],  # 第一行：4个非零元素
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 第二行：全是0
                          [5, 0, 0, 0, 6, 0, 0, 0, 0, 0]   # 第三行：2个非零元素
                         ])
            ↓  ↓  ↓
        csr_matrix(dense)[0]:  # 转成稀疏矩阵第一行
          (np.int32(0), np.int32(0))	1
          (np.int32(0), np.int32(3))	2
          (np.int32(0), np.int32(5))	3
          (np.int32(0), np.int32(9))	4
            ↓  ↓  ↓
        csr_matrix(dense)[0].toarray()：  # 转成密集数组后
          [[1 0 0 2 0 3 0 0 0 4]]
          
        [[0. 0. 0. ... 0. 0. 0.]]          shape=(1, 10591)
             ↓  .ravel()展平  ↓         
         [0. 0. 0. ... 0. 0. 0.]           shape=(10591,)      
        """
        data_list.append(np.hstack((datum[0].toarray().ravel(), datum[1])))
    """
    torch.from_numpy(...)：将 NumPy 数组转换为 PyTorch Tensor。
    .long()：将数据类型转为int64（长整型），因为所有值都是整数（特征已经是整数编码，标签也是（0 / 1）
    data_list:[array([0., 0., 0., ..., 0., 0., 0.])] ——>  tensor([[0, 0, 0,  ..., 0, 0, 0]])
                            <class 'list'>           ——>         <class 'torch.Tensor'>
                            
    torch.from_numpy(...)的示例：
             <class 'numpy.ndarray'>      ——>      <class 'torch.Tensor'>  
                 shape=(10592,)           ——>     shape=torch.Size([10592])
    """
    return torch.from_numpy(np.array(data_list)).long()