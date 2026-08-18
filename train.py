import argparse
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from collections import Counter
import pickle
from tqdm import tqdm
from datetime import datetime
from model import VariationalGNN
from utils import train, evaluate, EHRData, collate_fn
import os
import logging
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,5,6"  # 只用前2张卡

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

def main():
    parser = argparse.ArgumentParser(description='configuraitons')
    parser.add_argument('--result_path', type=str, default='.', help='output path of model checkpoints')
    parser.add_argument('--data_path', type=str, default='./mimc', help='input path of processed dataset')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size')
    parser.add_argument('--num_of_layers', type=int, default=2, help='number of graph layers')
    parser.add_argument('--num_of_heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')  # 随机丢弃 40% 的神经元防止过拟合
    parser.add_argument('--reg', type=str, default="True", help='regularization')  # 控制是否使用 KL 散度正则化
    parser.add_argument('--lbd', type=float, default=1.0, help='regularization')  # KL散度的权重系数，我把 type=float 改成 type=float

    args = parser.parse_args()
    result_path = args.result_path
    data_path = args.data_path
    in_feature = args.embedding_size
    out_feature =args.embedding_size
    n_layers = args.num_of_layers - 1  # 用户输入的是总层数（包含编码器和解码器），VariationalGNN 中 n_layers 表示编码器层数
    lr = args.lr
    args.reg = (args.reg == "True")  # 将字符串转为布尔值
    n_heads = args.num_of_heads
    dropout = args.dropout
    alpha = 0.1  # LeakyReLU 的负斜率
    BATCH_SIZE = args.batch_size
    number_of_epochs = 50
    eval_freq = 1000  # 评估频率

    # 加载数据
    # 原代码，我修改了拼接方式
    # train_x, train_y = pickle.load(open(data_path + 'train_csr.pkl', 'rb'))
    # val_x, val_y = pickle.load(open(data_path + 'validation_csr.pkl', 'rb'))
    # test_x, test_y = pickle.load(open(data_path + 'test_csr.pkl', 'rb'))
    train_x, train_y = pickle.load(open(os.path.join(data_path, 'train_csr.pkl'), 'rb'))
    val_x, val_y = pickle.load(open(os.path.join(data_path, 'validation_csr.pkl'), 'rb'))
    test_x, test_y = pickle.load(open(os.path.join(data_path, 'test_csr.pkl'), 'rb'))
    # 以上为修改后的代码
    """
    加载数据的示例，以test_csr.pkl为例：
    test_x：
        (np.int32(0), np.int32(165))	1.0             
        (np.int32(0), np.int32(708))	1.0             type=<class 'scipy.sparse._csr.csr_matrix'>
        (np.int32(0), np.int32(709))	1.0             shape=(5043, 10591)
        :	:
        (np.int32(5042), np.int32(9205))	1.0
        (np.int32(5042), np.int32(9230))	1.0
        (np.int32(5042), np.int32(9246))	1.0
        
    test_y：  
        [[0.]       对稀疏矩阵求 .shape 返回的是逻辑上的稠密矩阵形状（即完整矩阵的行数和列数），而不是存储格式的形状
         [0.]       type=<class 'numpy.ndarray'>
         [0.]       shape=(5043, 1)
         ...
         [0.]
         [0.]
         [0.]]
    
    np.arange(len(train_y))：生成一个连续的整数数组，代表所有样本的索引。如[0, 1, 2, ..., 999]
    np.where(train_y == 1)[0]：找出所有标签为 1（正类）的样本索引。假设正样本索引为 [12, 45, 78, 230, 567]
    
    train_y的示例：
        [[0.]
         [0.]                        train_y.type=<class 'numpy.ndarray'>
         [0.]                        train_y..shape=(5043, 1)
         ...
         [0.]
         [0.]
         [0.]]


    np.where(train_y==1):
       (array([  14,   15,   34,   48,   50,   51,   55,   69,   73,   74,   77,...]), 
        array([   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,... ]))
     
    np.concatenate((..., ...))：将两部分的索引拼接成一个长数组。则示例结果：[0, 1, 2, ..., 999, 12, 45, 78, 230, 567]
    
    np.repeat的示例：
    arr2d = np.array([[1, 2], [3, 4]])
    np.repeat(arr2d, 2)，不指明维度，默认展平:  
       [1 1 2 2 3 3 4 4]
    np.repeat(arr2d, 2, axis=0):
       [[1 2]
        [1 2]
        [3 4]
        [3 4]]
    np.repeat(arr2d, 2, axis=1):
       [[1 1 2 2]
        [3 3 4 4]]
    """
    # MIMIC 需要对正样本“上采样 2 次”，指的是“把正样本的数量增加到原来的 2 倍”，也就是“复制 1 次”，而不是复制两次
    train_upsampling = np.concatenate((np.arange(len(train_y)), np.repeat(np.where(train_y == 1)[0], 1)))
    """
    一般来说稀疏矩阵可以当完整矩阵一样进行索引操作
    只有这 2 种情况需要特殊处理：
           情况	                              处理方法
         查看具体值	                   .toarray() 转为密集数组
    转换成 PyTorch Tensor	  需要转成密集数组（因为 PyTorch 不支持稀疏矩阵）
    """
    train_x = train_x[train_upsampling]  # train_x 是稀疏矩阵
    train_y = train_y[train_upsampling]

    # 创建结果保存目录 并 配置日志系统
    s = datetime.now().strftime('%Y%m%d%H%M%S')
    result_root = '%s/lr_%s-input_%s-output_%s-dropout_%s'%(result_path, lr, in_feature, out_feature, dropout)
    if not os.path.exists(result_root):
        os.mkdir(result_root)  # 只能创建单级目录，如果父目录不存在会报错
    # logging.root：根日志记录器，handler 是日志的输出目标，多个 handler = 日志会输出到多个地方
    for handler in logging.root.handlers[:]:  # 遍历列表时同时删除元素，会改变列表长度，导致跳过元素或报错，所以用 handlers[:] 而不是 handlers
        logging.root.removeHandler(handler)  # handlers[:]相当于创建了一个副本，遍历副本得到该删掉的元素，索引用的是副本的，删的内容是原本的6
    # % (asctime)s：时间戳             % (message)s：日志消息内容
    # 默认级别是 WARNING，如果不配置 INFO 会被忽略
    logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info("Time:%s" %(s))  # ()是为了防止与 %s 搞混产生歧义

    # 这是总的实际医疗概念（节点）的个数，不包括 padding=0 还有目标节点 length+1
    # 示例：train_x.shape：(5043, 10591)
    num_of_nodes = train_x.shape[1]  # 我把 train_x.shape[1] + 1 改成 train_x.shape[1]
    device_ids = range(torch.cuda.device_count())  # 获取当前机器上可用的 GPU 数量
    "eICU 数据集中有一个关于'既往再入院史'的特征，我们没有将其包含在图结构中。"
    model = VariationalGNN(in_feature, out_feature, num_of_nodes, n_heads, n_layers,
                           dropout=dropout, alpha=alpha, variational=args.reg, none_graph_features=0).to(device)
    """
    batch 数据 → 分发到多个 GPU
    ├── GPU 0: 处理 8 个样本
    ├── GPU 1: 处理 8 个样本
    ├── GPU 2: 处理 8 个样本
    └── GPU 3: 处理 8 个样本
         ↓
    收集各 GPU 的输出
         ↓
    在 GPU 0 上计算损失和反向传播
    """
    model = nn.DataParallel(model, device_ids=device_ids)  # 将模型包装成 DataParallel，在多个 GPU 上并行训练
    val_loader = DataLoader(dataset=EHRData(val_x, val_y), batch_size=BATCH_SIZE,
                            collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=False)
    """
    示例：
    创建一个模型：
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 3 个线性层
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 1)
    
    # 假设要冻结 fc1（迁移学习，保持预训练参数不动）
    for param in model.fc1.parameters():
        param.requires_grad = False  # 冻结 fc1 的参数
    """
    # 创建 Adam 优化器
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-8)
    """
    torch.optim.lr_scheduler.StepLR：一种阶梯式学习率调整策略。
    step_size=5：每 5 个 Epoch（轮次），调整一次学习率。
    gamma=0.5：调整系数。
    效果：每次调整时，学习率 = 当前学习率 × 0.5。即学习率每 5 轮减半。
    """
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练模型
    for epoch in range(number_of_epochs):
        """
        optimizer（优化器）内部有一个属性叫 param_groups，它是一个列表，用来存储不同的参数组。
        Optimizer会单独存储一份需要计算梯度的参数引用。优化器内部是以字典（dict）形式存储参数和超参数的。
        我们所有参数都用同一个学习率，所以只需要 1 个组（即 param_groups[0]）
        
        示例：设置不同学习率,两个参数组，不同学习率
        optimizer = optim.Adam([
            {'params': model.fc1.parameters(), 'lr': 1e-5},
            {'params': model.fc2.parameters(), 'lr': 1e-3}
        ])

        组1学习率: optimizer.param_groups[0]['lr'] = 1e-5
        组2学习率: optimizer.param_groups[1]['lr'] = 1e-3
        """
        # 监控学习率的变化
        print("Learning rate:{}".format(optimizer.param_groups[0]['lr']))
        # 添加一段代码，原代码的 train_y.shape=(40229, 1)，Counter处理不了二维数组，所以做出修改
        if len(train_y.shape) > 1:
            train_y = train_y.ravel()  # (40229, 1) → (40229,)
        # 以上为修改后添加的代码
        # Counter(train_y)：统计训练集标签 train_y 中 0 和 1 的个数
        # train_y是上采样后的标签数组。计算正负样本比例，用于给损失函数中的正样本加权
        ratio = Counter(train_y)
        """
        train_loader 返回一个可迭代的批次加载器对象，每次迭代返回一个完整的 Tensor 批次。
        每次迭代的返回结果的示例：
          tensor([[0, 0, 0,  ..., 0, 0, 0],
                  [0, 0, 0,  ..., 0, 0, 0],        当 batch_size = 16 时，
                  [0, 0, 0,  ..., 0, 0, 0],        形状: torch.Size([16, 10592])
                  [0, 0, 0,  ..., 0, 0, 0],        
                  [0, 0, 0,  ..., 0, 0, 0]])
                  
        DataLoader 如何使用 Dataset，DataLoader 内部会做：
         1. 生成随机索引（shuffle=True）
         2. 调用 dataset[idx] 取样本
         3. 收集 batch_size 个样本
         4. 传给 collate_fn 拼装
        如果没有 EHRData ，DataLoader 不知道：
          - 数据集有多大？
          - 怎么批量取？               
          - 怎么打乱顺序？
          
        EHRData 把原始的 NumPy/稀疏矩阵数据，包装成 PyTorch 能识别的 Dataset 对象，让 DataLoader 能够正常取数据
        collate_fn 是把多行样本数据（特征+标签）拼装成可以整齐输出的 Tensor 批次
        """
        train_loader = DataLoader(dataset=EHRData(train_x, train_y), batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn, num_workers=torch.cuda.device_count(), shuffle=True)
        # ratio：Counter({np.float32(0.0): 36009, np.float32(1.0): 4220})，表示标签为 0 的样本有 36009 个
        # pos_weight：正样本权重，用于类别不平衡处理
        pos_weight = torch.ones(1).float().to(device) * (ratio[False] / ratio[True])
        """
        reduction = "sum"：对 batch 内的所有样本损失求和（而不是求平均），因为 KL 散度也是 batch 求和
        "mean"：求平均（默认），sum / batch_size
        
        BCEWithLogitsLoss：二分类交叉熵损失（内部自带Sigmoid激活函数）
                     分开写	                                      合并写
        sigmoid(logits) → BCE(pred, label)	          BCEWithLogitsLoss(logits, label)
        """
        criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
        """
        tqdm库的进度条函数：
          iter(train_loader)把 DataLoader 变成迭代器
          leave = False：进度条完成后自动消失
          total = len(train_loader)，进度条总步数 = batch 总数
        """
        # 把 train_loader 包装成一个带进度条的迭代器，它只是把原来的迭代器（train_loader）包装了一层，添加了进度条显示功能，但迭代功能完全保留
        t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
        # model.train()  # 在 train() 函数里也调用了 model.train()，这里有点多余
        total_loss = np.zeros(3)
        for idx, batch_data in enumerate(t):
            # idx 是批次的索引
            loss, kld, bce = train(batch_data, model, optimizer, criterion, args.lbd, 5)
            total_loss += np.array([loss, bce, kld])  # 把它们包装成一个形状为 [3] 的 NumPy 数组
            if idx % eval_freq == 0 and idx > 0:
                # model.state_dict() 是 PyTorch 中获取模型所有参数的方法，返回一个字典，键是参数名称，值是参数张量
                torch.save(model.state_dict(), "{}/parameter{}_{}".format(result_root, epoch, idx))  # model.state_dict()：模型的所有参数（权重和偏置）
                val_auprc, _ = evaluate(model, val_loader, len(val_y))  # val_y 是样本总数
                logging.info('epoch:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' %
                             (epoch + 1, val_auprc, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
                print('epoch:%d AUPRC:%f; loss: %.4f, bce: %.4f, kld: %.4f' %
                      (epoch + 1, val_auprc, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
            if idx % 50 == 0 and idx > 0:
                # 每 50 个batch更新一次进度条：
                # t.set_description(...)：更新进度条前面的描述文字
                # t.refresh()：强制刷新显示
                t.set_description('[epoch:%d] loss: %.4f, bce: %.4f, kld: %.4f' %
                                  (epoch + 1, total_loss[0]/idx, total_loss[1]/idx, total_loss[2]/idx))
                t.refresh()
        scheduler.step()  # 更新学习率


if __name__ == '__main__':
    main()
