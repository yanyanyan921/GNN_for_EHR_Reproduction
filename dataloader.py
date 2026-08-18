from torch.utils.data import Dataset
import pickle
import numpy as np
import torch
import os


def load_pickle(fname):
    with open(fname, 'rb') as f:  
        return pickle.load(f)


# 针对不平衡数据进行下采样的函数，目的是平衡正负样本的比例
def downsample(train_idx, neg_young, train_idx_pos):
    """
    train_idx：所有训练样本的索引（通常包含正样本和负样本）。
    neg_young：负样本（年轻患者）的索引。
    train_idx_pos：正样本（老年患者/患病者）的索引。
    """
    # np.random.permutation 是 NumPy 中用于随机排列（打乱）数组元素的函数。它返回一个打乱顺序后的新数组，不修改原数组
    # Python 列表（打印有逗号）        NumPy 数组（打印无逗号，空格分隔）
    downsamples = np.random.permutation(neg_young)[:450000]
    mask=np.ones(len(train_idx), bool)  # 创建一个全 True 的布尔数组，长度等于总样本数
    mask[downsamples] = False  # 将 downsamples 中选中的负样本位置设为 False，可以直接指示列表内的索引的位置
    # train_idx[mask]:根据掩码保留的样本（所有正样本 + 剩余负样本）
    # np.repeat(train_idx_pos, 50)：将正样本索引重复 50 次（过采样）
    # np.concatenate(...)：将上述两部分拼接成新的索引数组
    downsample_idx = np.concatenate((train_idx[mask], np.repeat(train_idx_pos,50)))
    return downsample_idx


# 用于封装数据加载和预处理逻辑
class OriginalData:
    def __init__(self, path):
        self.path = path
        self.feature_selection = load_pickle(os.path.join(path,'frts_selection.pkl'))  # 修改文件拼接路径方式
        """
        frts_selection:
        [40 153 168 ... 133276 133277 133278]
        
        preprocess_x:这是所有的数据  
        preprocess_x.shape:(1000, 133279)
        (np.int32(0), np.int32(123))	1
        (np.int32(0), np.int32(1650))	1
        : :
        (np.int32(999), np.int32(131245))	1
        (np.int32(999), np.int32(133011))	1
        (np.int32(999), np.int32(133277))	1
        
        y_bin:这是所有的数据
        [False False ... False False False]
        """
        self.x = load_pickle(os.path.join(path,'preprocess_x.pkl'))[:, self.feature_selection]
        self.y = load_pickle(os.path.join(path,'y_bin.pkl'))
    # 根据索引文件加载数据，并可选地进行采样平衡
    def datasampler(self, idx_path, train = True):
        idx = load_pickle(os.path.join(self.path,idx_path))
        if train:  # 在Python中，True 和 1 是相等的，可以比较；  self.y[idx]：从标签数组中取出 idx 对应位置的标签
            downsample_idx = downsample(idx, load_pickle(os.path.join(self.path,'neg_young.pkl')), idx[self.y[idx] == 1])
            """
            根据 downsample_idx 从特征矩阵和标签中提取对应的行
            
            preprocess_x[downsample_idx,:]的示例，假如downsample_idx=[0 1 7 7]
            此处的行索引是针对筛选出的数据而言的，不是preprocess_x的原来行索引，而且是稀疏矩阵的形式，只显示值不为0的索引
            (np.int32(0), np.int32(123))     1
            (np.int32(0), np.int32(1650))    1
            : :
            (np.int32(3), np.int32(129677))  1
            (np.int32(3), np.int32(133273))  1
            
            假如downsample_idx=[0 1 3 3]，y_bin=np.array([False,False,False,True])
            y_bin[downsample_idx]=[False False True True]
            """
            return self.x[downsample_idx, :], self.y[downsample_idx]
        return self.x, self.y


