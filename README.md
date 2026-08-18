# VGNN-EHR-Reproduction

这是论文 *Variationally regularized graph-based representation learning for electronic health records* 代码的**完整复现过程**。

---

## 📌 项目说明

我将原代码的多处问题都做了修改，感觉作者上传的代码并非是他们自己运行成功的一版，有几处大的问题。因为我是跨考生，所以代码是逐行跟着调试过的，我对所有代码都做了非常详细的注释和笔记，这些记录了我对代码的熟悉过程。

因为论文比较早，所有的包我都进行了适配版本的下载，并都写在了 `requirements.txt` 里，希望能给到帮助。

我只在 **MIMIC-III** 上训练并验证了。论文实验结果是 **0.7102 ± 0.0046**，我的第一次复现结果为 **0.7111**。

---

## 📁 数据准备

本仓库提供预处理过的数据文件，可以直接运行，**不需要自己下载 MIMIC-III 原始数据**，只需要解压storage这个文件夹到该项目的目录下：

- `train_csr.pkl`
- `validation_csr.pkl`
- `test_csr.pkl`

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行训练（参考命令示例）

```bash
python3 train.py \
  --data_path /data/newhome/wzy/Projects/GNN_for_EHR-master/storage \
  --embedding_size 768 \
  --dropout 0.2 \
  --result_path /data/newhome/wzy/Projects/GNN_for_EHR-master/reuslt
```

其中：

- `storage`：存放预处理后的 `.pkl` 数据文件，解压即可
- `result`：存放训练日志和模型参数

只需将相对路径修改为项目所在的目录即可。其他超参数可对照论文附录进行修改。

> ⚠️ 注意：`num_of_layers` 指的是总层数，需要减去解码器的一层，不过一般不用改。

---

## ⚠️ 复现细节与坑点

### 数据集上采样倍数不同

对不同数据集，需要在 `train.py` 中按照论文要求修改对应的上采样倍数：

```python
train_upsampling = np.concatenate((np.arange(len(train_y)), np.repeat(np.where(train_y == 1)[0], 1)))
```

- **MIMIC-III**：正样本上采样 2 倍，复制 1 次即可
- **AD-EHR**：正样本上采样 50 倍
- **eICU**：正样本上采样 2 倍

---

## 📊 实验结果

| 指标 | 论文结果 | 本仓库复现结果 |
|:---|:---|:---|
| AUPRC | 0.7102 ± 0.0046 | **0.7111** |

---

## 📝 备注

- 所有代码均已添加详细中文注释，便于理解
- 依赖包版本已锁定，见 `requirements.txt`
- 如有问题，欢迎提 Issue 交流
