'''
This code is adapted from process steps on eICU of previous works (cited)
https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer
'''

import pandas as pd
import csv
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import sys
import pickle
from sklearn import model_selection
import argparse
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix


class EncounterInfo(object):  # 用于存储一次医疗就诊/访视（Encounter）的完整信息
    def __init__(self, patient_id, encounter_id, encounter_timestamp, expired):
        self.patient_id = patient_id
        self.encounter_id = encounter_id  # 本次就诊唯一标识ID
        self.encounter_timestamp = encounter_timestamp  # 就诊时间戳
        self.expired = expired  # 表示该就诊记录是否已过期/失效
        self.dx_ids = []  # 用于存储本次就诊的诊断代码ID（Diagnosis IDs）
        self.rx_ids = []  # 用于存储本次就诊的处方/药物ID
        self.labs = {}  # 用于存储实验室检查数据
        self.physicals = []  # 用于存储本次就诊的体格检查数据（如血压、心率等多项检查结果）
        self.treatments = []  # 用于存储本次就诊的治疗方案或操作记录
        self.labvalues = []  # 用于存储实验室检查的数值结果

    def __str__(self):  # 在类中定义 __str__ 方法，当使用 print() 时自动调用：
        return (f"EncounterInfo(\n"
                f"  patient_id={self.patient_id},\n"
                f"  encounter_id={self.encounter_id},\n"
                f"  timestamp={self.encounter_timestamp},\n"
                f"  expired={self.expired},\n"
                f"  dx_ids={self.dx_ids},\n"
                f"  rx_ids={self.rx_ids},\n"
                f"  labs={self.labs},\n"
                f"  physicals={self.physicals},\n"
                f"  treatments={self.treatments},\n"
                f"  labvalues={self.labvalues}\n"
                f")")


# 字典键为就诊ID，值是存存储一次医疗就诊的完整信息的类，{encounter_id1：EncounterInfo1，encounter_id2：EncounterInfo2}
def process_patient(infile, encounter_dict, min_length_of_stay=0):
    inff = open(infile, 'r')
    count = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)  # 每处理10000条记录打印进度
            sys.stdout.flush()
        patient_id = line['SUBJECT_ID']
        encounter_id = line['HADM_ID']
        # strptime 是 "string parse time" 的缩写，意思是将字符串解析为时间对象
        encounter_timestamp = datetime.strptime(line['ADMITTIME'], '%Y-%m-%d %H:%M:%S')  # 将入院时间字符串转换为datetime对象
        expired = line['HOSPITAL_EXPIRE_FLAG'] == "1"  # 判断患者是否在住院期间死亡。返回True或False
        if (datetime.strptime(line['DISCHTIME'], '%Y-%m-%d %H:%M:%S') - encounter_timestamp).days < min_length_of_stay:
            continue  # 计算住院天数（出院时间 - 入院时间），如果住院天数小于 min_length_of_stay 参数指定的最小天数，则跳过该条记录（不加入最终结果）

        ei = EncounterInfo(patient_id, encounter_id, encounter_timestamp, expired)
        if encounter_id in encounter_dict:
            print('Duplicate encounter ID!!')
            print(encounter_id)
            sys.exit(1)  # 如果重复 → 立即停止当前 Python 程序的运行
        encounter_dict[encounter_id] = ei
        count += 1
    inff.close()
    print('')
    return encounter_dict


# 添加单次医疗就诊的ICD-9诊断代码
def process_diagnosis(infile, encounter_dict):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        encounter_id = line['HADM_ID']
        dx_id = line['ICD9_CODE'].lower()  # 提取ICD-9诊断代码
        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].dx_ids.append(dx_id)
        count += 1
    inff.close()
    print('')
    print('Diagnosis without Encounter ID: %d' % missing_eid)
    return encounter_dict


# 添加单次医疗就诊的ICD-9治疗代码
def process_treatment(infile, encounter_dict):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        encounter_id = line['HADM_ID']
        treatment_id = line['ICD9_CODE'].lower()
        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].treatments.append(treatment_id)
        count += 1
    inff.close()
    print('')
    print('Treatment without Encounter ID: %d' % missing_eid)
    return encounter_dict


# 将实验室检测数据按照检查项目ID分组计算平均值和方差，整理为{'ITEMID1': (mean1, std1), 'ITEMID2': (mean2, std2)}
def get_lab_mean_std(lab_file, train_ids):
    lab_data = pd.read_csv(lab_file)  # pd.read_csv()一次性加载全部数据到内存
    """
    步骤1: 将SUBJECT_ID转换为字符串    步骤2: 将HADM_ID格式化为整数（去掉小数）    步骤3: 拼接成 "SUBJECT_ID:HADM_ID" 格式
    步骤4: 检查是否在train_ids列表中    步骤5: 用布尔索引筛选DataFrame
    """
    lab_data = lab_data[(lab_data['SUBJECT_ID'].astype('str') + ':' +
         lab_data['HADM_ID'].apply(lambda x: f'{x:.0f}')).isin(train_ids)]
    # 将 VALUENUM 列中的数据强制转换为数值类型，errors = 错误处理方式，如果无法转换则变成 NaN（空值）
    lab_data['VALUENUM'] = pd.to_numeric(lab_data['VALUENUM'], errors='coerce')
    # VALUE是检验结果的原始文本值，VALUENUM是检验结果的数值化值，只存储可数值化的结果
    lab_data = lab_data[lab_data['VALUENUM'].notna()]  # 不用点属性而是方括号取列更好
    """
    grouped.groups的输出: {50809: [0, 1, 2, 6], 51265: [3, 4, 7], 51221: [5]}，每个组包含对应的行索引
    .reset_index()将索引重置为整数序列，让ITEMID 从索引变为普通列
    mean_std（pd.DataFrame）的示例如下：
                      VALUENUM                 ← 第一层列名（原始列名）
                       mean          std       ← 第二层列名（聚合函数名）
            ITEMID                             ← 行索引名（分组键）
      ─────────────────────────────────────
      0     50802    -3.272727     4.485392
      1     50804    22.656566     4.305103
    """
    mean_std = lab_data.groupby('ITEMID').agg({'VALUENUM': ['mean', "std"]}).reset_index()
    mean_std = mean_std[mean_std['VALUENUM']['mean'].notna() & mean_std['VALUENUM']['std'].notna()]
    """
    zip()组合将 ITEMID列表和 (mean, std) 列表配对，astype('str')可以把整个列表的元素转换类型，如[('50809', (7.275, 0.171)), ('51265', (139.0, 3.606))]
    zip() 返回的是一个迭代器（iterator），只能遍历一次：遍历完后就不能再次使用
    mean_std.iterrows()，遍历 DataFrame 的每一行，返回 (索引, 行数据)
    """
    mean_std = dict(zip(np.array(mean_std['ITEMID']).astype('str'),
                        [(row['VALUENUM']['mean'], row['VALUENUM']['std'])
                         for _, row in mean_std.iterrows()]))
    return mean_std  # 返回的结果是{'50809': (7.275, 0.171), '51265': (139.0, 3.606)}


# 将入院24小时内的实验室检验值离散化，并添加到对应的就诊记录中
def process_lab(infile, encounter_dict, mean_std):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        encounter_id = line['HADM_ID']
        if len(encounter_id) == 0:  # 就诊ID得存在
            continue
        # 示例：ITEMID = 50809(血小板计数)，ITEMID = 51265(收缩压)，ITEMID = 51221(血糖)
        lab_id = line['ITEMID']  # 我删了.lower()
        lab_time = datetime.strptime(line['CHARTTIME'], '%Y-%m-%d %H:%M:%S')
        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        if lab_id in mean_std:  # 项目ID得有效，存在平均值和方差
            try:
                lab_value = float(line['VALUENUM'])
            except:
                missing_eid += 1
                continue
            mean, std = mean_std[lab_id]
            suffix = "_(>10)"  # 设置默认后缀为 _(>10)，表示偏离均值超过10个标准差
            """
            1. 读取整个 CSV 文件
            2. 按 ITEMID 分组，计算每个检验项目的均值和标准差
            3. 对每个具体的检验值，使用其对应 ITEMID 的均值和标准差来打标签
            """
            for lab_range in ['-10', '-3', '-1', '-0.5', '0.5', '1', '3', '10']:  # 遍历阈值列表，找到第一个满足条件的范围
                if lab_value < mean + float(lab_range) * std :
                    suffix = "_({})".format(lab_range)
                    break
            admission_time = encounter_dict[encounter_id].encounter_timestamp
            if (lab_time - admission_time).days < 1:
                encounter_dict[encounter_id].labvalues.append(lab_id + suffix)  # 将检验值离散化为范围标签
        count += 1
    inff.close()
    print('')
    print('Lab without Encounter ID: %d' % missing_eid)
    return encounter_dict


# 将就诊数据转换为TensorFlow的SequenceExample格式，单次就诊数据转化为一个seqex
def build_seqex(encounter_dict,
                skip_duplicate=False,
                min_num_codes=1,
                max_num_codes=50):
    key_list = []
    seqex_list = []
    dx_str2int = {}  # 诊断代码字符串→整数映射字典
    treat_str2int = {}  # 治疗代码字符串→整数映射字典
    lab_str2int = {}  # 检验代码字符串→整数映射字典
    num_cut = 0
    num_duplicate = 0
    count = 0
    num_dx_ids = 0
    num_treatments = 0
    num_labs = 0
    num_unique_dx_ids = 0
    num_unique_treatments = 0
    num_unique_labs = 0
    min_dx_cut = 0
    min_treatment_cut = 0
    min_lab_cut = 0
    max_dx_cut = 0
    max_treatment_cut = 0
    max_lab_cut = 0
    num_expired = 0

    for _, encounter_info in encounter_dict.items():  # 取键值对
        if skip_duplicate:
            if (len(encounter_info.dx_ids) > len(set(encounter_info.dx_ids)) or len(encounter_info.treatments) > len(set(encounter_info.treatments))):
                num_duplicate += 1
                continue

        if len(set(encounter_info.dx_ids)) < min_num_codes:
            min_dx_cut += 1
            continue

        if len(set(encounter_info.treatments)) < min_num_codes:
            min_treatment_cut += 1
            continue

        if len(set(encounter_info.labvalues)) < min_num_codes:
            min_lab_cut += 1
            continue

        if len(set(encounter_info.dx_ids)) > max_num_codes:
            max_dx_cut += 1
            continue

        if len(set(encounter_info.treatments)) > max_num_codes:
            max_treatment_cut += 1
            continue

        if len(set(encounter_info.labvalues)) > max_num_codes:
            max_lab_cut += 1
            continue

        count += 1
        num_dx_ids += len(encounter_info.dx_ids)
        num_treatments += len(encounter_info.treatments)
        num_labs += len(encounter_info.labvalues)
        num_unique_dx_ids += len(set(encounter_info.dx_ids))
        num_unique_treatments += len(set(encounter_info.treatments))
        num_unique_labs += len(set(encounter_info.labvalues))

        for dx_id in encounter_info.dx_ids:
            if dx_id not in dx_str2int:
                dx_str2int[dx_id] = len(dx_str2int)  # 为每个诊断代码分配唯一的整数ID（从0开始递增）

        for treat_id in encounter_info.treatments:
            if treat_id not in treat_str2int:
                treat_str2int[treat_id] = len(treat_str2int)

        for lab_id in encounter_info.labvalues:
            if lab_id not in lab_str2int:
                lab_str2int[lab_id] = len(lab_str2int)
        """
        # tf.train.Feature
        Feature = Union[List[bytes],
                        List[int64],
                        List[float]]

        # tf.train.FeatureList
        FeatureList = List[Feature]

        # tf.train.FeatureLists
        FeatureLists = Dict[str, FeatureList]

        # tf.train.SequenceExample
        class SequenceExample(typing.NamedTuple):
          context: Dict[str, Feature]  (上下文特征)
          feature_lists: FeatureLists  (序列特征)
        """
        seqex = tf.train.SequenceExample()  # 创建一个TensorFlow SequenceExample对象（用于序列数据）
        """
        context是一个类实例{
          feature={'patientId':["533:100009"],'label':[1]}
        }
        """
        seqex.context.feature['patientId'].bytes_list.value.append(bytes(encounter_info.patient_id + ':' +
                                                                         encounter_info.encounter_id, 'utf-8'))  # 将字符串编码为字节串（bytes），类型要求
        if encounter_info.expired:
            seqex.context.feature['label'].int64_list.value.append(1)
            num_expired += 1
        else:
            seqex.context.feature['label'].int64_list.value.append(0)
        """
        feature_lists是一个类实例{
          feature_list={'dx_ids':FeatureList}
        }
        FeatureList是一个类实例{
          feature=[['41401', '99604', '4142', '25808', '27888', '8535']]
        }
        """
        dx_ids = seqex.feature_lists.feature_list['dx_ids']
        """
        .feature.add() 的作用是创建一个新的 Feature 对象（每个代表一个时间步），然后向其中填充数据。
        FeatureList (dx_ids)
        └── feature: []  ← 空的（没有 Feature）
        
        实例：
        encounter_info = {'patient_id': 533,
                  'encounter_id': 100009,
                  'timestamp': '2162-05-16 15:56:00',
                  'expired': False,
                  'dx_ids': ['41401', '99604', '4142', '25808', '27888', '8535'],
                  'rx_ids': [],
                  'labs': {},
                  'physicals': [],
                  'treatments': ['3613', '3615', '3795', '3961'],
                  'labvalues': ['50821_3', '50822_3', '50861_10', '50862_3']
                  }
                  
        seqex.feature_lists.feature_list大概长这样：
        {
         'dx_ids': feature [["27888","4142","41401","99604","8535","25808"],[...]],
         'dx_ints': feature [[4,2,0,1,5,3],[...]],
         
         'proc_ids': feature [["3613","3615","3961","3795"],[...]],
         'proc_ints': feature [[0,1,3,2],[...]],
         
         'lab_ids': feature [["50861_10","50862_3","50822_3","50821_3"],[...]],
         'dx_ints': feature [[2,3,1,0],[...]],
        }
        """
        dx_ids.feature.add().bytes_list.value.extend(list([bytes(s, 'utf-8') for s in set(encounter_info.dx_ids)]))

        dx_int_list = [dx_str2int[item] for item in set(encounter_info.dx_ids)]  # 删了list()
        dx_ints = seqex.feature_lists.feature_list['dx_ints']
        dx_ints.feature.add().int64_list.value.extend(dx_int_list)

        proc_ids = seqex.feature_lists.feature_list['proc_ids']
        proc_ids.feature.add().bytes_list.value.extend(list([bytes(s, 'utf-8') for s in set(encounter_info.treatments)]))

        proc_int_list = [treat_str2int[item] for item in list(set(encounter_info.treatments))]
        proc_ints = seqex.feature_lists.feature_list['proc_ints']
        proc_ints.feature.add().int64_list.value.extend(proc_int_list)

        lab_ids = seqex.feature_lists.feature_list['lab_ids']
        lab_ids.feature.add().bytes_list.value.extend(list([bytes(s, 'utf-8') for s in set(encounter_info.labvalues)]))

        lab_int_list = [lab_str2int[item] for item in list(set(encounter_info.labvalues))]
        lab_ints = seqex.feature_lists.feature_list['lab_ints']
        lab_ints.feature.add().int64_list.value.extend(lab_int_list)

        seqex_list.append(seqex)
        key = seqex.context.feature['patientId'].bytes_list.value[0]  # 相当于取列表第一个值
        key_list.append(key)  # key_list：[b'533:100009'] <class 'bytes'>

    print('Filtered encounters due to duplicate codes: %d' % num_duplicate)
    print('Filtered encounters due to thresholding: %d' % num_cut)
    print('Average num_dx_ids: %f' % (num_dx_ids / count))
    print('Average num_treatments: %f' % (num_treatments / count))
    print('Average num_labs: %f' % (num_labs/ count))
    print('Average num_unique_dx_ids: %f' % (num_unique_dx_ids / count))
    print('Average num_unique_treatments: %f' % (num_unique_treatments / count))
    print('Average num_unique_labs: %f' % (num_unique_labs / count))
    print('Min dx cut: %d' % min_dx_cut)
    print('Min treatment cut: %d' % min_treatment_cut)
    print('Min lab cut: %d' % min_lab_cut)
    print('Max dx cut: %d' % max_dx_cut)
    print('Max treatment cut: %d' % max_treatment_cut)
    print('Max lab cut: %d' % max_lab_cut)
    print('Number of expired: %d' % num_expired)
    return key_list, seqex_list, dx_str2int, treat_str2int, lab_str2int


def train_val_test_split(patient_ids, random_seed=0):
    train_ids, test_ids = model_selection.train_test_split(patient_ids, test_size=0.2, random_state=random_seed)
    test_ids, val_ids = model_selection.train_test_split(test_ids, test_size=0.5, random_state=random_seed)
    return train_ids, val_ids, test_ids


# 用于根据指定的ID集合筛选SequenceExample数据
def get_partitions(seqex_list, id_set=None):
    total_visit = 0
    new_seqex_list = []
    for seqex in seqex_list:
        total_visit += 1  # 换了个位置
        if total_visit % 1000 == 0:
            sys.stdout.write('Visit count: %d\r' % total_visit)
            sys.stdout.flush()
        key = seqex.context.feature['patientId'].bytes_list.value[0].decode('utf-8')  # b'533:100009'  ——>  '533:100009'
        if (id_set is not None and key not in id_set):                                # <class 'bytes'>     <class 'str'>
            continue
        new_seqex_list.append(seqex)
    return new_seqex_list


# 定义解析函数 parser_fn，用于将序列化的SequenceExample解析为模型可用的格式
def parser_fn(serialized_example):
    context_features_config = {
        'patientId': tf.io.VarLenFeature(tf.string),  # 使用 VarLenFeature（变长特征），类型为字符串
        'label': tf.io.FixedLenFeature([1], tf.int64),  # 使用 FixedLenFeature（定长特征），形状为 [1]，类型为整数
    }
    sequence_features_config = {
        'dx_ints': tf.io.VarLenFeature(tf.int64),  # 因为版本改变，把tf.VarLenFeature改为tf.io.VarLenFeature
        'proc_ints': tf.io.VarLenFeature(tf.int64),
        'lab_ints': tf.io.VarLenFeature(tf.int64)
    }
    """
    使用 tf.io.parse_single_sequence_example 将序列化的字节串解析为：
    batch_context：Context特征字典
    batch_sequence：Sequence特征字典
    
    不能直接传递 SequenceExample 对象给 tf.io.parse_single_sequence_example，该函数期望接收的是序列化的字节串（bytes 类型）
    SequenceExample 对象，需要先调用 .SerializeToString()
    
    SparseTensor 用 (批次索引, 位置索引) 的格式来表示数据的位置，实例如下：
    indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
              ↑       ↑       ↑       ↑       ↑       ↑
             位置0   位置1   位置2   位置3   位置4   位置5
            (第0行) (第0行) (第0行) (第0行) (第0行) (第0行)
    values=[2, 5, 1, 0, 4, 3],
    
    tf.io.parse_single_sequence_example()将序列化的数据解析为两个字典，示例如下：
    batch_context = {
    'patientId': SparseTensor(...),  # 或 Tensor
    'label': Tensor([0])             # 或 Tensor
    }
    batch_sequence = {
    'dx_ints': SparseTensor(...),
    'proc_ints': SparseTensor(...),
    'lab_ints': SparseTensor(...)
    }
    """
    (batch_context, batch_sequence) = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=context_features_config,
        sequence_features=sequence_features_config)
    # tf.cast：转换为float32类型      tf.squeeze：压缩维度（去掉大小为1的维度），比如[1.0] ——> 1.0
    labels = tf.squeeze(tf.cast(batch_context['label'], tf.float32))  # Tensor(1.0)而不是 Tensor([1.0])
    return batch_sequence, labels


# 将TFRecord格式的SequenceExample转换为CSR（Compressed Sparse Row）稀疏矩阵格式
def tf2csr(output_path, partition, maps):
    """
    output_path：输出文件路径前缀
    partition：数据集分区名称（如 'train'、'val'、'test'）
    maps：词汇表映射列表 [dx_str2int, treat_str2int, lab_str2int]
    """
    num_epochs = 1
    buffer_size = 32
    dataset = tf.data.TFRecordDataset(os.path.join(output_path, partition + ".tfrecord"))  # 从TFRecord文件读取数据集，修改了路径拼接方式
    dataset = dataset.shuffle(buffer_size)  # 打乱数据集顺序（缓冲区大小为32）
    # 没有repeat：只能遍历一次   有repeat：可以多次遍历
    dataset = dataset.repeat(num_epochs)  # 重复数据集（1次，即不重复）
    # map_func：要应用的函数，接收一个输入，返回一个或多个输出        num_parallel_calls：并行处理的线程数（用于加速）
    # 数据格式: [(sequence1, label1), (sequence2, label2), ...]
    dataset = dataset.map(parser_fn, num_parallel_calls=4)  # 对数据集中的每个样本应用转换函数
    dataset = dataset.batch(1)  # 批次大小为1（逐个处理样本）
    dataset = dataset.prefetch(16)  # 预取16个样本，提高数据加载效率
    count = 0
    np_data = []
    np_label = []
    for data in dataset:
        count += 1
        np_datum = np.zeros(sum([len(m) for m in maps]))  # 创建全零向量，maps的所有元素的长度组成一个列表，再统计列表元素之和，全零向量长度 = 所有词汇表大小之和
        """
        data[0] = {'dx_ints': SparseTensor, 'proc_ints': SparseTensor, 'lab_ints': SparseTensor}
        
        稀疏张量（只存储非零值的位置和值）
        SparseTensor(
            indices=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
            values=[4, 1, 2, 3, 0, 5],
            dense_shape=[1, 6]
        )
        ↓ to_dense() ↓
        密集张量（存储所有值）
        Tensor([[4, 1, 2, 3, 0, 5]], shape=(1, 6), dtype=int64)
        ↓ .numpy() ↓
        NumPy二维数组
        array([[4, 1, 2, 3, 0, 5]])
        ↓ .ravel() ↓
        NumPy一维数组
        array([4, 1, 2, 3, 0, 5])
        """
        dx_pos = tf.sparse.to_dense(data[0]['dx_ints']).numpy().ravel()  # shape(1, 6) 表示这是一个二维数组 ——> shape:(6,)
        """
        提取治疗代码的整数索引，并加上诊断词汇表的大小作为偏移量:
        maps[:1] = [dx_str2int]
        offset = len(dx_str2int) = 1000
        proc_pos = [3, 1, 0, 2] + 1000 = [1003, 1001, 1000, 1002]
        """
        proc_pos = tf.sparse.to_dense(data[0]['proc_ints']).numpy().ravel() + \
                   sum([len(m) for m in maps[:1]])
        lab_pos = tf.sparse.to_dense(data[0]['lab_ints']).numpy().ravel() + \
                  sum([len(m) for m in maps[:2]])
        """
        假设 np_datum 长度为 3500，将对应位置的元素设为1（One-hot编码）：
        dx_pos = [4, 1, 2, 3, 0, 5] → 这些位置设为1
        proc_pos = [1003, 1001, 1000, 1002] → 这些位置设为1
        lab_pos = [1500, 1503, 1502, 1501] → 这些位置设为1
        """
        np_datum[dx_pos] = 1
        np_datum[proc_pos] = 1
        np_datum[lab_pos] = 1
        np_data.append(np_datum)
        # data[1].numpy()[0] → 转换为numpy并取第一个值
        # data[1].shape:()，data[1]:0.0
        # type(data[1]):<class 'tensorflow.python.framework.ops.EagerTensor'>
        np_label.append(data[1].numpy())  # 我把data[1].numpy()[0]改成了data[1].numpy()
        sys.stdout.write('%d\r' % count)
        sys.stdout.flush()
    """
    稀疏矩阵的表示示例：
    matrix = [
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 1]
    ]
    indptr = [0, 1, 3, 4]    维度=行数+1，计算行数
              │  │  │  │
              │  │  │  └──   结束的data位置（总非零元素数 = 4），4不参与计算
              │  │  └─────   行2的data起始位置 = 3
              │  └────────   行1的data起始位置 = 1
              └───────────   行0的data起始位置 = 0
    data =    [1, 1, 1, 1]
    indices = [1, 0, 2, 3]   每个非零值所在的列索引，计算列数
    
    pickle 是Python的对象序列化模块，可以将任何Python对象保存到文件中
    
    np_data 是一个列表，包含所有样本的特征向量:
    np_data = [
         [0, 0, 1, 0, 0, 0, 1, 0, ...],  # 样本1
         [0, 1, 0, 0, 0, 0, 0, 0, ...],  # 样本2
         [0, 0, 0, 0, 1, 0, 0, 1, ...],  # 样本3
    ]
    
    np_label 是一个列表，包含所有标签:
    np_label = [0, 1, 0, 0, 1, 0, ...]
    
    _csr.pkl文件读取情况：（这是元组格式）
    ①csr_matrix(np.array(np_data):  
    (np.int32(0), np.int32(0))	1.0
    (np.int32(0), np.int32(1))	1.0
    (np.int32(0), np.int32(2))	1.0
    (np.int32(0), np.int32(3))	1.0
    (np.int32(0), np.int32(4))	1.0     (行, 列) = 值
    (np.int32(0), np.int32(5))	1.0     对应的矩阵（1行 × 14列 的稀疏矩阵）：
    (np.int32(0), np.int32(6))	1.0     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    (np.int32(0), np.int32(7))	1.0
    (np.int32(0), np.int32(8))	1.0
    (np.int32(0), np.int32(9))	1.0
    (np.int32(0), np.int32(10))	1.0
    (np.int32(0), np.int32(11))	1.0
    (np.int32(0), np.int32(12))	1.0
    (np.int32(0), np.int32(13))	1.0
    ②np.array(np_label):  
    [0.]
    
    _csr.pkl文件内容的大致格式：（features：scipy.sparse._csr.csr_matrix,labels:numpy.ndarray），示例：
    features.shape:(5043, 10591)          labels.shape:(5043, 1)
    """
    pickle.dump((csr_matrix(np.array(np_data)), np.array(np_label)), \
                open(os.path.join(output_path, partition + "_csr.pkl"),'wb'))  # 'wb' = 以二进制写入模式打开，修改了文件路径的拼接情况


"""
Set <input_path> to where the raw MIMIC CSV files are located.
Set <output_path> to where you want the output files to be.
"""


def main():
    # 让程序能够接收命令行参数，允许用户在运行脚本时指定输入和输出路径，初始化一个命令行参数解析器，用于解析用户传入的参数
    parser = argparse.ArgumentParser(description='File path')
    # '--input_path'：参数的名称（用户在命令行使用）    default='.'：默认值是当前目录（如果用户不指定）
    parser.add_argument('--input_path', type=str, default='.', help='input path of original dataset')
    parser.add_argument('--output_path', type=str, default='.', help='output path of processed dataset')
    """
    实际解析用户传入的命令行参数，返回一个包含所有参数值的对象:
    args的内容:
    Namespace(input_path='.', output_path='.')
    """
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    admission_dx_file = input_path + '/ADMISSIONS.csv'
    diagnosis_file = input_path + '/DIAGNOSES_ICD.csv'
    treatment_file = input_path + '/PROCEDURES_ICD.csv'
    lab_file = input_path + '/LABEVENTS.csv'
    encounter_dict = process_patient(admission_dx_file, {})
    encounter_dict = process_diagnosis(diagnosis_file, encounter_dict)
    encounter_dict = process_treatment(treatment_file, encounter_dict)
    patient_ids = np.array([(encounter_dict[key].patient_id + ':'
                    + key) for key in encounter_dict])  # np.array用于创建多维数组（ndarray），更省内存，向量化运算更快
    train_ids, val_ids, test_ids = train_val_test_split(patient_ids)
    mean_std = get_lab_mean_std(lab_file, train_ids)
    encounter_dict = process_lab(lab_file, encounter_dict, mean_std)
    key_list, seqex_list, dx_map, proc_map, lab_map = build_seqex(
        encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=200)
    train_seqex = get_partitions(seqex_list, set(train_ids))
    val_seqex = get_partitions(seqex_list, set(val_ids))
    test_seqex = get_partitions(seqex_list, set(test_ids))
    pickle.dump(dx_map, open(output_path + '/dx_map.p', 'wb'), -1)  # -1 - 协议版本（最高压缩）
    pickle.dump(proc_map, open(output_path + '/proc_map.p', 'wb'), -1)
    pickle.dump(lab_map, open(output_path + '/lab_map.p', 'wb'), -1)
    print("Split done.")

    # SerializeToString() 是 Protocol Buffer (protobuf) 对象的方法，用于将对象序列化为二进制字节串。
    # 因为对象不能直接保存或传输
    with tf.io.TFRecordWriter(output_path + '/train.tfrecord') as writer:
        for seqex in train_seqex:
            writer.write(seqex.SerializeToString())

    with tf.io.TFRecordWriter(output_path + '/validation.tfrecord') as writer:
        for seqex in val_seqex:
            writer.write(seqex.SerializeToString())

    with tf.io.TFRecordWriter(output_path + '/test.tfrecord') as writer:
        for seqex in test_seqex:
            writer.write(seqex.SerializeToString())

    for partition in ['train', 'validation', 'test']:
        tf2csr(output_path, partition, [dx_map, proc_map, lab_map])
    print('done')


if __name__ == '__main__':
    main()
