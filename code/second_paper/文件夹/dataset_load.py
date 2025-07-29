from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from collate_fn import collate_fn
from scipy.signal.windows import hann
from scipy.signal.windows import triang
from scipy.signal import boxcar
from scipy.spatial import KDTree
from imblearn.over_sampling import SMOTE
from 噪声 import add_motion_artifacts

# X_train= np.load(r'D:\dataset\data\新建文件夹\fold10\window_features_train.npy')
# y_train = np.load(r'D:\dataset\data\新建文件夹\fold10\window_labels_train.npy')
#
# X_test= np.load(r'D:\dataset\data\新建文件夹\fold10\window_features_test.npy')
# y_test = np.load(r'D:\dataset\data\新建文件夹\fold10\window_labels_test.npy')

##########################################################################################
X_train= np.load(r'C:\Users\WjtDg\Desktop\dataset\old\window_features_train.npy')


y_train= np.load(r'C:\Users\WjtDg\Desktop\dataset\old\window_labels_train.npy')
X_test= np.load(r'C:\Users\WjtDg\Desktop\dataset\old\window_features_test.npy')


y_test= np.load(r'C:\Users\WjtDg\Desktop\dataset\old\window_labels_test.npy')


######################################################################################

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# X_train= np.load(r'C:\Users\WjtDg\Desktop\shujuji\6\xtrain.npy')
#
#
# y_train= np.load(r'C:\Users\WjtDg\Desktop\shujuji\6\ytrain.npy')
# X_test= np.load(r'C:\Users\WjtDg\Desktop\shujuji\6\xval.npy')
#
#
# y_test= np.load(r'C:\Users\WjtDg\Desktop\shujuji\6\yval.npy')





# # 假设 X_train 是你的数据，维度为 (4546, 100, 18)，y_train 是对应的标签向量，维度为 (4546,)
# X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # 将三维数据重塑成二维数组
#
# # # 初始化SMOTE对象并进行过采样
# # sos = SMOTE(random_state=0)
# # X_train_resampled, y_train = sos.fit_resample(X_train_reshaped, y_train)
# #
# # # 将过采样后的数据恢复成原始形状
# # X_train = X_train_resampled.reshape(X_train_resampled.shape[0], X_train.shape[1], X_train.shape[2])
# #
# # # 打印过采样后的数据形状
# # print("过采样后的特征数据形状:", X_train.shape)
# # print("过采样后的标签数据形状:", y_train.shape)



# np.random.seed(200)
# np.random.shuffle(X_train)
# np.random.seed(200)
# np.random.shuffle(y_train)
#
# np.random.seed(200)
# np.random.shuffle(X_test)
# np.random.seed(200)
# np.random.shuffle(y_test)


y_train = np.expand_dims(y_train, axis=1)
y_test=np.expand_dims(y_test, axis=1)
# y_val=np.expand_dims(y_val, axis=1)

##############################################################################

#添加窗函数
def apply_window(data):
    # 创建矩形窗
    window = boxcar(data.shape[1])
    windowed_data = np.apply_along_axis(lambda x: x * window, axis=1, arr=data)
    print("加窗了")
    return windowed_data

# #
# X_train = apply_window(X_train)
# X_test = apply_window(X_test)
##########################################################################

# 打印训练集和测试集的形状
print("Training set shape - Features:", X_train.shape, "Labels:", y_train.shape)
print("Test set shape - Features:", X_test.shape, "Labels:", y_test.shape)
# print("Test set shape - Features:", X_val.shape, "Labels:", y_val.shape)


###################################################
##添加噪声


# X_train = add_motion_artifacts(X_train, artifact_level=0.1, artifact_prob=0.01)


# sampling_rate = 100  # 原始采样频率为100 Hz
# # 应用低通滤波器（去除高频噪声）
# cutoff_freq_low = 15  # 截止频率为15 Hz
# filtered_data_lowpass1 = butter_lowpass_filter(X_train, cutoff_freq_low, sampling_rate)
# # 应用高通滤波器（消除重力影响）
# cutoff_freq_high = 0.2  # 截止频率为0.2 Hz
# X_train = butter_highpass_filter(filtered_data_lowpass1, cutoff_freq_high, sampling_rate)
#
#
# cutoff_freq_low = 15  # 截止频率为15 Hz
# filtered_data_lowpass2 = butter_lowpass_filter(X_test, cutoff_freq_low, sampling_rate)
# # 应用高通滤波器（消除重力影响）
# cutoff_freq_high = 0.2  # 截止频率为0.2 Hz
# X_test = butter_highpass_filter(filtered_data_lowpass2, cutoff_freq_high, sampling_rate)

####################################################


def my_train_data():

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.flatten(), dtype=torch.long)  # 使用 long 类型标签进行多分类
    X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test.flatten(), dtype=torch.long)
    print(X_train_tensor.shape,y_train_tensor.shape)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    #train_loader = DataLoader(train_dataset, batch_size=150,num_workers=0,drop_last=False,sampler=ImbalancedDatasetSampler(train_dataset),collate_fn=lambda x: collate_fn(x, max_len=100))
    train_loader = DataLoader(train_dataset, batch_size=150, num_workers=0, drop_last=False,

                              shuffle=True,
                              collate_fn=lambda x: collate_fn(x, max_len=100))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=150)

    counter = Counter()
    for _, labels,_ in train_loader:
        counter.update(labels.numpy())
    print("++++++++++++++处理后每个类别的占比++++++++++++++++++++++++++++++")
    # 输出每个类别的样本数量
    print(counter)


    counter_val = Counter()
    for _, labels in val_loader:
        counter_val.update(labels.numpy())

    # 输出每个类别的样本数量
    print("++++++++++++++处理后每个类别的占比++++++++++++++++++++++++++++++")
    print(counter_val)


    return  train_dataset,train_loader


def my_val_data():

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 使用 long 类型标签进行多分类
    X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test.flatten(), dtype=torch.long)
    print(X_train_tensor.shape,y_train_tensor.shape)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=150)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=150,num_workers=0,drop_last=False,collate_fn=lambda x: collate_fn(x, max_len=100))



    return  val_dataset,val_loader


# def my_test_data():
#
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 使用 long 类型标签进行多分类
#     X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
#     y_val_tensor = torch.tensor(y_val.flatten(), dtype=torch.long)
#     print(X_train_tensor.shape,y_train_tensor.shape)
#
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=150)
#     val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
#     val_loader = DataLoader(val_dataset, batch_size=150,num_workers=0,drop_last=False,collate_fn=lambda x: collate_fn(x, max_len=100))
#
#
#
#     return  val_dataset,val_loader




