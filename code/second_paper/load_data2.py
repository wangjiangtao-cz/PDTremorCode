import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
from collate_fn import collate_fn



train=np.load(r'C:\Users\WjtDg\Desktop\数据处理2新\ouer\data0.npy')
train_lable=np.load(r'C:\Users\WjtDg\Desktop\数据处理2新\ouer\lable0.npy')

val=np.load(r'C:\Users\WjtDg\Desktop\数据处理2新\ouer\data1.npy')
vallable=np.load(r'C:\Users\WjtDg\Desktop\数据处理2新\ouer\lable1.npy')

train= train[:, :, :132]

val = val[:, :, :132]

##################################


##############################
def my_train_data():

    X_train_tensor = torch.tensor(train, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_lable.flatten(), dtype=torch.long)  # 使用 long 类型标签进行多分类
    X_val_tensor = torch.tensor(val, dtype=torch.float32)
    y_val_tensor = torch.tensor(vallable.flatten(), dtype=torch.long)
    print(X_train_tensor.shape,y_train_tensor.shape)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=150, num_workers=0, drop_last=False,
                              shuffle=True,
                              collate_fn=lambda x: collate_fn(x, max_len=20))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=150)
    return  train_dataset,train_loader

def my_val_data():

    X_train_tensor = torch.tensor(train, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_lable, dtype=torch.long)  # 使用 long 类型标签进行多分类
    X_val_tensor = torch.tensor(val, dtype=torch.float32)
    y_val_tensor = torch.tensor(vallable.flatten(), dtype=torch.long)
    print(X_train_tensor.shape,y_train_tensor.shape)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=150)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=150,num_workers=0,drop_last=False,collate_fn=lambda x: collate_fn(x, max_len=20))

    return  val_dataset,val_loader



# def my_test_data():
#
#     X_train_tensor = torch.tensor(xtrain, dtype=torch.float32)
#     y_train_tensor = torch.tensor(ytrain, dtype=torch.long)  # 使用 long 类型标签进行多分类
#     X_val_tensor = torch.tensor(xtest, dtype=torch.float32)
#     y_val_tensor = torch.tensor(ytest.flatten(), dtype=torch.long)
#     print(X_train_tensor.shape,y_train_tensor.shape)
#
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=150)
#     val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
#     val_loader = DataLoader(val_dataset, batch_size=150,num_workers=0,drop_last=False,collate_fn=lambda x: collate_fn(x, max_len=20))
#
#     return  val_dataset,val_loader



