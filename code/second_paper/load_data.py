import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
from collate_fn import collate_fn



X=np.load(r'C:\Users\WjtDg\Desktop\数据处理2\data.npy')
y=np.load(r'C:\Users\WjtDg\Desktop\数据处理2\lable.npy')
xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.1, random_state=42)



# xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)
# xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.05, random_state=42)

def my_train_data():

    X_train_tensor = torch.tensor(xtrain, dtype=torch.float32)
    y_train_tensor = torch.tensor(ytrain.flatten(), dtype=torch.long)  # 使用 long 类型标签进行多分类
    X_val_tensor = torch.tensor(xval, dtype=torch.float32)
    y_val_tensor = torch.tensor(yval.flatten(), dtype=torch.long)
    print(X_train_tensor.shape,y_train_tensor.shape)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=150, num_workers=0, drop_last=False,
                              shuffle=True,
                              collate_fn=lambda x: collate_fn(x, max_len=20))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=150)
    return  train_dataset,train_loader

def my_val_data():

    X_train_tensor = torch.tensor(xtrain, dtype=torch.float32)
    y_train_tensor = torch.tensor(ytrain, dtype=torch.long)  # 使用 long 类型标签进行多分类
    X_val_tensor = torch.tensor(xval, dtype=torch.float32)
    y_val_tensor = torch.tensor(yval.flatten(), dtype=torch.long)
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



