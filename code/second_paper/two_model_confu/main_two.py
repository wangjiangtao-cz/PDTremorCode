import os

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from fileternet.filter_config import Configs
import numpy as np
from device import device
# from two_model_confu import model
from two_model_confu import 测试用model

from load_data2 import my_val_data,my_train_data


from torch import nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import copy
from Nadam import Nadam
from earlystoping import EarlyStopping
import seaborn as sns
np.random.seed(2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##输出模型参数
configs = Configs()
configs.display()

train_dataset,train_loader = my_train_data()
val_dataset,val_loader = my_val_data()

# filter_model=model.Model()
filter_model=测试用model.Model()

filter_model=filter_model.to(device)



num_epoch=30
#损失函数
criterion = nn.CrossEntropyLoss().to(device)

def train(model,train_loader,val_loader,criterion):
    best_val_acc = 0.0  # 初始化最佳验证集准确率
    best_model = None  # 初始化最佳模型对象
    lr = 0.001
    for i in (np.arange(1, 2) * 5):
        optimizer = Nadam(model.parameters(), lr=lr)

        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=lr * 0.001)

        early_stopping = EarlyStopping(r'C:\Users\WjtDg\Desktop\second_paper\checkpoint')
        for epoch in range(num_epoch):
            model.train()
            total_loss = 0.0
            correct_train = 0
            total_train = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epoch}', leave=False)

            for batch_x, batch_y, padding_mask in progress_bar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                padding_mask = padding_mask.float().to(device)

                optimizer.zero_grad()

                outputs = model(batch_x, padding_mask, None, None)

                loss = criterion(outputs, batch_y.long().squeeze(-1))

                loss.backward()  # 反向传播

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)

                optimizer.step()  # 更新权重

                total_loss += loss.item() * len(batch_x)

                # 计算训练集准确率
                _, predicted = torch.max(nn.functional.softmax(outputs, dim=1), 1)  # 使用 argmax 得到预测结果

                correct_train += (predicted == batch_y).sum().item()
                total_train += batch_y.size(0)
                train_acc = correct_train / total_train

                progress_bar.set_postfix({'Loss': total_loss / total_train, 'Train Acc': train_acc})

            val_loss, val_acc = calculate_accuracy1(model, criterion, val_loader)

            print(
                f'Epoch [{epoch + 1}/{num_epoch}], Loss: {total_loss / len(train_dataset):.4f}, Train Acc: {train_acc:.4f}, ,Val Loss: {val_loss:.4f} ,Val Acc: {val_acc:.4f}')

            scheduler.step()

            # 检查当前验证集准确率是否优于最佳准确率，并保存当前模型作为最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # 创建一个新的模型对象并加载当前模型的状态字典
                best_model = copy.deepcopy(model)

                #######################################
                # 保存模型
                model_path = r'C:\Users\WjtDg\Desktop\second_paper\checkpoint\model_timenet.pth'
                torch.save(best_model.state_dict(), model_path)
                print(f"Model saved to {model_path} with validation accuracy {val_acc}")

                #######################################

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                lr = lr / 2
                break  # 跳出迭代，结束训练

    return best_model


def calculate_accuracy1(model, criterion, loader):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y, padding_mask in loader:
            # batch_y=batch_y.squeeze()
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            padding_mask = padding_mask.float().to(device)

            output = model(batch_X, padding_mask, None, None)  # torch.Size([348, 18, 100]) torch.Size([348, 1])

            loss = criterion(output, batch_y.long().squeeze(-1))
            total_loss += loss.item() * len(batch_X)
            predicted = torch.argmax(nn.functional.softmax(output, dim=1), 1)  # 使用 argmax 得到预测结果
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    return total_loss / total, correct / total












def test(model, test_loader, criterion):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    progress_bar = tqdm(test_loader, desc='Testing', leave=False)

    with torch.no_grad():  # 在测试阶段，不需要计算梯度
        for batch_x, batch_y, padding_mask in progress_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            padding_mask = padding_mask.float().to(device)

            outputs = model(batch_x, padding_mask, None, None)

            loss = criterion(outputs, batch_y.long().squeeze(-1))
            total_loss += loss.item() * len(batch_x)

            _, predicted = torch.max(nn.functional.softmax(outputs, dim=1), 1)  # 使用 argmax 得到预测结果

            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())


            progress_bar.set_postfix({'Loss': total_loss / total, 'Test Acc': correct / total})
    print(y_true)
    print(y_pred)
    avg_loss = total_loss / total
    avg_acc = correct / total

    print(f'Test Loss: {avg_loss:.4f}, Test Acc: {avg_acc:.4f}')

    confusion_matrix_show(y_true, y_pred)


def confusion_matrix_show(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # 计算每一行的总和
    row_sums = cm.sum(axis=1, keepdims=True)

    # 计算准确率（归一化混淆矩阵的每一行）
    cm_normalized = cm / row_sums

    # 设置标签名称
    labels = ['Class 0', 'Class 1', 'Class 2']

    # 绘制归一化后的混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2%', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix (Accuracy)')
    plt.show()

    # 打印 classification_report
    report = classification_report(y_true, y_pred, target_names=labels,digits=4)
    print(report)



if __name__ == '__main__':
    model1 = train(filter_model,train_loader,val_loader,criterion )



    print('Validation !!')

    # test_dataset, test_loader = my_val_data()
    test(model1, val_loader, criterion)














