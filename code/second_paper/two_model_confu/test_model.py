import os

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from fileternet.filter_config import Configs
import numpy as np
from device import device
from two_model_confu import model


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

filter_model=model.Model()
model=filter_model.to(device)

num_epoch=50
#损失函数
criterion = nn.CrossEntropyLoss().to(device)




checkpoint_path = r'C:\Users\WjtDg\Desktop\数据处理2新\ouer\model_timenet.pth'



model.load_state_dict(torch.load(checkpoint_path))
model.eval()  # 设置为评估模式


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


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def confusion_matrix_show(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # 归一化混淆矩阵（按行归一化）
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm / row_sums

    # 设置标签名称
    labels = ['Class 0', 'Class 1', 'Class 2']

    # 创建画布
    plt.figure(figsize=(8, 6), dpi=300)
    ax = sns.heatmap(cm_normalized*100, annot=True, cmap="Oranges", fmt=".2f",  # 柔和蓝绿色
                     xticklabels=labels, yticklabels=labels, linewidths=0.5,
                     linecolor='gray', annot_kws={"size": 14, "weight": "bold"})

    # 设置轴标签样式
    plt.xlabel('Predicted labels', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('True labels', fontsize=14, fontweight='bold', labelpad=10)

    # 设置标题
    # plt.title('Normalized Confusion Matrix (Accuracy)', fontsize=16, fontweight='bold', pad=15)

    # 旋转 x 轴标签，防止重叠
    plt.xticks(rotation=0, fontsize=12, weight='bold')
    plt.yticks(rotation=0, fontsize=12, weight='bold')

    # **关键步骤：调整布局**
    plt.tight_layout()  # 自动调整边距，防止裁剪
    # plt.subplots_adjust(left=0.15)  # 也可以手动调整左边距，数值可以调整
    # 保存图像
    plt.savefig(r'C:\Users\WjtDg\Desktop\heatmap_results00.png', dpi=900, bbox_inches='tight')
    # 显示混淆矩阵
    plt.show()

    # 打印分类报告
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(report)












if __name__ == '__main__':

    test(model, val_loader, criterion)



