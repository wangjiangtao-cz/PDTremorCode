import os
import torch
import numpy as np
import seaborn as sns
import copy

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# 自定义模块
from fileternet.filter_config import Configs
from device import device
from two_model_confu import model
from load_data2 import my_val_data, my_train_data
from Nadam import Nadam
from earlystoping import EarlyStopping

np.random.seed(2)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# **加载模型参数**
configs = Configs()
configs.display()

# **加载数据**
train_dataset, train_loader = my_train_data()
val_dataset, val_loader = my_val_data()

# **初始化模型**
filter_model = model.Model()
model = filter_model.to(device)

# **加载预训练权重**
checkpoint_path = r'C:\Users\WjtDg\Desktop\数据处理2新\ouer\model_timenet.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()  # 设置为评估模式

# **定义损失函数**
criterion = nn.CrossEntropyLoss().to(device)


def test(model, test_loader, criterion):
    """
    测试模型并进行 T-SNE 可视化
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []
    features_list = []  # 存储模型的输出特征

    progress_bar = tqdm(test_loader, desc='Testing', leave=False)

    with torch.no_grad():
        for batch_x, batch_y, padding_mask in progress_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            padding_mask = padding_mask.float().to(device)

            # **获取模型输出**
            outputs = model(batch_x, padding_mask, None, None)

            loss = criterion(outputs, batch_y.long().squeeze(-1))
            total_loss += loss.item() * len(batch_x)

            # **获取预测类别**
            _, predicted = torch.max(nn.functional.softmax(outputs, dim=1), 1)

            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # **存储 logits（模型的最后一层特征）**
            features_list.append(outputs.cpu().numpy())

            progress_bar.set_postfix({'Loss': total_loss / total, 'Test Acc': correct / total})

    # **拼接所有特征**
    features = np.concatenate(features_list, axis=0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Acc: {avg_acc:.4f}')

    # **绘制混淆矩阵**
    # confusion_matrix_show(y_true, y_pred)

    # **T-SNE 可视化**
    tsne_visualization(features, y_true)





def tsne_visualization(features, labels):
    """
    使用 T-SNE 可视化模型分类效果
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)  # 2D 降维
    tsne_result = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.7, s=25)

    # 创建图例
    handles, labels_legend = scatter.legend_elements()
    plt.legend(handles, labels_legend, title="Class", loc='upper right')

    # 去除颜色条（如之前的修改）
    # plt.colorbar(scatter, ticks=range(len(set(labels))))

    # 取消 X 轴和 Y 轴的刻度
    plt.xticks([])  # 隐藏 X 轴刻度
    plt.yticks([])  # 隐藏 Y 轴刻度

    # 去除多余的空白区域
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    # 保存图像为 900 DPI
    plt.savefig(r'C:\Users\WjtDg\Desktop\第二篇对比实验\tsne_visualization.png', dpi=900)


    plt.show()



if __name__ == '__main__':
    test(model, val_loader, criterion)
