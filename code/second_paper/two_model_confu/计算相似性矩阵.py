import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from device import device
from two_model_confu import model
from two_model_confu import 测试用model
from load_data2 import my_val_data
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import scipy.ndimage

# 设置随机种子，保证实验可复现
np.random.seed(2)
torch.manual_seed(2)

# 设定 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 读取验证集数据
_, val_loader = my_val_data()

# 加载不含该模块的模型
model_without = 测试用model.Model().to(device)
model_without.load_state_dict(
    torch.load(r'C:\Users\WjtDg\Desktop\second_paper\checkpoint\model_timenet.pth'))
model_without.eval()

# 加载含该模块的模型
model_with = model.Model().to(device)
model_with.load_state_dict(torch.load(r'C:\Users\WjtDg\Desktop\数据处理2新\ouer\model_timenet.pth'))
model_with.eval()

# 计算模型预测值
def get_predictions(model, data_loader):
    all_preds = []
    with torch.no_grad():
        for batch_x, _, padding_mask in tqdm(data_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            padding_mask = padding_mask.float().to(device)

            outputs = model(batch_x, padding_mask, None, None)
            probs = torch.softmax(outputs, dim=1)  # 获取概率分布
            all_preds.append(probs.cpu().numpy())  # 存储整个概率向量

    return np.concatenate(all_preds, axis=0)  # 变成 (样本数, 3)


# 获取两种模型的预测结果
preds_without = get_predictions(model_without, val_loader)
preds_with = get_predictions(model_with, val_loader)

# 降采样函数
def downsample_matrix(matrix, factor=4):
    """
    对矩阵进行降采样
    :param matrix: 原始矩阵
    :param factor: 降采样因子
    :return: 降采样后的矩阵
    """
    return scipy.ndimage.zoom(matrix, (1/factor, 1/factor), order=1)  # 使用线性插值进行降采样

# 计算时间相似性矩阵（使用 Pearson 相关系数）
def compute_temporal_similarity(preds):
    # 使用 np.corrcoef 计算时间相似性
    return np.corrcoef(preds)

# 计算样本相似性矩阵（使用欧式距离）
def compute_spatial_similarity(preds):
    # 使用欧式距离计算样本相似性
    return squareform(pdist(preds.reshape(-1, 1), metric='euclidean'))

# 计算 STSM 矩阵
stsm_temporal_without = compute_temporal_similarity(preds_without)
stsm_temporal_with = compute_temporal_similarity(preds_with)

stsm_spatial_without = compute_spatial_similarity(preds_without)
stsm_spatial_with = compute_spatial_similarity(preds_with)

# 降采样相似性矩阵
stsm_temporal_without = downsample_matrix(stsm_temporal_without, factor=4)
stsm_temporal_with = downsample_matrix(stsm_temporal_with, factor=4)
stsm_spatial_without = downsample_matrix(stsm_spatial_without, factor=4)
stsm_spatial_with = downsample_matrix(stsm_spatial_with, factor=4)

# 计算归一化
def normalize_matrix(matrix):
    scaler = MinMaxScaler()
    return scaler.fit_transform(matrix)

stsm_temporal_without = normalize_matrix(stsm_temporal_without)
stsm_temporal_with = normalize_matrix(stsm_temporal_with)
stsm_spatial_without = normalize_matrix(stsm_spatial_without)
stsm_spatial_with = normalize_matrix(stsm_spatial_with)

# 计算变化矩阵
delta_temporal = normalize_matrix(stsm_temporal_with - stsm_temporal_without)
delta_spatial = normalize_matrix(stsm_spatial_with - stsm_spatial_without)

# 设定颜色范围
vmin, vmax = 0, 1
diff_vmin, diff_vmax = -0.5, 0.5

# 绘制热力图
# fig, axes = plt.subplots(2, 2, figsize=(14, 9))
# 修改为一行两列
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# 时间相似性
sns.heatmap(stsm_temporal_without, ax=axes[0], cmap="coolwarm", vmin=vmin, vmax=vmax, alpha=0.5)
axes[0].set_title("Temporal Similarity (Without Module)")
axes[0].set_xticks([])  # 隐藏 X 轴刻度
axes[0].set_yticks([])  # 隐藏 Y 轴刻度

sns.heatmap(stsm_temporal_with, ax=axes[1], cmap="coolwarm", vmin=vmin, vmax=vmax, alpha=0.5)
axes[1].set_title("Temporal Similarity (With Module)")
axes[1].set_xticks([])  # 隐藏 X 轴刻度
axes[1].set_yticks([])  # 隐藏 Y 轴刻度

# 保存图像
plt.savefig(r'C:\Users\WjtDg\Desktop\heatmap_results1.png', dpi=900, bbox_inches='tight')

# 显示图像
plt.show()

# # 样本相似性
# sns.heatmap(stsm_spatial_without, ax=axes[1, 0], cmap="coolwarm", vmin=vmin, vmax=vmax,alpha=0.8)
# axes[1, 0].set_title("Spatial Similarity (Without Module)")
# axes[1, 0].set_xticks([])  # 隐藏 X 轴刻度
# axes[1, 0].set_yticks([])  # 隐藏 Y 轴刻度
#
# sns.heatmap(stsm_spatial_with, ax=axes[1, 1], cmap="coolwarm", vmin=vmin, vmax=vmax,alpha=0.8)
# axes[1, 1].set_title("Spatial Similarity (With Module)")
# axes[1, 1].set_xticks([])  # 隐藏 X 轴刻度
# axes[1, 1].set_yticks([])  # 隐藏 Y 轴刻度
# # 变化矩阵
# sns.heatmap(delta_temporal, ax=axes[2, 0], cmap="RdBu_r", center=0, vmin=diff_vmin, vmax=diff_vmax)
# axes[2, 0].set_title("Δ Temporal Similarity (With - Without)")
# axes[2, 0].set_xticks([])  # 隐藏 X 轴刻度
# axes[2, 0].set_yticks([])  # 隐藏 Y 轴刻度
# sns.heatmap(delta_spatial, ax=axes[2, 1], cmap="RdBu_r", center=0, vmin=diff_vmin, vmax=diff_vmax)
# axes[2, 1].set_title("Δ Spatial Similarity (With - Without)")
# axes[2, 1].set_xticks([])  # 隐藏 X 轴刻度
# axes[2, 1].set_yticks([])  # 隐藏 Y 轴刻度
# plt.tight_layout()

# Adjust the layout to remove excess white space
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# Save the figure with 900 dpi and without extra whitespace
# plt.savefig(r'C:\Users\WjtDg\Desktop\heatmap_results1.png', dpi=900, bbox_inches='tight')
#
# plt.show()
