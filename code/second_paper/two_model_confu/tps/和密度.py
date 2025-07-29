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
from scipy.stats import gaussian_kde

from models import TimesNet
from config import Configs
##输出模型参数
configs = Configs()
# 设置随机种子，保证实验可复现
np.random.seed(2)
torch.manual_seed(2)

# 设定 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 读取验证集数据
_, val_loader = my_val_data()

# 加载不含该模块的模型
model_without = TimesNet.Model(configs).to(device)
model_without.load_state_dict(
    torch.load(r'C:\Users\WjtDg\Desktop\second_paper\checkpoint\jixian.pth'))
model_without.eval()

from two_model_confu.timenet_xiugai import timenet_two_xiugai2



# 加载含该模块的模型
model_with = timenet_two_xiugai2.Model(configs).to(device)
model_with.load_state_dict(torch.load(r'C:\Users\WjtDg\Desktop\second_paper\checkpoint\zhuyili.pth'))
model_with.eval()


# 计算分类结果
def evaluate_model(model, data_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y, padding_mask in tqdm(data_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # 确保 batch_y 也在正确的设备上
            padding_mask = padding_mask.float().to(device)

            outputs = model(batch_x, padding_mask, None, None)
            preds = torch.argmax(outputs, dim=1)  # 取最大概率类别

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())  # 收集真实标签

    return np.array(all_preds), np.array(all_labels)

# 获取两种模型的预测结果和真实标签
preds_without, true_labels = evaluate_model(model_without, val_loader)
preds_with, _ = evaluate_model(model_with, val_loader)  # 真实标签已在上一步获取

# 计算分类偏差
diff_without = preds_without - true_labels
diff_with = preds_with - true_labels


# 计算 KDE
kde_without = gaussian_kde(diff_without)
kde_with = gaussian_kde(diff_with)

# 设置 X 轴范围
x_range = np.linspace(-1.5, 1.5, 1000)

# 计算直方图数据
num_bins = 50
hist_range = (-1.5, 1.5)
counts_without, bin_edges_without = np.histogram(diff_without, bins=num_bins, range=hist_range, density=True)
counts_with, bin_edges_with = np.histogram(diff_with, bins=num_bins, range=hist_range, density=True)

# 计算 bin 中心点
bin_centers = (bin_edges_without[:-1] + bin_edges_without[1:]) / 2
bin_width = bin_edges_without[1] - bin_edges_without[0]

# 绘制 KDE 曲线
plt.figure(figsize=(10, 6))
plt.plot(x_range, kde_without(x_range), label='Without Module', linestyle='--', color='teal')
plt.plot(x_range, kde_with(x_range), label='With Module', linestyle='--', color='coral')

# 绘制小矩形密度分布
# for x, y in zip(bin_centers, counts_without):
#     plt.gca().add_patch(plt.Rectangle((x, 0), width=bin_width, height=y * 0.05, color='teal', alpha=0.3))
#
# for x, y in zip(bin_centers, counts_with):
#     plt.gca().add_patch(plt.Rectangle((x, 0), width=bin_width, height=y * 0.05, color='coral', alpha=0.3))

# 添加图例
plt.legend()
# plt.title('Effect of Module on Classification Distribution')
plt.xlabel('Difference Value')
plt.ylabel('Density')
plt.xlim(hist_range)
# 保存高分辨率图片，去除多余白边
# plt.savefig(r"C:\Users\WjtDg\Desktop\classification_distribution.png", dpi=900, bbox_inches='tight', pad_inches=0)

# 显示图片
plt.show()
