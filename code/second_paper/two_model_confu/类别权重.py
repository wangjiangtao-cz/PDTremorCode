# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn.functional as F
#
#
# def visualize_attention(x_enc, attention_weights, sample_idx=0):
#     """
#     可视化输入数据和注意力权重，并展示加权后的输入数据
#
#     Parameters:
#     - x_enc: 输入数据，形状为 (batch_size, window_size, feature_dim)，例如 (150, 20, 132)
#     - attention_weights: 注意力权重，形状为 (batch_size, feature_dim)，例如 (150, 132)
#     - sample_idx: 要可视化的样本索引，默认为 0
#
#     Returns:
#     - None
#     """
#     # 获取选择样本的输入数据和注意力权重
#     sample_input = x_enc[sample_idx, :, :].cpu().numpy()  # 获取第 sample_idx 样本的输入数据，形状 (20, 132)
#     attention_weights=attention_weights.detach().cpu().numpy()
#     # 将注意力权重应用到输入数据中
#     weighted_input = sample_input * attention_weights[sample_idx]# (20, 132)
#
#     # 可视化原始输入数据
#     plt.figure(figsize=(10, 6))
#     plt.subplot(3, 1, 1)
#     plt.plot(sample_input)
#     plt.title(f'Original Input (Sample {sample_idx})')
#     plt.xlabel('Time Step')
#     plt.ylabel('Feature Value')
#
#     # 可视化注意力权重
#     plt.subplot(3, 1, 2)
#     plt.plot(attention_weights)
#     plt.title(f'Attention Weights (Sample {sample_idx})')
#     plt.xlabel('Feature Index')
#     plt.ylabel('Attention Weight')
#
#     # 可视化加权后的输入数据
#     plt.subplot(3, 1, 3)
#     plt.plot(weighted_input)
#     plt.title(f'Weighted Input (Sample {sample_idx}) After Attention')
#     plt.xlabel('Time Step')
#     plt.ylabel('Weighted Feature Value')
#
#     plt.tight_layout()
#     plt.show()
#
#
#
# # 示例用法：
# # 假设你有 x_enc 和 attention_weights
# # 你可以通过调用 visualize_attention 函数来查看注意力的效果
# # visualize_attention(x_enc, attention_weights, sample_idx=0)
import torch

# 检查GPU是否可用
if torch.cuda.is_available():
    print("GPU可用！")
else:
    print("GPU不可用，将使用CPU进行计算。")