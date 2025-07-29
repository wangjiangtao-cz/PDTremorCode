import torch
import torch.nn as nn
from two_model_confu import paifiter_two
######
####依据模型的需要对各个组件进行修改


###
from two_model_confu import paifiter_two
# from two_model_confu.paifilter import paifiter_two_xiugai
from two_model_confu.timenet_xiugai import timenet_two_xiugai2
from device import device

class Configs1:
    def __init__(self):
        # 定义所有的配置参数及其默认值
        self.seq_len = 20
        self.pred_len=3 #分类类别数
        self.enc_in=132
        self.hidden_size=256

class Configs2:
    def __init__(self):
        # 定义所有的配置参数及其默认值
        self.task_name = "classification"
        self.seq_len = 20
        self.pred_len = 0
        self.e_layers = 2
        self.enc_in = 132
        self.d_model = 32
        self.embed = "fixed"
        self.freq = "hour"
        self.dropout = 0.1
        self.num_class = 3
        self.top_k=1
        self.d_ff=64
        self.num_kernels=6
conf1=Configs1()
conf2=Configs2()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.filtermodel = paifiter_two.Model(conf1).to(device)
        self.timenetmodel = timenet_two_xiugai2.Model(conf2).to(device)

        # 定义融合后用于分类的全连接层
        self.fc = nn.Sequential(
            nn.Linear(200, 128),  # 200 = 100 (filtermodel) + 100 (timenetmodel)
            nn.ReLU(),
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(128, 3)  # 3 分类
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 获取两个模型的输出
        output1 = self.filtermodel(x_enc, x_mark_enc, None, None)  # (150, 100)###将NOne改成了mask
        output2 = self.timenetmodel(x_enc, x_mark_enc, None, None)  # (150, 100)

        # 拼接两个模型的输出
        fused_output = torch.cat((output1, output2), dim=1)  # (150, 200)

        # 输入全连接层进行分类
        final_output = self.fc(fused_output)  # (150, 3)

        return final_output




