import torch
from torch import nn

from config import Configs
from device import device

from cnntransformer.informer import Model
from cnntransformer.cnn4 import multiple_cnn1D5_level


from dctnet_test import dct_channel_block,dct

# from  dctnet_test2 import dwt_channel_block

import torch.nn.functional as F
configs = Configs(task_name='classification',enc_in=288,d_model=256,embed='fixed',freq='s',dropout=0.1,dec_in=7,factor=1
                  ,output_attention=0,n_heads=8,d_ff=256,activation='activation',e_layers=3,distil=1,d_layers=1,c_out=7
                  ,seq_len=46,num_class=5)
print(configs)


# configs = Configs(task_name='classification',enc_in=288,d_model=256,embed='fixed',freq='s',dropout=0.1,dec_in=7,factor=1
#                   ,output_attention=0,n_heads=8,d_ff=256,activation='activation',e_layers=3,distil=1,d_layers=1,c_out=7
#                   ,seq_len=46,num_class=5)

class PX_TRANS(nn.Module):
    def __init__(self,nb):
        super(PX_TRANS, self).__init__()

        self.cnn=multiple_cnn1D5_level(nb).to(device)
        self.transformer_list =Model(configs)

        # self.a=nn.Sigmoid()

#####################################################
##离散余弦变换
        self.dct_layer = dct_channel_block(288)
        self.dct_norm = nn.LayerNorm([288], eps=1e-6)


# ##小波变换
#         self.dwt_layer=dwt_channel_block(288)



#####################################################

    def forward(self, x,x_mark_enc, x_dec, x_mark_dec):
        x = x.permute(0, 2, 1)

        output=self.cnn(x)

        #####################
#离散余弦变换
        mid= self.dct_layer(output)
        enc_out = output+mid
        output = self.dct_norm(enc_out) #norm 144

# ##小波变换
#
#         mid = self.dwt_layer(output)
#         enc_out = output + mid
#         output = self.dct_norm(enc_out)  # norm 144

        ######################

        output=self.transformer_list(output,x_mark_enc,None,None)


        # output=self.a(output)
        return output




class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x














