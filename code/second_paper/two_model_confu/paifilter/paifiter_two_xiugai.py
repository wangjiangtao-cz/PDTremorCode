import torch
import torch.nn as nn
from fileternet.layers.RevIN import RevIN

from two_model_confu.tps.gta import BERTEmbedding2
from two_model_confu.tps.gta import BasicBlock1D
from two_model_confu.tps.gta import FTABlockB





class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size

        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len)  ##
        )

        self.projection = nn.Linear(
            128 * configs.seq_len, 100)



        #+++++++++++++++++++++++++++++++++++++++++++++++++++

        self.gta=Classifier_FCN_FTA_B(input_shape=132,D=1, length=20,ffh=16)


        #++++++++++++++++++++++++++++++++++++++++++++++++++


    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        x = x.permute(0, 2, 1)

        x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D

        # 150 162 20
        x = self.fc(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #(150,20,132)
        x=self.gta(x.permute(0, 2, 1),x_mark_enc.unsqueeze(-1).permute(0, 2, 1))







        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        output = x.reshape(x.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)

        return output




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Classifier_FCN_FTA_B(nn.Module):

    def __init__(self, input_shape, D=1, length=20,ffh=16):
        super(Classifier_FCN_FTA_B, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlockB(channel=int(128 / D),length=length,reduction=ffh)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlockB(channel=int(256 / D),length=length,reduction=ffh)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlockB(channel=int(128 / D),length=length,reduction=ffh)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x,mask):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        #(150,20,132)
        x = self.conv1(x)
        x = self.FTA1(x,mask)
        x = self.conv2(x)
        x = self.FTA2(x,mask)
        x = self.conv3(x)
        x = self.FTA3(x,mask)
        # x = self.AVG(x)
        return x










































