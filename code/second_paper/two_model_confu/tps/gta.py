import torch.nn as nn
import torch

class LearnedPositionalEmbedding2(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(device='cuda')
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe)
        torch.nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        return self.pe[:, :x.size(1)]




class BERTEmbedding2(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding2(d_model=input_dim, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence) + sequence
        return self.dropout(x)






class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, kernel_size=8, padding=3):
        super(BasicBlock1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, padding=padding)
        self.bn1 = norm_layer(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out



class Se1Block_dropB(nn.Module):

    def __init__(self, channel,length, reduction=16):
        super(Se1Block_dropB, self).__init__()

        self.fc0 = nn.Sequential(nn.Linear(channel,1,bias=False),
                                 #nn.ReLU(inplace=True)
                                 nn.Sigmoid()
                                 )
        self.fc = nn.Sequential(
            nn.Linear(length, length // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(length // reduction, length, bias=False),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x,mask):
        b, l, _ = x.size()
        y = self.fc0(x).view(b, l).masked_fill(mask[:,:,0]==0,1e-9)
        y = self.drop(y)
        y = self.fc(y).view(b, l, 1)
        # print(x.shape)
        # print(mask.shape)
        # print(y.shape)
        # print(mask[:,:,0].shape)
        return x.masked_fill((mask[:,:,0:1]==0).expand_as(x),1e-9) * y.expand_as(x)




class FTABlockB(nn.Module):

    def __init__(self, channel=64,length=24, reduction=16):
        super(FTABlockB, self).__init__()
        # self.SE=Se1Block(channel, reduction=reduction)
        self.SE = Se1Block_dropB(channel=channel,length=length, reduction=reduction)

    def _addRI(self, input):
        input = torch.stack([input, torch.zeros_like(input)], -1)
        return input

    def forward(self, x,mask):
        x = x.permute(0, 2, 1)
        mask = mask.permute(0,2,1)
        x = self.SE(x,mask)
        x = x.permute(0, 2, 1)
        return x