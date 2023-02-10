import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from itertools import count

from utils import *

class PrintLayer(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        ic(x.shape)
        return x

class CenterCrop(nn.Module):
    def __init__(self, ks=3):
        super().__init__()
        self.pad = ks // 2

    def forward(self, x):
        i = self.pad
        out = x[..., i:-i, i:-i, i:-i]
        return out

def conv_layer(n_in, n_out, Norm, Act, ks=3, suffix=''):
    return nn.Sequential(OrderedDict([
        (f'conv{suffix}', nn.Conv3d(n_in, n_out, kernel_size=ks, stride=1, padding=0)),
        (f'norm{suffix}', Norm(n_out // 4, n_out)),
        (f'act{suffix}', Act(inplace=True))
    ]))

def create_cnn(in_dim, n_features=[8, 16, 32], n_linear=[32], Act=nn.Mish, Norm=nn.GroupNorm):
    assert isinstance(n_features, list) and len(n_features) > 0
    assert isinstance(n_linear,   list) and len(n_linear) > 0
    feats = [in_dim] + n_features
    lins = [n_features[-1]] + n_linear if len(n_linear) > 0 else []
    layers = [conv_layer(n_in, n_out, Norm=Norm, Act=Act, suffix=i)
        for i, n_in, n_out in zip(count(1), feats, feats[1:])]
    lin_layers = [conv_layer(n_in, n_out, Norm=Norm, Act=Act, ks=1, suffix=i)
        for i, n_in, n_out in zip(count(1), lins, lins[1:])]
    last_in = n_linear[-2] if len(n_linear) > 1 else n_features[-1]
    last = nn.Conv3d(last_in, n_linear[-1], kernel_size=1, stride=1, padding=0)
    return nn.Sequential(OrderedDict([
        ('convs', nn.Sequential(*layers)), 
        ('linears', nn.Sequential(*lin_layers)), 
        ('last', last)
        ]))

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, n_features=[8, 16, 32], n_linear=[32],
        Act=nn.Mish, Norm=nn.GroupNorm, residual=False):
        super().__init__()
        assert isinstance(n_features, list) and len(n_features) > 0
        assert isinstance(n_linear, list) and len(n_linear) > 0
        self.residual = residual
        feats = [in_dim] + n_features
        if residual:
            lins = [n_features[-1] + in_dim] + n_linear if len(n_linear) > 0 else []
            last_in = n_linear[-2] + in_dim if len(n_linear) > 1 else n_features[-1] + in_dim
            self.crop = CenterCrop(ks=len(n_features)*2)
        else:
            lins = [n_features[-1]] + n_linear if len(n_linear) > 0 else []
            last_in = n_linear[-2] if len(n_linear) > 1 else n_features[-1]

        convs = [conv_layer(n_in, n_out, Norm=Norm, Act=Act, suffix=i)
            for i, n_in, n_out in zip(count(1), feats, feats[1:])]
        lins = [conv_layer(n_in, n_out, Norm=Norm, Act=Act, ks=1, suffix=i)
            for i, n_in, n_out in zip(count(1), lins, lins[1:])]
        self.convs = nn.Sequential(*convs)
        self.lins = nn.Sequential(*lins)
        self.last = nn.Conv3d(last_in, n_linear[-1], kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.residual:
            skip = self.crop(x)
            x = self.convs(x)
            x = self.lins(torch.cat([skip, x], dim=1))
            return self.last(torch.cat([skip, x], dim=1))
        else:
            return self.last(self.lins(self.convs(x)))


class PAWSNet(nn.Module):
    def __init__(self, in_dim, conv_layers, hidden_sz, out_classes, head_bottleneck=4):
        super().__init__()
        self.encoder = create_cnn(in_dim=in_dim, n_features=conv_layers, n_linear=[conv_layers[-1]])
        NF = conv_layers[-1]
        NH = hidden_sz
        self.head = nn.Sequential(OrderedDict([
            ('bn0',   nn.BatchNorm1d(NF)),
            ('fc1',   nn.Linear(NF, NH//head_bottleneck)),
            ('bn1',   nn.BatchNorm1d(NH//head_bottleneck)),
            ('mish1', nn.Mish(True)),
            ('fc2',   nn.Linear(NH//head_bottleneck, NF))
        ]))
        self.proj = nn.Sequential(OrderedDict([
            ('bn0',   nn.BatchNorm1d(NF)),
            ('fc1',   nn.Linear(NF, NH)),
            ('bn1',   nn.BatchNorm1d(NH)),
            ('mish1', nn.Mish(True)),
            ('fc2',   nn.Linear(NH, NH)),
            ('bn2',   nn.BatchNorm1d(NH)),
            ('mish2', nn.Mish(True)),
            ('fc3',   nn.Linear(NH, NF))
        ]))
        self.predict = nn.Sequential(OrderedDict([
            ('bn0',   nn.BatchNorm1d(NF)),
            ('fc1',   nn.Linear(NF, NH)),
            ('bn1',   nn.BatchNorm1d(NH)),
            ('mish1', nn.Mish(True)),
            ('fc2',   nn.Linear(NH, out_classes))
        ]))

    def forward(self, x, return_class_pred=False):
        z = self.encoder(x).squeeze() # BS, F, D,H,W -> BS, F
        feat = self.proj(z)
        pred = self.head(feat)
        if return_class_pred:
            clas = self.predict(z.detach())
            return feat, pred, clas
        else:
            return feat, pred

    def forward_fullvol(self, x):
        z = self.encoder(x).permute(0, 2,3,4, 1)
        shap = z.shape # BS, D,H,W, F
        clas = self.predict(z.reshape(-1, z.size(-1)))
        return clas.view(*shap[:4], -1).permute(0, 4, 1,2,3).contiguous()
