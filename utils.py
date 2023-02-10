import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from itertools import count

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from icecream import ic, argumentToString
@argumentToString.register(np.ndarray)
def _(t): return f'{tuple(t.shape)} of type {t.dtype} (NumPy) in value range [{t.min().item():.3f}, {t.max().item():.3f}]'
@argumentToString.register(torch.Tensor)
def _(t): return f'{tuple(t.shape)} of type {t.dtype} ({t.device.type}) in value range [{t.min().item():.3f}, {t.max().item():.3f}]'
ic.configureOutput(prefix='')

class PrintLayer(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        ic(x.shape)
        return x

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


def split_squeeze(t, bs, f):
    n = t.size(0)
    X, Y, Z = t[None, None].expand(bs, f, n, 3).split(1, dim=-1)
    BS = torch.arange(bs)[:, None, None].expand(bs, f, n)
    F =  torch.arange(f)[None, :, None].expand(bs, f, n)
    return (BS, F, X.squeeze(-1), Y.squeeze(-1), Z.squeeze(-1))

def feature_std(t, reduce_dim=None, feature_dim=-1):
    ''' Computes standard deviation of feature distances to their mean

    Args:
        t (torch.Tensor): The tensor to compute the feature standard deviation of
        reduce_dim (int or tuple or list, optional): Dimensions to reduce over. Defaults to all except `feature_dim`.
        feature_dim (int, optional): Dimension of the feature to compute distances to their mean over. Defaults to -1.

    Returns:
        torch.Tensor: Standard deviation of feature distances. Has same shape as input without `reduce_dim` and `feature_dim` dimensions
    '''
    if reduce_dim is None: 
        reduce_dim = list(range(t.ndim))
        del reduce_dim[feature_dim]
    else:
        if isinstance(reduce_dim, (tuple, list)):
            assert feature_dim not in reduce_dim and (feature_dim + t.ndim) not in reduce_dim
        else:
            assert reduce_dim != feature_dim
    mean = t.mean(dim=reduce_dim)
    for d in sorted(reduce_dim): mean.unsqueeze_(d)
    return torch.linalg.vector_norm(t - mean, dim=feature_dim).float().mean(dim=reduce_dim)

def norm_minmax(t):
    mi, ma = t.min(), t.max()
    return (t - mi) / (ma - mi)

def norm_mean_std(t, mu=0, std=1):
    return (t.float() - t.float().mean()) * std / t.float().std() + mu

def split_squeeze3d(t):
    n = t.size(0)
    X, Y, Z = t.split(1, dim=-1)
    return (X.squeeze(-1), Y.squeeze(-1), Z.squeeze(-1))

def similarity_matrix(vectors, mode):
    ''' Computes similarity matrix between F-vectors

    Args:
        vectors (torch.Tensor): vectors of shape (N, F)
    '''
    if mode == 'cosine':
        sim = torch.einsum('nf,mf->nm', [vectors, vectors])
    elif mode == 'l2':
        sim = torch.cdist(vectors.unsqueeze(0), vectors.unsqueeze(0)).squeeze(0)
        sim /= torch.nan_to_num(sim).max()
    else:
        raise Exception(f'nope for the mode: {mode}')
    return sim

def plot_similarity_matrix(sim, labels, mode):
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    cax = ax.matshow(sim, interpolation='nearest')
    ax.grid(True)
    ax.set_title(f'Similarity ({mode})')
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(cax, ticks=[-1.0, -.8, -.6, -.4, -.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return fig

def plot_confusion_matrix(confusion, labels):
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(confusion.size(0)):
        for j in range(confusion.size(1)):
            ax.text(x=j, y=i, s=confusion[i,j].item(), va='center', ha='center', size='large')
    ax.set_title(f'Confusion Matrix')
    ax.set_xlabel('Targets', fontsize=16)
    ax.set_ylabel('Predictions', fontsize=16)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    return fig

def cluster_kmeans(features, num_classes):
    ''' Cluster `features` using KMeans algorithm

    Args:
        features (torch.Tensor): Tensor of shape (BS, F, D, H, W)
        num_classes (int): Number of classes/clusters
    '''
    km = KMeans(n_clusters=num_classes)
    km.fit(features.detach().permute(0, 2, 3, 4, 1).reshape(-1, features.size(1)).cpu())
    return km.labels_.reshape(*features.shape[-3:])

def project_pca(features, n_dim=3):
    '''Applies PCA to `features and returns the `n_dim` most relevant dimensions

    Args:
        features (torch.Tensor): Tensor of shape (BS, F, D, H, W)
        n_dim (int): Number of dimensions to return after PCA
    '''
    pca = PCA(n_components=n_dim)
    comp = pca.fit_transform(features.detach().permute(0, 2, 3, 4, 1).reshape(-1, features.size(1)).cpu())
    return norm_minmax(comp).reshape(*features.shape[-3:], n_dim)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Transforms
def transform_paws_crops(crops, noise_std=0.05, flip=True, permute=True):
    if noise_std > 0.0:
        anchors = crops + torch.randn_like(crops) * noise_std
        positiv = crops + torch.randn_like(crops) * noise_std
    else:
        anchors, positiv = crops, crops.clone()
    if permute:
        permutations = [  # All Possible permutations for volume
            (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
        ]
        idx_anc, idx_pos = torch.randint(0, len(permutations), (2,)).tolist()
        pre_shap = tuple(range(crops.ndim - 3))
        post_shap_anc = tuple(map(lambda d: d + len(pre_shap), permutations[idx_anc]))
        post_shap_pos = tuple(map(lambda d: d + len(pre_shap), permutations[idx_pos]))
        anchors = anchors.permute(*pre_shap, *post_shap_anc)
        positiv = anchors.permute(*pre_shap, *post_shap_pos)
    if flip:
        flips = torch.rand(6) < 0.5
        anchors = anchors.flip(dims=[i for i,f in enumerate(flips.tolist()[:3]) if f])
        positiv = positiv.flip(dims=[i for i,f in enumerate(flips.tolist()[3:]) if f])

    return torch.cat([anchors, positiv], dim=0)

class CenterCrop(nn.Module):
    def __init__(self, ks=3):
        super().__init__()
        self.pad = ks // 2

    def forward(self, x):
        i = self.pad
        out = x[..., i:-i, i:-i, i:-i]
        return out
