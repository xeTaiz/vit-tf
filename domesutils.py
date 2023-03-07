import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, os

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from icecream import ic, argumentToString
@argumentToString.register(np.ndarray)
def _(t):
    contig = f'{"C" if t.flags["C_CONTIGUOUS"]} {"F" if t.flags["F_CONTIGUOUS"]}'.strip()
    if len(contig) == 0: contig = "NOT"
    return f'{tuple(t.shape)} of type {t.dtype} (NumPy) in value range [{t.min().item():.3f}, {t.max().item():.3f}] ({contig} contiguous)'

@argumentToString.register(torch.Tensor)
def _(t):
    if t.is_contiguous():
        contig = '(C contiguous)'
    else:
        contig = '(NOT contiguous!!)'
    return f'{tuple(t.shape)} of type {t.dtype} ({t.device.type}) in value range [{t.min().item():.3f}, {t.max().item():.3f}] {contig}'
ic.configureOutput(prefix='')

def setup_seed_and_debug(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        torch.autograd.set_detect_anomaly(True)
        ic.configureOutput(includeContext=True)

def parse_basics(parser):
    parser.add_argument('--data', type=str, required=True, help='Path to Data with {vol, mask, labels} keys in .pt file')
    parser.add_argument('--label-percentage', type=float, default=0.1, help='Percentage of labels to use for optimization')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of optimization steps')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate for optimization')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--lr-schedule', type=str, default='onecycle', help='Learning rate schedule')
    parser.add_argument('--wandb-tags', type=str, nargs='*', help='Additional tags to use for W&B')
    parser.add_argument('--pos-encoding', type=str, choices=['true', 'false'], default='true', help='Use positional encoding with input (3D coordinate)')
    parser.add_argument('--normalize',    type=str, choices=['true', 'false'], default='true', help='Normalize input to 0-mean and 1-std')
    parser.add_argument('--fp16',         type=str, choices=['true', 'false'], default='true', help='Use 16bit mixed-precision')
    parser.add_argument('--validation-every', type=int, default=500, help='Run validation step every n-th iteration')
    parser.add_argument('--no-validation', action='store_true', help='Skip validation')
    parser.add_argument('--debug', action='store_true', help='Turn of WandB, some more logs')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for experiment')

def get_index_upscale_function(vol_scaling_factor, device=None):
    up = int(1./vol_scaling_factor)
    assert up >= 1
    if up == 1.0: return (lambda x: x)
    x,y,z = torch.meshgrid(torch.arange(up), torch.arange(up), torch.arange(up), indexing='ij')
    mg = torch.stack([x,y,z], dim=-1).reshape(-1, 3)
    if device is not None:
        mg = mg.to(device)
        def idx_up(idx):
            return up*idx + mg[torch.randint(0, mg.size(0), (idx.size(0),))]
    else:
        def idx_up(idx):
            return up*idx + mg[torch.randint(0, mg.size(0), (idx.size(0),))].to(idx.device)
    return idx_up

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
