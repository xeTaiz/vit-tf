import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def split_squeeze(t, bs, f):
    n = t.size(0)
    X, Y, Z = t[None, None].expand(bs, f, n, 3).split(1, dim=-1)
    BS = torch.arange(bs)[:, None, None].expand(bs, f, n)
    F =  torch.arange(f)[None, :, None].expand(bs, f, n)
    return (BS, F, X.squeeze(-1), Y.squeeze(-1), Z.squeeze(-1))

def log_tensor(t, name):
    print(f'{name}: {tuple(t.shape)} in value range [{t.min().item():.3f}, {t.max().item():.3f}] and of type {t.dtype}')

def norm_minmax(t):
    mi, ma = t.min(), t.max()
    return (t - mi) / (ma - mi)

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
