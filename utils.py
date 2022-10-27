import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def split_squeeze(t, bs, f):
    n = t.size(0)
    X, Y, Z = t[None, None].expand(bs, f, n, 3).split(1, dim=-1)
    BS = torch.arange(bs)[:, None, None].expand(bs, f, n)
    F =  torch.arange(f)[None, :, None].expand(bs, f, n)
    return (BS, F, X.squeeze(-1), Y.squeeze(-1), Z.squeeze(-1))

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
