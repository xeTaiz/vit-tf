import time, json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score, confusion_matrix

from pathlib import Path
from argparse import ArgumentParser

from infer import make_3d, make_4d, make_5d, sample_features3d
from compare_feat_sampling import sample_uniform, sample_surface, sample_both

def get_gradient_magnitude(volume):
    ''' Computes central differences of volume.
    Args:
        volume (torch.Tensor): volume ([F,] W,H,D)

    Returns:
        torch.Tensor: Gradient magnitude (W, H, D)
    '''
    win = torch.tensor([-0.5, 0, 0.5])[None, None, None, None].to(volume.dtype)
    out = F.conv3d(volume, win, groups=volume.size(1), padding=(0,0,1))**2
    out += F.conv3d(volume, win.transpose(3, 4), groups=volume.size(1), padding=(0,1,0))**2
    out += F.conv3d(volume, win.transpose(2, 4), groups=volume.size(1), padding=(1,0,0))**2
    return out.sqrt()

def get_neighbors6(volume):
    pad_vol = F.pad(volume, (1,1,1,1,1,1), mode='replicate')
    return torch.cat([
        pad_vol[..., 2:,   1:-1, 1:-1],
        pad_vol[..., 1:-1, 2:,   1:-1],
        pad_vol[..., 1:-1, 1:-1, 2:  ],
        pad_vol[...,  :-2, 1:-1, 1:-1],
        pad_vol[..., 1:-1,  :-2, 1:-1],
        pad_vol[..., 1:-1, 1:-1,  :-2],
    ], dim=-4)

def get_coordinates(volume):
    return torch.stack(torch.meshgrid(torch.arange(volume.size(-3)), torch.arange(volume.size(-2)), torch.arange(volume.size(-1)), indexing='ij'))

def compose_features(volume):
    intensity = make_4d(volume)
    grad_mag  = get_gradient_magnitude(make_5d(volume)).squeeze(0)
    neighbors = get_neighbors6(intensity)
    coords    = get_coordinates(intensity)
    features  = torch.cat([intensity, grad_mag, neighbors, coords], dim=0)
    return (features - features.mean(dim=(-1,-2,-3), keepdim=True)) / features.std(dim=(-1,-2,-3), keepdim=True)

def sample_train_data(features, labels, annotations):
    ''' Samples features and labels from volume and annotations.
    Args:
        features (torch.Tensor): features (F, W, H, D)
        labels (torch.Tensor): labels (W, H, D)
        annotations (dict): annotations { classname: (N, 3) }

    Returns:
        torch.Tensor: sampled features (N, F)
        torch.Tensor: sampled labels (N,)
    '''
    sampled_features = []
    sampled_labels = []
    for classname, annotation in annotations.items():
        rel_coords = make_3d((annotation + 0.5) / torch.tensor(labels.shape[-3:]))
        sampled_features.append(sample_features3d(features, rel_coords).squeeze(dim=(0,1)))
        sampled_labels.append(sample_features3d(make_4d(labels), rel_coords).squeeze(dim=(0,1,-1)))
        print('sampled_features', sampled_features[-1].shape)
        print('sampled_labels', sampled_labels[-1].shape)
    return torch.cat(sampled_features, dim=0).numpy(), torch.cat(sampled_labels, dim=0).byte().numpy()


if __name__ == '__main__':
    parser = ArgumentParser("Predict Segmentation using SVM and Random Forests")
    parser.add_argument('--data', type=str, required=True, help='Path to volume and annotation data')
    parser.add_argument('--svm-kernel', type=str, choices=['linear', 'poly', 'rgb', 'sigmoid', 'precomputed'], default='rbf', help='SVM Kernel function, see scikit-learn docs')
    args = parser.parse_args()

    dir = Path(args.data)
    volume      = np.load(dir / 'volume.npy',      allow_pickle=True)  # (W,H,D,1)
    annotations = np.load(dir / 'annotations.npy', allow_pickle=True)[()]  # { classname: (N, 3) }
    labels      = np.load(dir / 'labels.npy',      allow_pickle=True)  # (W,H,D)

    print(f'Volume: {volume.shape}, {volume.dtype}')
    print(f'Labels: {labels.shape}, {labels.dtype}')
    print('Annotations: ', {k: v.shape for k, v in annotations.items()})
    print('Adding background annotations')
    annotations['background'] = sample_uniform(labels == 0, 128, thin_to_reasonable=True)
    print('Annotations: ', {k: v.shape for k, v in annotations.items()})
    # input has 11-dim features containing intensity, grad mag, intensities of 6 neighbors and voxel coordinate, normalized to mean 0 std 1
    features = compose_features(torch.from_numpy(volume).float())
    print(features.shape, features.mean(), features.std())

    train_X, train_y = sample_train_data(features, torch.from_numpy(labels).float(), annotations)
    print(f'Training data: {train_X.shape} ({train_X.dtype}), {train_y.shape} ({train_y.dtype})')
    # input to fit must be (N, F)
    # labels to fit must be (N,)
    clf = SVC(kernel=args.svm_kernel)
    t_svm_0 = time.time()
    clf.fit(train_X, train_y)
    t_svm_1 = time.time()
    svm_pred = clf.predict(features.permute(1,2,3,0).reshape(-1, features.shape[0]).numpy())
    t_svm_2 = time.time()
    print('SVM fit time:', t_svm_1 - t_svm_0)
    print('SVM predict time:', t_svm_2 - t_svm_1)
    np.save(dir / 'svm_pred.npy', svm_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(labels.reshape(-1), svm_pred, average=None)
    cm = confusion_matrix(labels.reshape(-1), svm_pred)
    acc = cm.diagonal() / cm.sum(axis=1)
    iou = jaccard_score(labels.reshape(-1), svm_pred, average=None)
    svm_metrics = {
        'accuracy': acc.tolist(),
        'precision': prec.tolist(),
        'recall': rec.tolist(),
        'f1': f1.tolist(),
        'iou': iou.tolist(),
        'confusion_matrix': cm.tolist(),
        'fit_time': t_svm_1 - t_svm_0,
        'predict_time': t_svm_2 - t_svm_1,
    }
    print('SVM Metrics:', svm_metrics)
    with open(dir / 'svm_metrics.json', 'w') as f:
        json.dump(svm_metrics, f)
