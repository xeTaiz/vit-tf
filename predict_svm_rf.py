import time, json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score, confusion_matrix
import sys

from pathlib import Path
from argparse import ArgumentParser
from pprint import pprint

from infer import make_3d, make_4d, make_5d, sample_features3d
from compare_feat_sampling import sample_uniform, sample_surface, sample_both

sampling_modes = {
    'uniform': sample_uniform,
    'surface': sample_surface,
    'both': sample_both,
}

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
    i = 0
    for classname, annotation in annotations.items():
        rel_coords = make_3d((annotation + 0.5) / torch.tensor(features.shape[-3:]))
        sampled_features.append(sample_features3d(features, rel_coords).squeeze(0).squeeze(0))
        if labels is not None:
            sampled_labels.append(sample_features3d(make_4d(labels), rel_coords).squeeze(0).squeeze(0).squeeze(-1))
        else:
            sampled_labels.append(torch.ones(rel_coords.size(1), dtype=torch.uint8) * i)
        i += 1
    return torch.cat(sampled_features, dim=0).numpy(), torch.cat(sampled_labels, dim=0).byte().numpy()


if __name__ == '__main__':
    parser = ArgumentParser("Predict Segmentation using SVM and Random Forests")
    parser.add_argument('--data', type=str, required=True, help='Path to volume and annotation data')
    parser.add_argument('--svm-kernel', type=str, choices=['linear', 'poly', 'rgb', 'sigmoid', 'precomputed'], default='rbf', help='SVM Kernel function, see scikit-learn docs')
    parser.add_argument('--load-sims', action='store_true', help='Load similarities from file')
    parser.add_argument('--num-samples', type=float, default=0.0, help='Number of samples to use for training, 0 to use annotations.npy')
    parser.add_argument('--sampling-mode', type=str, choices=['uniform', 'surface', 'both'], default='uniform', help='Sampling mode')
    args = parser.parse_args()

    dir = Path(args.data)
    if (dir / f'svm_metrics_{args.sampling_mode}{args.num_samples}.json').exists() and (dir / f'rf_metrics_{args.sampling_mode}{args.num_samples}.json').exists():
        print(f'Already inferred SVM and RF metrics for {dir} using sampling mode {args.sampling_mode} and {args.num_samples} samples')
        sys.exit(0)
    else:
        print(f'Inferring for {dir} using sampling mode {args.sampling_mode} and {args.num_samples} samples')

    volume      = np.load(dir / 'volume.npy',      allow_pickle=True)  # (W,H,D,1)
    if (dir / 'labels.npy').exists():
        labels      = np.load(dir / 'labels.npy',      allow_pickle=True)  # (W,H,D)
    else:
        assert args.num_samples == 0.0, 'Cannot sample labels if they are not provided'
        labels = None

    draw_samples = sampling_modes[args.sampling_mode]
    if args.num_samples == 0.0:
        annotations = np.load(dir / 'annotations.npy', allow_pickle=True)[()]  # { classname: (N, 3) }
    elif args.num_samples > 1.0:
        annotations = {}
        for i in range(1, labels.max()+1):
            mask = torch.as_tensor(labels == i)
            N_SAMPLES = min(int(args.num_samples), mask.sum().item())
            annotations[f'ntf{i}'] = draw_samples(mask, N_SAMPLES, thin_to_reasonable=True)
    elif args.num_samples > 0.0:
        annotations = {}
        for i in range(1, labels.max()+1):
            mask = torch.as_tensor(labels == i)
            N_SAMPLES = int(args.num_samples * mask.sum().item())
            annotations[f'ntf{i}'] = draw_samples(mask, N_SAMPLES, thin_to_reasonable=True)
    else:
        raise Exception(f'Invalid value for --num-samples: {args.num_samples}')

    BG_SAMPLES = max(list(map(lambda v: v.size(0), annotations.values()))) if args.num_samples != 0.0 else 128
    if labels is not None:
        annotations['background'] = draw_samples(labels == 0, BG_SAMPLES, thin_to_reasonable=True)
    else:  # Sample from border
        bg_samples = torch.ones(volume.shape[-3:], dtype=torch.bool)
        bg_samples[4:-4,4:-4,4:-4] = False
        annotations['background'] = draw_samples(bg_samples, BG_SAMPLES, thin_to_reasonable=True)

    # input has 11-dim features containing intensity, grad mag, intensities of 6 neighbors and voxel coordinate, normalized to mean 0 std 1
    features = compose_features(torch.from_numpy(volume.astype(np.float32)))

    if labels is not None:
        train_X, train_y = sample_train_data(features, torch.from_numpy(labels).float(), annotations)
    else:
        train_X, train_y = sample_train_data(features, None, annotations)
    # input to fit must be (N, F)
    # labels to fit must be (N,)
    if not (dir / f'svm_metrics{args.num_samples}{args.sampling_mode}.json').exists():
        clf = SVC(kernel=args.svm_kernel)
        t_svm_0 = time.time()
        clf.fit(train_X, train_y)
        t_svm_1 = time.time()
        svm_pred = clf.predict(features.permute(1,2,3,0).reshape(-1, features.shape[0]).numpy())
        t_svm_2 = time.time()
        print('SVM fit time:', t_svm_1 - t_svm_0)
        print('SVM predict time:', t_svm_2 - t_svm_1)
        np.save(dir / f'svm_pred{args.num_samples}{args.sampling_mode}.npy', svm_pred.reshape(volume.shape))
        if labels is not None:
            prec, rec, f1, _ = precision_recall_fscore_support(labels.reshape(-1), svm_pred, average=None)
            cm = confusion_matrix(labels.reshape(-1), svm_pred)
            acc = cm.diagonal() / cm.sum(axis=1)
            iou = jaccard_score(labels.reshape(-1), svm_pred, average=None)
            svm_metrics = {
                'accuracy': dict(zip(annotations.keys(), acc.tolist())),
                'precision': dict(zip(annotations.keys(), prec.tolist())),
                'recall': dict(zip(annotations.keys(), rec.tolist())),
                'f1': dict(zip(annotations.keys(), f1.tolist())),
                'iou': dict(zip(annotations.keys(), iou.tolist())),
                'confusion_matrix': dict(zip(annotations.keys(), cm.tolist())),
                'fit_time': t_svm_1 - t_svm_0,
                'predict_time': t_svm_2 - t_svm_1,
            }
            print('SVM Metrics:')
            pprint(svm_metrics)
            with open(dir / f'svm_metrics{args.num_samples}{args.sampling_mode}.json', 'w') as f:
                json.dump(svm_metrics, f)

    if not (dir / f'rf_metrics{args.num_samples}{args.sampling_mode}.json').exists():
        clf = RandomForestClassifier(n_estimators=32)
        t_rf_0 = time.time()
        clf.fit(train_X, train_y)
        t_rf_1 = time.time()
        rf_pred = clf.predict(features.permute(1,2,3,0).reshape(-1, features.shape[0]).numpy())
        t_rf_2 = time.time()
        print('RF fit time:', t_rf_1 - t_rf_0)
        print('RF predict time:', t_rf_2 - t_rf_1)
        np.save(dir / f'rf_pred{args.num_samples}{args.sampling_mode}.npy', rf_pred.reshape(volume.shape))
        if labels is not None:
            prec, rec, f1, _ = precision_recall_fscore_support(labels.reshape(-1), rf_pred, average=None)
            cm = confusion_matrix(labels.reshape(-1), rf_pred)
            acc = cm.diagonal() / cm.sum(axis=1)
            iou = jaccard_score(labels.reshape(-1), rf_pred, average=None)
            rf_metrics = {
                'accuracy': dict(zip(annotations.keys(), acc.tolist())),
                'precision': dict(zip(annotations.keys(), prec.tolist())),
                'recall': dict(zip(annotations.keys(), rec.tolist())),
                'f1': dict(zip(annotations.keys(), f1.tolist())),
                'iou': dict(zip(annotations.keys(), iou.tolist())),
                'confusion_matrix': dict(zip(annotations.keys(), cm.tolist())),
                'fit_time': t_rf_1 - t_rf_0,
                'predict_time': t_rf_2 - t_rf_1,
            }
            print('RF Metrics:')
            pprint(rf_metrics)
            with open(dir / f'rf_metrics{args.num_samples}{args.sampling_mode}.json', 'w') as f:
                json.dump(rf_metrics, f)
