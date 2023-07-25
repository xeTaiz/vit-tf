import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, jaccard_score, confusion_matrix
import time, json

from infer import make_3d, make_4d, make_5d, sample_features3d, norm_minmax
from compare_feat_sampling import sample_uniform, sample_surface, sample_both
from bilateral_solver3d import apply_bilateral_solver3d, crop_pad, write_crop_into

from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

def compute_similarities(volume, features, annotations, bilateral_solver=False):
    ''' Computes similarities between features and annotations.
    Args:
        volume (torch.Tensor): volume (W, H, D)
        features (torch.Tensor): features (F, W, H, D)
        annotations (dict): annotations { classname: (N, 3) }
        bilateral_solver (bool): use bilateral solver. Default: False

    Returns:
        dict: similarities { classname: (W, H, D) }
    '''
    similarities = {}
    for classname, points in annotations.items():
        # Compute similarities
        with torch.no_grad():
            dev, typ = features.device, features.dtype
            in_dims = tuple(volume.shape[-3:])
            sim_shape = tuple(map(lambda d: d//2, in_dims))
            vol_extent = torch.tensor([[*in_dims]], device=dev, dtype=typ)
            def split_into_classes(t):
                sims = {}
                idx = 0
                for k,v in annotations.items():
                    sims[k] = t[:, idx:idx+v.size(0)]
                    idx += v.size(0)
                return sims
            if len(annotations) == 0: return  # No NTFs
            abs_coords = torch.cat(list(annotations.values())).to(dev).to(typ)
            if abs_coords.numel() == 0: return # No annotation in any of the NTFs
            rel_coords = (abs_coords.float() + 0.5) / vol_extent * 2.0 - 1.0

            print(f'rel_coords: {rel_coords.shape}')
            qf = sample_features3d(features, rel_coords, mode='bilinear').squeeze(0) # (1, A, F)

            print(f'Features: {features.shape}, qf: {qf.shape}')
            sims = torch.einsum('fwhd,caf->cawhd', (features, qf)).squeeze(1)

            lr_abs_coords = torch.round((rel_coords * 0.5 + 0.5) * (torch.tensor([*sims.shape[-3:]]).to(dev).to(typ) - 1.0)).long() # (A, 3)
            lr_abs_coords = split_into_classes(make_3d(lr_abs_coords)) # (1, A, 3) -> {NTF_ID: (1, a, 3)}
            similarities = {}
            rel_coords_dict = split_into_classes(make_3d(rel_coords))
            for k,sim in split_into_classes(sims).items():
                sim = torch.where(sim >= 0.25, sim, torch.zeros(1, dtype=typ, device=dev)) ** 2.5 # Throw away low similarities & exponentiate
                sim = sim.mean(dim=1)
                if bilateral_solver:
                    print('Reducing & Solving ', k, sim.shape)
                    bls_params = {
                        'sigma_spatial': 5,
                        'sigma_chroma':3,
                        'sigma_luma': 3,
                    }
                    vol = F.interpolate(make_5d(torch.as_tensor(volume)), sim_shape, mode='trilinear').squeeze()
                    vol = make_4d(vol.squeeze())
                    print('vol after interpolation', vol.shape)
                    vol = norm_minmax(vol)
                    vol = (255.0 * vol).to(torch.uint8)
                    if tuple(sim.shape[-3:]) != sim_shape:
                        print(f'Resizing {k} similarity to', sim_shape)
                        sim = F.interpolate(make_5d(sim), sim_shape, mode='trilinear').squeeze(0)
                    # Apply Bilateral Solver
                    print('sim.shape', sim.shape, 'vol.shape', vol.shape)
                    crops, mima = crop_pad([sim, vol], thresh=0.1, pad=2)
                    csim, cvol = crops
                    csim = apply_bilateral_solver3d(make_4d(csim), cvol.expand(3, -1,-1,-1), grid_params=bls_params)
                    sim = write_crop_into(sim, csim, mima)
                    print('Wrote crop into original similarity map', csim.shape, '->', sim.shape)
                    similarities[k] = (255.0 / 0.99 * sim).cpu().to(torch.uint8).squeeze()
                else:
                    similarities[k] = (255.0 / 0.99 * sim).cpu().to(torch.uint8).squeeze()
            return similarities

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to features, annotations, volume etc.')
    parser.add_argument('--bilateral-solver', action='store_true', help='Use bilateral solver')
    parser.add_argument('--num-samples', type=float, default=0.0, help='Number of samples to use for each NTF')
    args = parser.parse_args()

    # Load data
    dir = Path(args.data)
    volume =      np.load(dir / 'volume.npy', allow_pickle=True).astype(np.float32)
    annotations = np.load(dir / 'annotations.npy', allow_pickle=True)
    labels =      np.load(dir / 'labels.npy', allow_pickle=True)
    features =    torch.as_tensor(np.load(dir / 'dino_features.npy', allow_pickle=True)).squeeze().float()

    if args.num_samples == 0.0:
        annotations = np.load(dir / 'annotations.npy', allow_pickle=True)[()]  # { classname: (N, 3) }
    elif args.num_samples > 1.0:
        annotations = {}
        for i in range(1, labels.max()+1):
            mask = torch.as_tensor(labels == i)
            N_SAMPLES = min(int(args.num_samples), mask.sum().item())
            annotations[f'ntf{i}'] = sample_uniform(mask, N_SAMPLES, thin_to_reasonable=True)
    elif args.num_samples > 0.0:
        annotations = {}
        for i in range(1, labels.max()+1):
            mask = torch.as_tensor(labels == i)
            N_SAMPLES = int(args.num_samples * mask.sum().item())
            annotations[f'ntf{i}'] = sample_uniform(mask, N_SAMPLES, thin_to_reasonable=True)
    else:
        raise Exception(f'Invalid value for --num-samples: {args.num_samples}')

    # BG_SAMPLES = max(list(map(lambda v: v.size(0), annotations.values()))) if args.num_samples != 0.0 else 128
    # bg_samples = sample_uniform(labels == 0, BG_SAMPLES, thin_to_reasonable=True)
    # annotations = { 'background': bg_samples, **annotations }
    # Compute similarities
    print(f'Computing similarties for {tuple(volume.shape)} with features {tuple(features.shape)}')
    t0 = time.time()
    t1 = t0
    similarities = compute_similarities(volume, features, annotations, bilateral_solver=args.bilateral_solver)
    t2 = time.time()
    print('Similarities:', {k: v.shape for k,v in similarities.items()})
    sims = torch.stack(list(similarities.values()))
    #pred[1:] = torch.where(pred[1:] < 50, 0, pred[1:])
    pred = torch.zeros_like(sims[0])
    pred_vals = torch.zeros_like(sims[0])
    min_sim = 50
    for i, sim in enumerate(sims):
        mask = (sim > min_sim) & (sim > pred_vals)
        pred[mask] = i+1
        pred_vals[mask] = sim[mask]
    pred = pred.cpu().numpy().astype(np.uint8)
    np.save(dir / f'ntf_pred{args.num_samples}.npy', pred)
    if tuple(pred.shape[-3:]) != tuple(volume.shape[-3:]):
        pred = F.interpolate(make_5d(torch.as_tensor(pred)), tuple(volume.shape[-3:]), mode='nearest').squeeze().numpy()
    print('Pred:', pred.shape, pred.min(), pred.max())
    print('NTF fit time:', t1 - t0)
    print('NTF predict time:', t2 - t1)
    pred = pred.reshape(-1)
    prec, rec, f1, _ = precision_recall_fscore_support(labels.reshape(-1), pred, average=None)
    cm = confusion_matrix(labels.reshape(-1), pred)
    acc = cm.diagonal() / cm.sum(axis=1)
    iou = jaccard_score(labels.reshape(-1), pred, average=None)
    ntf_metrics = {
        'accuracy': dict(zip(annotations.keys(), acc.tolist())),
        'precision': dict(zip(annotations.keys(), prec.tolist())),
        'recall': dict(zip(annotations.keys(), rec.tolist())),
        'f1': dict(zip(annotations.keys(), f1.tolist())),
        'iou': dict(zip(annotations.keys(), iou.tolist())),
        'confusion_matrix': dict(zip(annotations.keys(), cm.tolist())),
        'fit_time': t1 - t0,
        'predict_time': t2 - t1,
    }
    print('NTF Metrics:')
    pprint(ntf_metrics)
    with open(dir / f'ntf_metrics{args.num_samples}.json', 'w') as f:
        json.dump(ntf_metrics, f)
