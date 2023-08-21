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
from icecream import ic
import domesutils


sampling_modes = {
    'uniform': sample_uniform,
    'surface': sample_surface,
    'both': sample_both,
}

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
                print(f'split_into_classes() {k}: {v.shape}, t: {t.shape}')
                sims[k] = t[:, idx:idx+v.size(0)]
                idx += v.size(0)
            return sims
        if len(annotations) == 0:
            return  # No NTFs
        abs_coords = torch.cat(list(annotations.values())).to(dev).to(typ)
        if abs_coords.numel() == 0:
            return  # No annotation in any of the NTFs
        rel_coords = (abs_coords.float() + 0.5) / vol_extent * 2.0 - 1.0

        print(f'rel_coords: {rel_coords.shape}')
        qf = sample_features3d(features, rel_coords, mode='bilinear').squeeze(0)  # (1, A, F)

        print(f'Features: {features.shape}, qf: {qf.shape}')
        if len(annotations) == 1 and list(annotations.values())[0].size(0) > 1024:
            sims = torch.einsum('fwhd,caf->cwhd', (features, qf)).squeeze(1).unsqueeze(-4) / qf.size(1)
        else:
            sims = torch.einsum('fwhd,caf->cawhd', (features, qf)).squeeze(1)

        lr_abs_coords = torch.round((rel_coords * 0.5 + 0.5) * (torch.tensor([*sims.shape[-3:]]).to(dev).to(typ) - 1.0)).long()  # (A, 3)
        lr_abs_coords = split_into_classes(make_3d(lr_abs_coords))  # (1, A, 3) -> {NTF_ID: (1, a, 3)}
        similarities = {}
        for k,sim in split_into_classes(sims).items():
            sim = torch.where(sim >= 0.25, sim, torch.zeros(1, dtype=typ, device=dev)) ** 2.5  # Throw away low similarities & exponentiate
            sim = sim.mean(dim=1)
            if bilateral_solver:
                print('Reducing & Solving ', k, sim.shape)
                bls_params = {
                    'sigma_spatial': 7,
                    'sigma_chroma':5,
                    'sigma_luma': 5,
                }
                vol = F.interpolate(make_5d(torch.as_tensor(volume)), sim_shape, mode='trilinear').squeeze()
                # vol = make_4d(vol.squeeze()).flip(-3)
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
                similarities[k] = (255.0 / (sim.quantile(q=0.9999)) * sim).cpu().to(torch.uint8).squeeze()
            else:
                similarities[k] = (255.0 / (sim.quantile(q=0.9999)) * sim).cpu().to(torch.uint8).squeeze()
                similarities[k] = F.interpolate(make_5d(similarities[k]), sim_shape, mode='nearest').squeeze()
        return similarities


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to features, annotations, volume etc.')
    parser.add_argument('--bilateral-solver', action='store_true', help='Use bilateral solver')
    parser.add_argument('--load-sims', action='store_true', help='Load similarities from file')
    parser.add_argument('--num-samples', type=float, default=0.0, help='Number of samples to use for each NTF')
    parser.add_argument('--sampling-mode', type=str, choices=['uniform', 'surface', 'both'], default='both', help='Sampling mode')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        dev, typ = torch.device('cuda'), torch.float16
    else:
        dev, typ = torch.device('cpu'), torch.float32
    # Load data
    dir = Path(args.data)
    if args.num_samples == 0.0:
        args.sampling_mode = 'annotated'
    bls_str = 'bls' if args.bilateral_solver else ''
    if (dir / f'ntf_pred{args.num_samples}{args.sampling_mode}{bls_str}.npy').exists():
        print(f'Already inferred NTF preds for {dir} using sampling mode {args.sampling_mode} and {args.num_samples} samples')
        exit(0)
    else:
        print(f'Inferring for {dir} using sampling mode {args.sampling_mode} and {args.num_samples} samples')

    feat_fns = list(filter(lambda p: 'features' in str(p) and 'pred' not in str(p), dir.iterdir()))
    if len(feat_fns) == 0:
        raise ValueError(f'No features found in {dir}')
    elif len(feat_fns) == 1:
        feat_fn = feat_fns[0]
    else:
        feat_fn = sorted(feat_fns, key=lambda p: p.stat().st_size)[-1]
        print(f'Found multiple features in {dir}. Using largest one {feat_fn.name}.')

    volume =      np.load(dir / 'volume.npy', allow_pickle=True).astype(np.float32)
    if (dir / 'labels.npy').exists():
        labels =      np.load(dir / 'labels.npy', allow_pickle=True)[()]
        labels = np.flip(labels, axis=-3).copy()
    else:
        assert args.num_samples == 0.0, 'Cannot sample labels if they are not provided'
        labels = None
    features =    np.load(dir / feat_fn, allow_pickle=True)[()]
    volume = np.flip(volume, axis=-3).copy()
    if isinstance(features, dict):
        features = torch.as_tensor(features['k']).float().squeeze()
    else:
        features = torch.as_tensor(features).float().squeeze()
    draw_samples = sampling_modes[args.sampling_mode]

    if args.num_samples == 0.0:
        annotations = np.load(dir / 'annotations.npy', allow_pickle=True)[()]  # { classname: (N, 3) }
        # annotations = {k: v[...,[2,1,0]] for k,v in annotations.items()}  # shuffle X,Y,Z -> Z,Y,X
        args.sampling_mode = 'annotated'
    elif args.num_samples > 1.0:
        annotations = {}
        for i in range(1, labels.max()+1):
            mask = torch.as_tensor(labels == i)
            N_SAMPLES = min(int(args.num_samples), mask.sum().item())
            if N_SAMPLES > 0:
                annotations[f'ntf{i}'] = draw_samples(mask, N_SAMPLES, thin_to_reasonable=True)
    elif args.num_samples > 0.0:
        annotations = {}
        for i in range(1, labels.max()+1):
            mask = torch.as_tensor(labels == i)
            N_SAMPLES = int(args.num_samples * mask.sum().item())
            if N_SAMPLES > 0:
                annotations[f'ntf{i}'] = draw_samples(mask, N_SAMPLES, thin_to_reasonable=True)
    else:
        raise Exception(f'Invalid value for --num-samples: {args.num_samples}')

    # BG_SAMPLES = max(list(map(lambda v: v.size(0), annotations.values()))) if args.num_samples != 0.0 else 128
    # bg_samples = draw_samples(labels == 0, BG_SAMPLES, thin_to_reasonable=True)
    # annotations = { 'background': bg_samples, **annotations }
    # Compute similarities
    print(f'Computing similarties for {tuple(volume.shape)} with features {tuple(features.shape)}')
    t0 = time.time()
    t1 = t0
    if args.load_sims:
        similarities = {k: torch.as_tensor(v) for k,v in np.load(dir / 'similarities.npy', allow_pickle=True)[()].items()}
    else:
        if torch.cat(list(annotations.values())).size(0) > 10000:
            similarities = {k: compute_similarities(volume, features.to(device=dev, dtype=typ), {k: v}, bilateral_solver=args.bilateral_solver)[k] for k,v in annotations.items()}
        else:
            similarities = compute_similarities(volume, features.to(device=dev, dtype=typ), annotations, bilateral_solver=args.bilateral_solver)
        similarities = {k: v.cpu().float() for k,v in similarities.items()}
    t2 = time.time()
    # Compare to similarities on disk
    # similarities_exported = np.load(dir / 'similarities.npy', allow_pickle=True)[()]
    # for k in similarities.keys():
        # sim_exp = similarities_exported[k]
        # sim = similarities[k]
        # dist = torch.abs(sim -sim_exp).float()
        # print(f'{k}: {sim.shape} ({sim.min()}, {sim.max()}) vs {sim_exp.shape} ({sim_exp.min()}, {sim_exp.max()})')
        # print('all close?', torch.allclose(sim, sim_exp), 'distance', dist.mean(), 'max distance', dist.max())
    print('Similarities:', {k: v.shape for k,v in similarities.items()})
    sims = torch.stack(list(similarities.values()))
    # pred[1:] = torch.where(pred[1:] < 50, 0, pred[1:])
    pred = torch.zeros_like(sims[0])
    pred_vals = torch.zeros_like(sims[0])
    ct_org_names = ['liver', 'bladder', 'lung', 'kidney', 'bone']
    ct_org_thresholds = [0.615, 0.93, 0.5, 0.85, 0.6]
    from itertools import count
    min_sim = int(0.6 * 255)
    for i, n, sim in zip(count(), ct_org_names, sims):
        mask = (sim > int(ct_org_thresholds[i] * 255)) & (sim > pred_vals)
        pred[mask] = i+1
        pred_vals[mask] = sim[mask]
    pred = pred.cpu().numpy().astype(np.uint8)
    np.save(dir / f'ntf_pred{args.num_samples}{args.sampling_mode}{bls_str}.npy', pred)
    if tuple(pred.shape[-3:]) != tuple(volume.shape[-3:]):
        pred = F.interpolate(make_5d(torch.as_tensor(pred)), tuple(volume.shape[-3:]), mode='nearest').squeeze().numpy()
    print('Pred:', pred.shape, pred.min(), pred.max())
    print('NTF fit time:', t1 - t0)
    print('NTF predict time:', t2 - t1)

    if labels is None:
        exit(0)
    pred = pred.reshape(-1)
    ic(pred)
    ic(labels.reshape(-1))
    prec, rec, f1, _ = precision_recall_fscore_support(labels.reshape(-1), pred, average=None)
    cm = confusion_matrix(labels.reshape(-1), pred)
    acc = cm.diagonal() / cm.sum(axis=1)
    iou = jaccard_score(labels.reshape(-1), pred, average=None)
    label_names = ['background'] + list(annotations.keys())
    ntf_metrics = {
        'accuracy': dict(zip(label_names, acc.tolist())),
        'precision': dict(zip(label_names, prec.tolist())),
        'recall': dict(zip(label_names, rec.tolist())),
        'f1': dict(zip(label_names, f1.tolist())),
        'iou': dict(zip(label_names, iou.tolist())),
        'confusion_matrix': dict(zip(label_names, cm.tolist())),
        'fit_time': t1 - t0,
        'predict_time': t2 - t1,
    }
    print('NTF Metrics:')
    pprint(ntf_metrics)
    with open(dir / f'ntf_metrics{args.num_samples}{args.sampling_mode}{bls_str}.json', 'w') as f:
        json.dump(ntf_metrics, f)
    # for i in range(0, labels.max()+1):
    #    nam = label_names[i]
    #    pred_bin = pred == i
    #    labl_bin = labels.reshape(-1) == i
    #    bin_iou = jaccard_score(labl_bin, pred_bin)
    #    print(f'Binary {nam} IOU:', bin_iou)
