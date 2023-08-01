import torch
import torch.nn.functional as F
import numpy as np
from infer import sample_features3d, make_3d, make_4d, make_5d, norm_minmax
from pathlib import Path
from argparse import ArgumentParser

from scipy.ndimage import binary_erosion, generate_binary_structure

DATA_DIR = Path('/run/media/dome/SSD/Data/Volumes/CT-ORG')
ONE = torch.ones(1)

def sample_uniform(vol, n_samples, thin_to_reasonable=False):
    idxs = torch.as_tensor(vol).nonzero()
    while thin_to_reasonable and idxs.size(0) > int(2**24):
        idxs = idxs[::2]
    return idxs[torch.multinomial(ONE.expand(idxs.size(0)), n_samples)]

def sample_surface(vol, n_samples, dist_from_surface=4):
    stel1 = generate_binary_structure(rank=3, connectivity=dist_from_surface)
    stel2 = generate_binary_structure(rank=3, connectivity=1)
    outer = binary_erosion(vol, stel1)
    inner = binary_erosion(outer, stel2)
    print('outer', outer.sum(), tuple(map(lambda c: (c.min(), c.max()), outer.nonzero())))
    print('inner', inner.sum(), tuple(map(lambda c: (c.min(), c.max()), inner.nonzero())))

    surface_idxs = torch.as_tensor(np.logical_xor(inner, outer)).nonzero()
    if surface_idxs.size(0) > n_samples:
        return surface_idxs[torch.multinomial(ONE.expand(surface_idxs.size(0)), n_samples)]
    else:
        print(f'Full surface only has {surface_idxs.size(0)} voxels (< n_samples={n_samples}).')
        return surface_idxs

def sample_both(vol, n_samples, dist_from_surface=4, thin_to_reasonable=False):
    return torch.cat([sample_uniform(vol, n_samples//2, thin_to_reasonable=thin_to_reasonable), sample_surface(vol, n_samples//2, dist_from_surface=dist_from_surface)])

if __name__ == '__main__':
    parser = ArgumentParser('Compare similarity maps for different samplings of ground truth segmentations')
    parser.add_argument('--num-samples', type=float, default=512, help='Number of samples. if < 1 it uses fraction')
    args = parser.parse_args()
    vol = torch.from_numpy(np.load(DATA_DIR / 'volume-10.npy', allow_pickle=True))
    feats = torch.from_numpy(np.load(DATA_DIR / 'volume-10.nii_DINOfeats_all.npy', allow_pickle=True)[()]['k'])
    label = np.load(DATA_DIR / 'labels-10.npy', allow_pickle=True)
    dev = torch.device('cpu')
    typ = torch.float32
    # Move Stuff
    feats = F.normalize(feats.to(dev).to(typ).squeeze(), dim=0)

    print('vol', vol.shape, vol.dtype, vol.min(), vol.max())
    print('feats', feats.shape, feats.dtype, feats.min(), feats.max())
    print('label', label.shape, label.dtype, label.min(), label.max())

    vol_extent = torch.tensor([[*vol.shape[-3:]]])
    def abs2rel(abs_coord):
        return (abs_coord.float() + 0.5) / vol_extent * 2.0 - 1.0

    for i in range(1,label.max()+1):
        mask = label == i
        if args.num_samples > 1.0:
            N_SAMPLES = min(int(args.num_samples), mask.sum().item())
        elif args.num_samples > 0.0:
            N_SAMPLES = int(args.num_samples * mask.sum().item())
        else:
            raise Exception(f'Invalid value for --num-samples: {args.num_samples}')

        print(f'Class {i} has {mask.sum()} voxels, sampling {N_SAMPLES}')
        for sample in [sample_uniform]:
            abs_coords = sample(mask, N_SAMPLES)
            print('abs_coords', abs_coords.shape)
            rel_coords = abs2rel(abs_coords).to(dev).to(typ)
            qf = sample_features3d(feats, make_3d(rel_coords), mode='bilinear').squeeze()
            print('qf', qf.shape)
            if qf.size(0) > int(2**14):
                sim = 0
                cnt = 0
                for q in qf.split(int(2**14)):
                    s = torch.sum(torch.einsum('fwhd,nf->nwhd', (feats, q))**2, dim=0)
                    sim = (sim * cnt + s) / (cnt + q.size(0))
                    cnt += q.size(0)
            else:
                sim = torch.einsum('fwhd,nf->nwhd', (feats, qf)) ** 2.0
                sim = sim.mean(dim=0)
            print('sim', sim.shape, sim.min(), sim.max())
            sim = (255.0 / sim.quantile(q=0.9999) * sim).clamp(0, 255).to(torch.uint8)
            print('sim', sim.shape, sim.min(), sim.max())
            np.save(DATA_DIR / f'sim_{i}_{sample.__name__}{args.num_samples}.npy', sim.numpy())
