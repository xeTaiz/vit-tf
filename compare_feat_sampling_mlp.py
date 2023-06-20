import torch
import torch.nn.functional as F
import numpy as np
from infer import sample_features3d, make_3d, make_4d, make_5d, norm_minmax
from pathlib import Path
from argparse import ArgumentParser

from scipy.ndimage import binary_erosion, generate_binary_structure
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

DATA_DIR = Path('/run/media/dome/SSD/Data/Volumes/CT-ORG')
ONE = torch.ones(1)

def sample_uniform(vol, n_samples):
    idxs = torch.from_numpy(vol).nonzero()
    return idxs[torch.multinomial(ONE.expand(idxs.size(0)), n_samples)]

def sample_surface(vol, n_samples, dist_from_surface=4):
    stel1 = generate_binary_structure(rank=3, connectivity=dist_from_surface)
    stel2 = generate_binary_structure(rank=3, connectivity=1)
    outer = binary_erosion(vol, stel1)
    inner = binary_erosion(outer, stel2)
    print('outer', outer.sum(), tuple(map(lambda c: (c.min(), c.max()), outer.nonzero())))
    print('inner', inner.sum(), tuple(map(lambda c: (c.min(), c.max()), inner.nonzero())))

    surface_idxs = torch.from_numpy(np.logical_xor(inner, outer)).nonzero()
    if surface_idxs.size(0) > n_samples:
        return surface_idxs[torch.multinomial(ONE.expand(surface_idxs.size(0)), n_samples)]
    else:
        print(f'Full surface only has {surface_idxs.size(0)} voxels (< n_samples={n_samples}).')
        return surface_idxs

def sample_both(vol, n_samples, dist_from_surface=4):
    return torch.cat([sample_uniform(vol, n_samples//2), sample_surface(vol, n_samples//2, dist_from_surface=dist_from_surface)])

if __name__ == '__main__':
    parser = ArgumentParser('Compare similarity maps for different samplings of ground truth segmentations')
    parser.add_argument('--num-samples', type=float, default=512, help='Number of samples. if < 1 it uses fraction')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--bs', type=int, default=32, help='Batch Size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight Decay')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')
    args = parser.parse_args()
    vol = torch.from_numpy(np.load(DATA_DIR / 'volume-10.npy', allow_pickle=True))
    feats = torch.from_numpy(np.load(DATA_DIR / 'volume-10.nii_DINOfeats_all.npy', allow_pickle=True)[()]['k'])
    label = np.load(DATA_DIR / 'labels-10.npy', allow_pickle=True) 
    label_lores = F.interpolate(make_5d(torch.from_numpy(label)), size=feats.shape[-3:], mode='nearest').squeeze()
    num_classes_with_bg = int(label.max()+1)

    head = torch.nn.Linear(feats.size(0), label.max())
    opt = torch.optim.AdamW(head.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epochs)
    iou = MulticlassJaccardIndex(num_classes=num_classes_with_bg, average='none')
    acc = MulticlassAccuracy(num_classes=num_classes_with_bg, average='none')

    dev = torch.device('cuda' if torch.cuda.is_available and not args.cpu else 'cpu')
    typ = torch.float32
    # Move Stuff
    feats = F.normalize(feats.to(dev).to(typ).squeeze(), dim=0)
    head = head.to(dev).to(typ)
    iou = iou.to(dev)
    acc = acc.to(dev)
    label_lores = label_lores.to(dev)

    print('head', head)
    print('vol', vol.shape, vol.dtype, vol.min(), vol.max())
    print('feats', feats.shape, feats.dtype, feats.min(), feats.max())
    print('label', label.shape, label.dtype, label.min(), label.max())

    vol_extent = torch.tensor([[*vol.shape[-3:]]])
    def abs2rel(abs_coord):
        return (abs_coord.float() + 0.5) / vol_extent * 2.0 - 1.0

    samples, gts = {}, {}
    for sample in [sample_uniform, sample_surface, sample_both]:
        for i in range(1,label.max()+1):
            mask = label == i
            if args.num_samples > 1.0:
                N_SAMPLES = min(int(args.num_samples), mask.sum().item())
            elif args.num_samples > 0.0:
                N_SAMPLES = int(args.num_samples * mask.sum().item())
            else:
                raise Exception(f'Invalid value for --num-samples: {args.num_samples}')
            print(f'mask {i}', mask.shape, mask.sum().item())
            print('N_SAMPLES', N_SAMPLES)
            samples[i] = sample(mask, N_SAMPLES)
            gts[i] = F.one_hot(torch.tensor([i-1]).expand(N_SAMPLES).long().to(dev), label.max()).to(typ)
            print(f'Class {i} has {mask.sum()} voxels, sampling {N_SAMPLES}')

        # Train
        abs_coords = torch.cat(list(samples.values()), dim=0)
        gt = torch.cat(list(gts.values()), dim=0)
        print('abs_coords', abs_coords.shape)
        rel_coords = abs2rel(abs_coords).to(dev).to(typ)
        qf = sample_features3d(feats, make_3d(rel_coords), mode='bilinear').squeeze()
        print('qf', qf.shape)
        print('gt', gt.shape)
        step = 0
        for i in range(args.num_epochs):
            rp = torch.randperm(qf.size(0))
            shuffled_qf = qf[rp]
            shuffled_gt = gt[rp]
            for inp, targ in zip(shuffled_qf.split(args.bs), shuffled_gt.split(args.bs)):
                opt.zero_grad()
                pred = head(inp)
                loss = F.binary_cross_entropy_with_logits(pred, targ)
                loss.backward()
                opt.step()
                step += 1
                print(f'{sample.__name__}: Step {step:03d}   Loss: {loss.item():.3f}')
            sched.step()

        # Infer
        with torch.no_grad():
            pred = head(feats.permute(1,2,3,0))
            ma, ama = pred.max(dim=-1)
            mask = torch.sigmoid(ma) > 0.5
            result = torch.zeros_like(ama).to(dtype=torch.uint8)
            result[mask] = (ama+1)[mask].to(torch.uint8)
            accuracy = [f'{a:.2f}' for a in acc(result, label_lores).tolist()]
            iouu =     [f'{a:.2f}' for a in iou(result, label_lores).tolist()]
            print('='*60)
            print(f'{sample.__name__}:  \n Accuracy: {accuracy} \n      IoU: {iouu}')

            np.save(DATA_DIR / f'pred_{sample.__name__}{args.num_samples}.npy', result.cpu().numpy())
