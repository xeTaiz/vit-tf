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
    while idxs.size(0) > int(2**24): 
        print(f'Class has > 2**24 options: {idxs.size(0)} -> {idxs.size(0)//2}')
        idxs = idxs[::2]
    return idxs[torch.multinomial(ONE.expand(idxs.size(0)), n_samples)]

def sample_surface(vol, n_samples, dist_from_surface=4):
    stel1 = generate_binary_structure(rank=3, connectivity=dist_from_surface)
    stel2 = generate_binary_structure(rank=3, connectivity=1)
    outer = binary_erosion(vol, stel1)
    inner = binary_erosion(outer, stel2)
    print('outer', outer.sum(), tuple(map(lambda c: (c.min(), c.max()), outer.nonzero())))
    print('inner', inner.sum(), tuple(map(lambda c: (c.min(), c.max()), inner.nonzero())))

    surface_idxs = torch.from_numpy(np.logical_xor(inner, outer)).nonzero()
    while surface_idxs.size(0) > int(2**24): 
        print(f'Class has > 2**24 options: {surface_idxs.size(0)} -> {surface_idxs.size(0)//2}')
        surface_idxs = surface_idxs[::2]
    if surface_idxs.size(0) > n_samples:
        return surface_idxs[torch.multinomial(ONE.expand(surface_idxs.size(0)), n_samples)]
    else:
        print(f'Full surface only has {surface_idxs.size(0)} voxels (< n_samples={n_samples}).')
        return surface_idxs

def sample_both(vol, n_samples, dist_from_surface=4):
    return torch.cat([sample_uniform(vol, n_samples//2), sample_surface(vol, n_samples//2, dist_from_surface=dist_from_surface)])

if __name__ == '__main__':
    parser = ArgumentParser('Compare similarity maps for different samplings of ground truth segmentations')
    parser.add_argument('--num-samples', type=float, default=4096, help='Number of samples. if < 1 it uses fraction')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--bs', type=int, default=256, help='Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight Decay')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')
    parser.add_argument('--mlp', type=str, default='', help='Additional feature map sizes comma-separated, i.e. "32,64"')
    args = parser.parse_args()
    vol = torch.from_numpy(np.load(DATA_DIR / 'volume-10.npy', allow_pickle=True))
    feats = torch.from_numpy(np.load(DATA_DIR / 'volume-10.nii_DINOfeats_all.npy', allow_pickle=True)[()]['k'])
    label = np.load(DATA_DIR / 'labels-10.npy', allow_pickle=True) 
    label_lores = F.interpolate(make_5d(torch.from_numpy(label)), size=feats.shape[-3:], mode='nearest').squeeze()
    num_classes_with_bg = int(label.max()+1)
    num_classes_wo_bg = num_classes_with_bg - 1

    if len(args.mlp) == 0:
        extra_layers = []
    else:
        extra_layers = [int(args.mlp)] if ',' not in args.mlp else [int(s) for s in args.mlp.split(',')] 
    mlp_feats = [feats.size(0)] + extra_layers
    layers = [torch.nn.Sequential(
        torch.nn.Linear(nin, nout),
        torch.nn.ReLU(True)
    ) for nin, nout in zip(mlp_feats, mlp_feats[1:])]
    layers += [torch.nn.Linear(mlp_feats[-1], num_classes_with_bg)]
    head = torch.nn.Sequential(*layers)
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
        for i in range(num_classes_with_bg):
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
            gts[i] = torch.tensor([i]).expand(N_SAMPLES).long().to(dev)
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
                loss = F.cross_entropy(pred, targ)
                loss.backward()
                opt.step()
                step += 1
                if step % 50 == 0:
                    print(f'{sample.__name__}: Step {step:03d}   Loss: {loss.item():.3f}')
            sched.step()

        # Infer
        with torch.no_grad():
            pred = head(feats.permute(1,2,3,0))
            ma, ama = pred.max(dim=-1)
            mask = torch.sigmoid(ma) > 0.5
            result = torch.zeros_like(ama).to(dtype=torch.uint8)
            result[mask] = ama[mask].to(torch.uint8)
            accuracy = [f'{a:.2f}' for a in acc(result, label_lores).tolist()]
            iouu =     [f'{a:.2f}' for a in iou(result, label_lores).tolist()]
            print('='*60)
            print(f'{sample.__name__}:  \n Accuracy: {accuracy} \n      IoU: {iouu}')

            np.save(DATA_DIR / f'pred_{sample.__name__}{args.num_samples}_wbg.npy', result.cpu().numpy())
            for i, sim in enumerate(pred.permute(3,0,1,2).softmax(dim=0)):
                np.save(DATA_DIR / f'sim_{i}_{sample.__name__}{args.num_samples}_wbg.npy', sim.cpu().numpy())
