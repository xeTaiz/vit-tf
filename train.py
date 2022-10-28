import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

from torchvtk.utils import make_5d

from rle_shit import decode_from_annotation

import wandb
from argparse import ArgumentParser

from utils import *

pltkwargs = {
    'dpi':  200,
    'tight_layout': True
}

def create_cnn(in_dim):
    return nn.Sequential(
        nn.GroupNorm(in_dim, in_dim),
        nn.Conv3d(in_dim, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.GroupNorm(8, 8),
        nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.GroupNorm(8, 16),
        nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
    )

if __name__ == '__main__':
    parser = ArgumentParser('Contrastive TF Design')
    parser.add_argument('data', type=str, help='Path to Data with {vol, mask, labels} keys in .pt file')
    parser.add_argument('--background-class', type=str, default='background', help='Name of the background class')
    parser.add_argument('--vol-scaling-factor', type=float, default=0.25, help='Scaling factor to reduce spatial resolution of volumes')
    parser.add_argument('--pos-encoding', action='store_true', help='Use positional encoding with input (3D coordinate)')
    parser.add_argument('--raw-data', action='store_true', help='Use raw data and do not normalize input to 0-mean 1-std')
    parser.add_argument('--fp32', action='store_true', help='Use full 32-bit precision')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for optimization')
    parser.add_argument('--iterations', type=int, default=10001, help='Number of optimization steps')
    parser.add_argument('--samples-per-iteration', type=int, default=32, help='Number of samples per class used in each iteration')
    parser.add_argument('--label-percentage', type=float, default=1.0, help='Percentage of labels to use for optimization')
    parser.add_argument('--wandb-tags', type=str, nargs='*', help='Additional tags to use for W&B')
    args = parser.parse_args()
    
    # Setup
    BG_CLASS = args.background_class
    FP16 = not args.fp32
    DOWNSAMPLE = args.vol_scaling_factor != 1.0
    NORMALIZE = not args.raw_data
    POS_ENCODING = args.pos_encoding

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    typ = torch.float16 if FP16 else torch.float32
    ONE = torch.ones(1, device=dev)

    # Data
    data = torch.load(args.data)
    if DOWNSAMPLE:
        vol  = F.interpolate(make_5d(data['vol']).float(), scale_factor=args.vol_scaling_factor, mode='nearest').squeeze()
        mask = F.interpolate(make_5d(data['mask']),        scale_factor=args.vol_scaling_factor, mode='nearest').squeeze()
    else:
        vol  = data['vol']
        mask = data['mask']
    vol_u8 = ((vol - vol.min()) * 255.0 / (vol.max() - vol.min())).cpu().numpy().astype(np.uint8)
    num_classes = len(data['labels'])
    label_dict = {i: n for i,n in enumerate(data['labels'])}

    if args.label_percentage < 1.0:
        non_bg_indices = [(mask == i).nonzero() for i in range(len(data['labels']))]
        # Choose 1 - label_pct non-bg samples to set to background
        to_drop = torch.cat([clas_idcs[torch.multinomial(
            ONE.expand(clas_idcs.size(0)), 
            int((1.0 - args.label_percentage) * clas_idcs.size(0))
        )] for clas_idcs in non_bg_indices if clas_idcs.size(0) > 0], dim=0)
        mask_reduced = mask.clone()
        mask_reduced[split_squeeze3d(to_drop)] = 0
    else:
        mask_reduced = mask

    class_indices = {
        n: (mask_reduced == i).nonzero().to(dev)
        for i, n in enumerate(data['labels'])
    }

    if NORMALIZE:
        vol = ((vol - vol.mean()) / vol.std()).to(typ).to(dev)
    else:
        vol = vol.to(typ).to(dev)

    if POS_ENCODING:
        x = torch.linspace(-1, 1, vol.size(-1))
        y = torch.linspace(-1, 1, vol.size(-2))
        z = torch.linspace(-1, 1, vol.size(-3))
        z, y, x = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack((z, y, x)) * 1.7185
        vol = torch.cat([vol[None], coords.to(typ).to(dev)], dim=0)
    else:
        vol = vol[None]

    model = create_cnn(in_dim=vol.size(0)).to(dev)

    sample_idxs = { n: l.size(0) for n, l in class_indices.items() }
    print('Volume Indices for each class:')
    pprint.pprint({n: v.shape for n, v in class_indices.items()})
    print('Number of annotations per class:')
    pprint.pprint(sample_idxs)

    # Dictionary that maps class to indices that are of a DIFFERENT class, for picking negatives
    different_class_indices = { n: torch.cat([v for m,v in class_indices.items() if m != n], dim=0) for n in class_indices.keys() }
    different_sample_idxs = { n: v.size(0) for n,v in different_class_indices.items() }
    print('Indices for negative samples for each class:')
    pprint.pprint({n: v.shape for n, v in different_class_indices.items()})
    print('Number of negative samples for each class:')
    pprint.pprint(different_sample_idxs)

    tags = [f'{args.label_percentage} Labels', 'RawData' if args.raw_data else 'NormalizedData', *args.wandb_tags]
    wandb.init(project='ntf', entity='viscom-ulm', tags=tags, config=vars(args))
    wandb.watch(model)

    # Training
    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for i in range(args.iterations):
        opt.zero_grad()
        # 1 Feed volume thru networks
        with torch.autocast('cuda', enabled=True, dtype=typ):
            features = model(make_5d(vol))
            q = F.normalize(features, dim=1)
        # 2 Choose samples from classes
        BS, FS = q.size(0), q.size(1)
        loss = 0.0
        for _ in range(args.samples_per_iteration):
            pos_samples = { # Pick samples_per_class indices to `class_indices`
                n: torch.multinomial(ONE.expand(v), 2)
                for n,v in sample_idxs.items() if v >= 2 and n != BG_CLASS
            }
            neg_samples = { # Pick samples_per_class indices to `class_indices`
                n: torch.multinomial(ONE.expand(v), int(2**16))
                for n,v in different_sample_idxs.items() if n != BG_CLASS
            }
            # 3 Sample samples
            for n, pos_idx in pos_samples.items():
                neg_idx = neg_samples[n]
                pos = q[split_squeeze(          class_indices[n][pos_idx], bs=BS, f=FS)]
                neg = q[split_squeeze(different_class_indices[n][neg_idx], bs=BS, f=FS)]
                sim = torch.einsum('bfp,bfn->bpn', [pos[..., [0]], torch.cat([pos[..., [1]], neg], dim=-1)]).squeeze(1)
                labels = torch.zeros(sim.size(0), dtype=torch.long, device=dev)
                loss += F.cross_entropy(sim, labels)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # Validate
        with torch.no_grad():
            log_dict = {'Charts/loss': loss}
            temp_cluster_center_cos = torch.stack([        q[split_squeeze(v, bs=BS, f=FS)].mean(dim=(0,2)) for n,v in class_indices.items() ])
            temp_cluster_center_l2  = torch.stack([ features[split_squeeze(v, bs=BS, f=FS)].mean(dim=(0,2)) for n,v in class_indices.items() ])
            if i > 0:
                cos_dist_traveled = 1.0 - torch.einsum('nf,mf->nm', [cluster_center_cos, temp_cluster_center_cos]).diagonal()
                l2_dist_traveled = F.pairwise_distance(temp_cluster_center_l2, cluster_center_l2)
            else:
                cos_dist_traveled, l2_dist_traveled = torch.zeros(temp_cluster_center_cos.size(0)), torch.zeros(temp_cluster_center_cos.size(0))
            # Update "old" cluster centers with current "temp" ones
            cluster_center_l2 = temp_cluster_center_l2
            cluster_center_cos = temp_cluster_center_cos

            log_dict.update({
                    f'Charts/Logits_dist_traveled/{n}': l2_dist_traveled[i].cpu()
                    for i, n in enumerate(class_indices.keys()) }
                )
            log_dict.update({
                    f'Charts/Cosine_dist_traveled/{n}': cos_dist_traveled[i].cpu()
                    for i, n in enumerate(class_indices.keys()) }
                )

            if i % 100 == 0:
                # Similarity / ClusterDistance matrices
                cos_sim = similarity_matrix(cluster_center_cos.float(), 'cosine')
                l2_sim  = similarity_matrix(cluster_center_l2.float(),   'l2')
                cos_sim_fig = plot_similarity_matrix(cos_sim.cpu(), data['labels'], 'cosine')
                l2_sim_fig  = plot_similarity_matrix(l2_sim.cpu(),  data['labels'], 'l2')
                log_dict.update({
                    'plots/similarity/cosine': cos_sim_fig,
                    'plots/similarity/logits': l2_sim_fig
                })

                # Cluster standard deviations
                log_dict.update({
                        f'Charts/std/cosine/{n}':         q[split_squeeze(v, bs=BS, f=FS)].std().cpu()
                        for n,v in class_indices.items() }
                    )
                log_dict.update({
                        f'Charts/std/logits/{n}': features[split_squeeze(v, bs=BS, f=FS)].std().cpu()
                        for n,v in class_indices.items() }
                    )

                # K-Means clustering
                pred_logits = cluster_kmeans(features, num_classes)
                pred_cosine = cluster_kmeans(q, num_classes)
                log_dict.update({
                    'plots/kmeans/logits/z': wandb.Image(vol_u8[62], masks={
                        'predictions':  {'mask_data': pred_logits[62] },
                        'ground_truth': {'mask_data': mask[62].numpy(), 'class_labels': label_dict}
                    }),
                    'plots/kmeans/logits/y': wandb.Image(vol_u8[:, 62], masks={
                        'predictions':  {'mask_data': pred_logits[:, 62] },
                        'ground_truth': {'mask_data': mask[:, 62].numpy(), 'class_labels': label_dict}
                    }),
                    'plots/kmeans/logits/x': wandb.Image(vol_u8[:, :, 62], masks={
                        'predictions':  {'mask_data': pred_logits[:, :, 62] },
                        'ground_truth': {'mask_data': mask[:, :, 62].numpy(), 'class_labels': label_dict}
                    }),
                    'plots/kmeans/cosine/z': wandb.Image(vol_u8[62], masks={
                        'predictions':  { 'mask_data': pred_cosine[62] },
                        'ground_truth': { 'mask_data': mask[62].numpy(), 'class_labels': label_dict }
                    }),
                    'plots/kmeans/cosine/y': wandb.Image(vol_u8[:, 62], masks={
                        'predictions':  { 'mask_data': pred_cosine[:, 62] },
                        'ground_truth': { 'mask_data': mask[:, 62].numpy(), 'class_labels': label_dict }
                    }),
                    'plots/kmeans/cosine/x': wandb.Image(vol_u8[:, :, 62], masks={
                        'predictions':  { 'mask_data': pred_cosine[:, :, 62] },
                        'ground_truth': { 'mask_data': mask[:, :, 62].numpy(), 'class_labels': label_dict }
                    })
                })

                # PCA Visualization
                pcs_logits = project_pca(features)
                pcs_cosine = project_pca(q)
                log_dict.update({
                    'plots/pca/logits/z': wandb.Image(pcs_logits[62]),
                    'plots/pca/logits/y': wandb.Image(pcs_logits[:, 62]),
                    'plots/pca/logits/x': wandb.Image(pcs_logits[:, :, 62]),
                    'plots/pca/cosine/z': wandb.Image(pcs_cosine[62]),
                    'plots/pca/cosine/y': wandb.Image(pcs_cosine[:, 62]), 
                    'plots/pca/cosine/x': wandb.Image(pcs_cosine[:, :, 62])
                })

            wandb.log(log_dict)
            plt.close(cos_sim_fig)
            plt.close(l2_sim_fig)
