import pprint
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex
import os

from torchvtk.utils import make_4d, make_5d

from rle_shit import decode_from_annotation

import wandb
from argparse import ArgumentParser

from utils import *
from semisparseconv import gather_receiptive_fields

pltkwargs = {
    'dpi':  200,
    'tight_layout': True
}

class CenterCrop(nn.Module):
    def __init__(self, ks=3):
        super().__init__()
        self.pad = ks // 2

    def forward(self, x):
        i = self.pad
        out = x[..., i:-i, i:-i, i:-i]
        return out

class PrintLayer(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        print(x.shape)
        return x

def conv_layer(n_in, n_out, Norm, Act):
    return nn.Sequential(
        nn.Conv3d(n_in, n_out, kernel_size=3, stride=1, padding=0),
        #Norm(n_out // 4, n_out),
        #CenterCrop(),
        Act(inplace=True)
    )

def create_cnn(in_dim, n_features=[8, 32, 16], Act=nn.Mish, Norm=nn.GroupNorm):
    feats = [in_dim] + n_features[:-1]
    layers = [conv_layer(n_in, n_out, Norm=Norm, Act=Act)
        for n_in, n_out in zip(feats, feats[1:])]
    last = nn.Conv3d(n_features[-2], n_features[-1], kernel_size=3, stride=1, padding=0)
    return nn.Sequential(*layers, last)

if __name__ == '__main__':
    parser = ArgumentParser('Contrastive TF Design')
    parser.add_argument('data', type=str, help='Path to Data with {vol, mask, labels} keys in .pt file')
    parser.add_argument('--background-class', type=str, default='background', help='Name of the background class')
    parser.add_argument('--vol-scaling-factor', type=float, default=0.25, help='Scaling factor to reduce spatial resolution of volumes')
    parser.add_argument('--no-pos-encoding', action='store_true', help='Use positional encoding with input (3D coordinate)')
    parser.add_argument('--raw-data', action='store_true', help='Use raw data and do not normalize input to 0-mean 1-std')
    parser.add_argument('--fp32', action='store_true', help='Use full 32-bit precision')
    parser.add_argument('--learning-rate', type=float, default=5e-3, help='Learning rate for optimization')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--lr-schedule', type=str, default='onecycle', help='Learning rate schedule')
    parser.add_argument('--iterations', type=int, default=10001, help='Number of optimization steps')
    parser.add_argument('--samples-per-iteration', type=int, default=32, help='Number of samples per class used in each iteration')
    parser.add_argument('--label-percentage', type=float, default=1.0, help='Percentage of labels to use for optimization')
    parser.add_argument('--wandb-tags', type=str, nargs='*', help='Additional tags to use for W&B')
    parser.add_argument('--no-validation', action='store_true', help='Skip validation')
    parser.add_argument('--cnn-layers', type=int, nargs='*', help='Number of features per CNN layer')
    parser.add_argument('--debug', action='store_true', help='Turn of WandB, some more logs')
    args = parser.parse_args()

    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # Setup
    BG_CLASS = args.background_class
    NEG_COUNT = int(2**15)
    FP16 = not args.fp32
    DOWNSAMPLE = args.vol_scaling_factor != 1.0
    NORMALIZE = not args.raw_data
    POS_ENCODING = not args.no_pos_encoding
    if args.lr_schedule.lower() == 'onecycle':
        LR_SCHED = partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=args.learning_rate, total_steps=args.iterations)
    elif args.lr_schedule.lower() == 'cosine':
        LR_SCHED = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=args.iterations)
    else:
        LR_SCHED = partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0)

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
    IDX = min(vol.shape[-3:])//2
    vol_u8 = ((vol - vol.min()) * 255.0 / (vol.max() - vol.min())).cpu().numpy().astype(np.uint8)
    num_classes = len(data['labels'])
    label_dict = {i: n for i,n in enumerate(data['labels'])}
    label2idx =  {n: i for i,n in label_dict.items()}

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

    to_remove = [(i,n) for i,n in enumerate(class_indices.keys()) if class_indices[n].size(0) < 3]
    for i,n in to_remove: # Remove classes with too few samples
        del class_indices[n]
        del label_dict[i]
        del label2idx[n]
        num_classes -= 1

    if NORMALIZE:
        vol = ((vol - vol.mean()) / vol.std()).to(typ).to(dev)
    else:
        vol = vol.to(typ).to(dev)

    if POS_ENCODING:
        x = torch.linspace(-1, 1, vol.size(-1), device=dev, dtype=typ)
        y = torch.linspace(-1, 1, vol.size(-2), device=dev, dtype=typ)
        z = torch.linspace(-1, 1, vol.size(-3), device=dev, dtype=typ)
        z, y, x = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack((z, y, x)) * 1.7185
        vol = torch.cat([vol[None], coords], dim=0)
    else:
        vol = vol[None]

    args.cnn_layers = args.cnn_layers if args.cnn_layers else [8, 32, 16]
    NF = args.cnn_layers[-1]
    model = create_cnn(in_dim=vol.size(0), n_features=args.cnn_layers).to(dev)
    REC_FIELD = len(args.cnn_layers) * 2 + 1

    tags = [f'{args.label_percentage} Labels', 'RawData' if args.raw_data else 'NormalizedData', *args.wandb_tags]
    wandb.init(project='ntf', entity='viscom-ulm', tags=tags, config=vars(args), mode='offline' if args.debug else 'online')
    wandb.watch(model)
    print(model)
    jaccard = JaccardIndex(num_classes=num_classes, average=None)
    best_logit_iou, best_cosine_iou, best_mlp_iou = torch.zeros(num_classes), torch.zeros(num_classes), torch.zeros(num_classes)

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

    close_plot = False

    # Training
    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = LR_SCHED(opt)

    for i in range(args.iterations):
        # Draw voxel samples to work with
        with torch.no_grad():
            pos_samples = { # Pick samples_per_class indices to `class_indices`
                n: torch.multinomial(ONE.expand(v), 2)
                for n,v in sample_idxs.items() if v >= 2 and n != BG_CLASS
            }
            neg_samples = { # Pick samples_per_class indices to `class_indices`
                n: torch.multinomial(ONE.expand(v), NEG_COUNT)
                for n,v in different_sample_idxs.items() if n != BG_CLASS
            }
            pos_crops = gather_receiptive_fields(make_4d(vol), torch.cat(list({n:           class_indices[n][v] for n,v in pos_samples.items()}.values()), dim=0), ks=REC_FIELD) # (C*2, 1, Z,Y,X)
            neg_crops = gather_receiptive_fields(make_4d(vol), torch.cat(list({n: different_class_indices[n][v] for n,v in neg_samples.items()}.values()), dim=0), ks=REC_FIELD) # (C*N, 1, Z,Y,X)
            print('Crops: ', pos_crops.shape, neg_crops.shape)
            crops = torch.cat([pos_crops, neg_crops], dim=0) # (C*2 + C*N, IN, Z,Y,X)
        opt.zero_grad()
        # 1 Feed volume thru networks
        with torch.autocast('cuda', enabled=True, dtype=typ):
            features = model(crops).squeeze() # (C*2 + C*N, F, 1,1,1)
            pos_feat = features[:2*(num_classes-1) ].reshape(num_classes-1, 2, NF)         # ((C-1)*2, F)
            neg_feat = features[ 2*(num_classes-1):].reshape(num_classes-1, NEG_COUNT, NF) # ( C*N,    F)
            print('pos_feat: ', pos_feat.shape)
            print('neg_feat: ', neg_feat.shape)
            pos_q, neg_q = F.normalize(pos_feat, dim=-1), F.normalize(neg_feat, dim=-1)
        # 2 Choose samples from classes
        if args.debug:
            print('pos_samples')
            pprint.pprint({k: v.shape for k,v in pos_samples.items()})
            print('neg_samples')
            pprint.pprint({k: v.shape for k,v in neg_samples.items()})
            print('pos_indices')
            pprint.pprint({k: class_indices[k][v].shape for k,v in pos_samples.items()})
        # Compute InfoNCE
        sim = torch.einsum('cpf,cnf->cpn', [pos_q[:,[0]], torch.cat([pos_q[:,[1]], neg_q], dim=1)]).squeeze(1)
        labels = torch.zeros(sim.size(0), dtype=torch.long, device=dev)
        loss = F.cross_entropy(sim, labels)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        # Validate
        with torch.no_grad():
            log_dict = {
                'Charts/loss': loss,
                'Charts/learning_rate': sched.get_last_lr()[0]
            }
            # Compute mean of normalized feature vectors per class -> cluster centers
            temp_cluster_center_cos = pos_q.mean(dim=1)
            # Magnitude of mean of normalized feature vectors is distance from 0, should get close to 1 for a decisive cluster
            cluster_center_cos_magnitude = torch.norm(temp_cluster_center_cos, dim=1)
            # Re-normalize to get a normalized feature vector as cluster center
            temp_cluster_center_cos = F.normalize(temp_cluster_center_cos, dim=1)
            # Cluster centers in logit space are just means of logit features
            temp_cluster_center_l2  = pos_feat.mean(dim=1)
            if i > 0: # Distance traveled as cosine distance between old and new cluster center or p-2 norm between old and new
                cos_dist_traveled = 1.0 - torch.einsum('nf,mf->nm', [cluster_center_cos, temp_cluster_center_cos]).diagonal()
                l2_dist_traveled = F.pairwise_distance(temp_cluster_center_l2, cluster_center_l2)
            else:
                cos_dist_traveled, l2_dist_traveled = torch.zeros(temp_cluster_center_cos.size(0)), torch.zeros(temp_cluster_center_cos.size(0))
            # Update "old" cluster centers with current "temp" ones
            cluster_center_l2 = temp_cluster_center_l2
            cluster_center_cos = temp_cluster_center_cos

            # Cluster standard deviations
            log_dict.update({
                f'StdDev_Cos/{n}': torch.mean((pos_q[2*i:2*i+2].mean(dim=0) - temp_cluster_center_cos[i])**2, dim=0).cpu()
                for i,n in enumerate(class_indices.keys()) }
            )
            log_dict.update({
                f'StdDev_Logits/{n}': torch.mean((pos_feat[2*i:2*i+2].mean(dim=0) - temp_cluster_center_cos[i])**2, dim=0).cpu()
                for i,n in enumerate(class_indices.keys()) }
            )

            log_dict.update({
                    f'CC_Logits_dist_traveled/{n}': l2_dist_traveled[i].cpu()
                    for i, n in enumerate(class_indices.keys()) }
                )
            log_dict.update({
                    f'CC_Cosine_dist_traveled/{n}': cos_dist_traveled[i].cpu()
                    for i, n in enumerate(class_indices.keys()) }
                )


            if (i == args.iterations-1) or (i % 100 == 0 and not args.no_validation):
                full_feats = model(make_5d(vol))
                full_qs = F.normalize(full_feats, dim=1)
                # Distance to cluster centers
                l2_center_distances = torch.pow(full_feats - cluster_center_l2[:,:,None,None,None].expand(-1, -1, 1, 1, 1), 2.0).sum(dim=1).sqrt()
                # Get closest (i.e. segmentation) cluster center class
                l2_closest = l2_center_distances.argmin(dim=0)
                # Distance to cluster center map
                l2_distance_map = torch.exp(-l2_center_distances)
                # Cosine distances as distance map
                cos_center_distances = torch.clamp(torch.einsum('fdhw,nf->ndhw', (full_qs.squeeze(0), cluster_center_cos)), 0, 1)
                # Get (cosine distance) closest cluster center segmentation
                cos_closest = cos_center_distances.argmax(dim=0)
                # Compute IoUs
                l2_iou = jaccard(l2_closest.cpu(), mask)
                jaccard.reset()
                best_logit_iou = torch.stack([best_logit_iou, l2_iou], dim=0).max(0).values
                cos_iou = jaccard(cos_closest.cpu(), mask)
                jaccard.reset()
                best_cosine_iou = torch.stack([best_cosine_iou, cos_iou], dim=0).max(0).values
                log_dict.update({
                    f'IoU_logits/{n}': v for n,v in zip(label_dict.values(), l2_iou)
                })
                log_dict.update({
                    f'IoU_cosine/{n}': v for n, v in zip(label_dict.values(), cos_iou)
                })
                log_dict.update({
                    f'BestIoU_cosine/{n}': v for n,v in zip(label_dict.values(), best_cosine_iou)
                })
                log_dict.update({
                    f'BestIoU_logits/{n}': v for n, v in zip(label_dict.values(), best_logit_iou)
                })
                log_dict.update({
                    f'Plots_Dist_L2/{n}/z': wandb.Image(l2_distance_map[i, IDX].cpu().float().numpy())
                    for i, n in enumerate(class_indices.keys())
                })
                log_dict.update({
                    f'Plots_Dist_L2/{n}/y': wandb.Image(l2_distance_map[i, :, IDX].cpu().float().numpy())
                    for i, n in enumerate(class_indices.keys())
                })
                log_dict.update({
                    f'Plots_Dist_L2/{n}/x': wandb.Image(l2_distance_map[i, :, :, IDX].cpu().float().numpy())
                    for i, n in enumerate(class_indices.keys())
                })
                log_dict.update({
                    f'Plots_Dist_Cos/{n}/z':
                    wandb.Image(cos_center_distances.cpu()[i, IDX])
                    for i, n in enumerate(class_indices.keys())
                })
                log_dict.update({
                    f'Plots_Dist_Cos/{n}/y':
                    wandb.Image(cos_center_distances.cpu()[i, :, IDX])
                    for i, n in enumerate(class_indices.keys())
                })
                log_dict.update({
                    f'Plots_Dist_Cos/{n}/x':
                    wandb.Image(cos_center_distances.cpu()[i, :, :, IDX])
                    for i, n in enumerate(class_indices.keys())
                })
                log_dict['Plots_Seg_L2dist/z'] = wandb.Image(vol_u8[IDX], masks={
                    'predictions':  {'mask_data': l2_closest[IDX].cpu().numpy(),   'class_labels': label_dict },
                    'ground_truth': {'mask_data': mask[IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg_L2dist/y'] = wandb.Image(vol_u8[:, IDX], masks={
                    'predictions':  {'mask_data': l2_closest[:, IDX].cpu().numpy(),   'class_labels': label_dict },
                    'ground_truth': {'mask_data': mask[:, IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg_L2dist/x'] = wandb.Image(vol_u8[:, :, IDX], masks={
                    'predictions':  {'mask_data': l2_closest[:, :, IDX].cpu().numpy(),   'class_labels': label_dict },
                    'ground_truth': {'mask_data': mask[:, :, IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg_CosDist/z'] = wandb.Image(vol_u8[IDX], masks={
                    'predictions':  {'mask_data': cos_closest[IDX].cpu().numpy(),  'class_labels': label_dict },
                    'ground_truth': {'mask_data': mask[IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg_CosDist/y'] = wandb.Image(vol_u8[:, IDX], masks={
                    'predictions':  {'mask_data': cos_closest[:, IDX].cpu().numpy(),  'class_labels': label_dict },
                    'ground_truth': {'mask_data': mask[:, IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg_CosDist/x'] = wandb.Image(vol_u8[:, :, IDX], masks={
                    'predictions':  {'mask_data': cos_closest[:, :, IDX].cpu().numpy(),  'class_labels': label_dict },
                    'ground_truth': {'mask_data': mask[:, :, IDX].numpy(), 'class_labels': label_dict }
                })

                # Similarity / ClusterDistance matrices
                cos_sim = similarity_matrix(cluster_center_cos.float(), 'cosine')
                l2_sim  = similarity_matrix(cluster_center_l2.float(),   'l2')
                cos_sim_fig = plot_similarity_matrix(cos_sim.cpu(), list(label_dict.values()), 'cosine')
                l2_sim_fig  = plot_similarity_matrix(l2_sim.cpu(),  list(label_dict.values()), 'l2')
                close_plot = True
                log_dict.update({
                    'Plots_CC_similarity/cosine': cos_sim_fig,
                    'Plots_CC_similarity/logits': l2_sim_fig
                })

                # K-Means clustering
                pred_logits = cluster_kmeans(features, num_classes)
                pred_cosine = cluster_kmeans(q, num_classes)
                log_dict.update({
                    'Plots_Seg_Kmeans/logits/z': wandb.Image(vol_u8[IDX], masks={
                        'predictions':  {'mask_data': pred_logits[IDX] },
                        'ground_truth': {'mask_data': mask[IDX].numpy(), 'class_labels': label_dict}
                    }),
                    'Plots_Seg_Kmeans/logits/y': wandb.Image(vol_u8[:, IDX], masks={
                        'predictions':  {'mask_data': pred_logits[:, IDX] },
                        'ground_truth': {'mask_data': mask[:, IDX].numpy(), 'class_labels': label_dict}
                    }),
                    'Plots_Seg_Kmeans/logits/x': wandb.Image(vol_u8[:, :, IDX], masks={
                        'predictions':  {'mask_data': pred_logits[:, :, IDX] },
                        'ground_truth': {'mask_data': mask[:, :, IDX].numpy(), 'class_labels': label_dict}
                    }),
                    'Plots_Seg_Kmeans/cosine/z': wandb.Image(vol_u8[IDX], masks={
                        'predictions':  { 'mask_data': pred_cosine[IDX] },
                        'ground_truth': { 'mask_data': mask[IDX].numpy(), 'class_labels': label_dict }
                    }),
                    'Plots_Seg_Kmeans/cosine/y': wandb.Image(vol_u8[:, IDX], masks={
                        'predictions':  { 'mask_data': pred_cosine[:, IDX] },
                        'ground_truth': { 'mask_data': mask[:, IDX].numpy(), 'class_labels': label_dict }
                    }),
                    'Plots_Seg_Kmeans/cosine/x': wandb.Image(vol_u8[:, :, IDX], masks={
                        'predictions':  { 'mask_data': pred_cosine[:, :, IDX] },
                        'ground_truth': { 'mask_data': mask[:, :, IDX].numpy(), 'class_labels': label_dict }
                    })
                })

                # PCA Visualization
                pcs_logits = project_pca(features)
                pcs_cosine = project_pca(q)
                log_dict.update({
                    'Plots_Feat/pca/logits/z': wandb.Image(pcs_logits[IDX]),
                    'Plots_Feat/pca/logits/y': wandb.Image(pcs_logits[:, IDX]),
                    'Plots_Feat/pca/logits/x': wandb.Image(pcs_logits[:, :, IDX]),
                    'Plots_Feat/pca/cosine/z': wandb.Image(pcs_cosine[IDX]),
                    'Plots_Feat/pca/cosine/y': wandb.Image(pcs_cosine[:, IDX]),
                    'Plots_Feat/pca/cosine/x': wandb.Image(pcs_cosine[:, :, IDX])
                })

            wandb.log(log_dict)
            if close_plot:
                plt.close(cos_sim_fig)
                plt.close(l2_sim_fig)
