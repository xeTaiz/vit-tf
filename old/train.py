from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex

from torchvtk.utils import make_5d

import wandb
from argparse import ArgumentParser

from utils import *
from models import FeatureExtractor

if __name__ == '__main__':
    parser = ArgumentParser('Contrastive TF Design')
    parser.add_argument('--background-class', type=str, default='background', help='Name of the background class')
    parser.add_argument('--vol-scaling-factor', type=float, default=0.25, help='Scaling factor to reduce spatial resolution of volumes')
    parser.add_argument('--samples-per-iteration', type=int, default=32, help='Number of samples per class used in each iteration')
    parser.add_argument('--cnn-layers', type=str, help='Number of features per CNN layer')
    parser.add_argument('--linear-layers', type=str, help='Number of features for linear layers after convs (per voxel)')
    parser.add_argument('--residual', type=str, choices=['true', 'false'], default='false', help='Use skip connections in network')
    parser.add_argument('--lambda-std', type=float, default=1.0, help='Scales loss on cluster standard deviations.')
    parser.add_argument('--lambda-ce', type=float, default=1.0, help='Scales cross entropy loss')
    parse_basics(parser)
    args = parser.parse_args()
    setup_seed_and_debug(args)

    # Setup
    BG_CLASS = args.background_class
    DOWNSAMPLE = args.vol_scaling_factor != 1.0
    NORMALIZE = args.normalize == 'true'
    POS_ENCODING = args.pos_encoding == 'true'
    if args.lr_schedule.lower() == 'onecycle':
        LR_SCHED = partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=args.learning_rate, total_steps=args.iterations)
    elif args.lr_schedule.lower() == 'cosine':
        LR_SCHED = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=args.iterations)
    else:
        LR_SCHED = partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    typ = torch.float16 if args.fp16 == 'true' else torch.float32
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
    vol_u8 = (255.0 * norm_minmax(vol)).cpu().numpy().astype(np.uint8)
    num_classes = len(data['labels'])
    label_dict = {i: n for i,n in enumerate(data['labels'])}
    label2idx =  {n: i for i,n in label_dict.items()}

    if args.label_percentage < 1.0:
        non_bg_indices = [(mask == i).nonzero() for i in range(len(data['labels']))]
        # Choose 1 - label_pct non-bg samples to set to background
        to_drop = torch.cat([clas_idcs[torch.from_numpy(np.random.choice(clas_idcs.size(0),
            int((1.0 - args.label_percentage) * clas_idcs.size(0))
        ))] for clas_idcs in non_bg_indices if clas_idcs.size(0) > 0], dim=0)
        mask_reduced = mask.clone()
        mask_reduced[split_squeeze3d(to_drop)] = 0
    else:
        mask_reduced = mask

    class_indices = {
        n: (mask_reduced == i).nonzero().to(dev)
        for i, n in enumerate(data['labels'])
    }

    if NORMALIZE:
        vol = norm_mean_std(vol).to(typ).to(dev)
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

    args.cnn_layers    = [int(n.strip()) for n in    args.cnn_layers.replace('[', '').replace(']', '').split(' ')] if args.cnn_layers    else [8, 16, 32]
    args.linear_layers = [int(n.strip()) for n in args.linear_layers.replace('[', '').replace(']', '').split(' ')] if args.linear_layers else [32]
    NF = args.linear_layers[-1]
    model = FeatureExtractor(in_dim=vol.size(0), n_features=args.cnn_layers, n_linear=args.linear_layers, residual=args.residual == 'true').to(dev)
    REC_FIELD = len(args.cnn_layers) * 2 + 1
    cls_head = nn.Linear(args.cnn_layers[-1], num_classes).to(dev)

    group = 'Contrastive Dense'
    tags = [f'{args.label_percentage} Labels', 'NormalizedData' if args.normalize else 'RawData', *(args.wandb_tags if args.wandb_tags else [])]
    wandb.init(project='ntf', entity='viscom-ulm', tags=tags, config=vars(args), group=group)
    wandb.watch(model)
    ic(model)
    jaccard = JaccardIndex(num_classes=num_classes, average=None)
    best_logit_iou, best_cosine_iou, best_mlp_iou = torch.zeros(num_classes), torch.zeros(num_classes), torch.zeros(num_classes)
    
    sample_idxs = { n: l.size(0) for n, l in class_indices.items() }
    ic(sample_idxs)

    # Dictionary that maps class to indices that are of a DIFFERENT class, for picking negatives
    different_class_indices = { n: torch.cat([v for m,v in class_indices.items() if m != n], dim=0) for n in class_indices.keys() }
    different_sample_idxs = { n: v.size(0) for n,v in different_class_indices.items() }
    ic(different_sample_idxs)

    close_plot = False

    # Training
    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = LR_SCHED(opt)

    for i in range(args.iterations):
        opt.zero_grad()
        # 1 Feed volume thru networks
        with torch.autocast('cuda', enabled=True, dtype=typ):
            features = model(F.pad(make_5d(vol), tuple([REC_FIELD//2]*6)))
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
                if args.lambda_ce > 0:
                    cls_pred = cls_head(pos.permute(0,2,1)).view(-1, num_classes)
                    ce_loss = args.lambda_ce * F.cross_entropy(cls_pred, ONE.long().expand(cls_pred.size(0)) * label2idx[n])
                    loss += ce_loss

        if args.lambda_std > 0:
            cos_std = args.lambda_std * torch.stack([q[split_squeeze(v, bs=BS, f=FS)].std() for v in class_indices.values() if v.numel() > 0]).sum(0)
            loss += cos_std
        else:
            cos_std = torch.zeros(1)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        # Validate
        with torch.no_grad():
            log_dict = {
                'Charts/loss': loss,
                'Charts/std_loss': cos_std.cpu(),
                'Charts/ce_loss': ce_loss.cpu(),
                'Charts/learning_rate': sched.get_last_lr()[0]
            }
            # Compute mean of normalized feature vectors per class -> cluster centers
            temp_cluster_center_cos = torch.nan_to_num(torch.stack([ q[split_squeeze(v, bs=BS, f=FS)].mean(dim=(0,2)) for n,v in class_indices.items() ]))
            # Magnitude of mean of normalized feature vectors is distance from 0, should get close to 1 for a decisive cluster
            cluster_center_cos_magnitude = torch.norm(temp_cluster_center_cos, dim=1)
            # Re-normalize to get a normalized feature vector as cluster center
            temp_cluster_center_cos = F.normalize(temp_cluster_center_cos, dim=1)
            # Cluster centers in logit space are just means of logit features
            temp_cluster_center_l2  = torch.nan_to_num(torch.stack([ features[split_squeeze(v, bs=BS, f=FS)].mean(dim=(0,2)) for n,v in class_indices.items() ]))
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
                f'StdDev_Cos/{n}':         q[split_squeeze(v, bs=BS, f=FS)].std().cpu()
                for n,v in class_indices.items() }
            )
            log_dict.update({
                f'StdDev_Logits/{n}': features[split_squeeze(v, bs=BS, f=FS)].std().cpu()
                for n,v in class_indices.items() }
            )
            # Log magnitude of (cosine space) cluster center before normalization
            log_dict.update({
                f'CC_Cos_mag/{n}': cluster_center_cos_magnitude[i]
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

            # Distance to cluster centers
            l2_center_distances = torch.pow(features - cluster_center_l2[:,:,None,None,None].expand(-1, -1, 1, 1, 1), 2.0).sum(dim=1).sqrt()
            # Get closest (i.e. segmentation) cluster center class
            l2_closest = l2_center_distances.argmin(dim=0)
            # Distance to cluster center map
            l2_distance_map = torch.exp(-l2_center_distances)
            # Cosine distances as distance map
            cos_center_distances = torch.clamp(torch.einsum('fdhw,nf->ndhw', (q.squeeze(0), cluster_center_cos)), 0, 1)
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
                f'IoU_logits/{n}': v for n,v in zip(data['labels'], l2_iou)
            })
            log_dict.update({
                f'IoU_cosine/{n}': v for n, v in zip(data['labels'], cos_iou)
            })
            log_dict.update({
                f'BestIoU_cosine/{n}': v for n,v in zip(data['labels'], best_cosine_iou)
            })
            log_dict.update({
                f'BestIoU_logits/{n}': v for n, v in zip(data['labels'], best_logit_iou)
            })

            if (i == args.iterations-1) or (i % 100 == 0 and not args.no_validation):
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
                # MLP Predictions
                pred = cls_head(q.permute(0,2,3,4,1)).argmax(-1).cpu().squeeze()
                mlp_iou = jaccard(pred, mask)
                best_mlp_iou = torch.stack([best_mlp_iou, mlp_iou], dim=0).max(0).values
                log_dict.update({
                    f'IoU_MLP/{n}': v for n,v in zip(data['labels'], mlp_iou)
                })
                log_dict.update({
                    f'BestIoU_MLP/{n}': v for n,v in zip(data['labels'], best_mlp_iou)
                })

                log_dict['Plots_Seg_MLP/z'] = wandb.Image(vol_u8[:, :, IDX], masks={
                    'predictions': { 'mask_data': pred[:, :, IDX].numpy(), 'class_labels': label_dict },
                    'ground_truth': {'mask_data': mask[:, :, IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg_MLP/y'] = wandb.Image(vol_u8[:, IDX], masks={
                    'predictions': { 'mask_data': pred[:, IDX].numpy(), 'class_labels': label_dict },
                    'ground_truth': {'mask_data': mask[:, IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg_MLP/x'] = wandb.Image(vol_u8[IDX], masks={
                    'predictions': { 'mask_data': pred[IDX].numpy(), 'class_labels': label_dict },
                    'ground_truth': {'mask_data': mask[IDX].numpy(), 'class_labels': label_dict }
                })

                # Similarity / ClusterDistance matrices
                cos_sim = similarity_matrix(cluster_center_cos.float(), 'cosine')
                l2_sim  = similarity_matrix(cluster_center_l2.float(),   'l2')
                cos_sim_fig = plot_similarity_matrix(cos_sim.cpu(), data['labels'], 'cosine')
                l2_sim_fig  = plot_similarity_matrix(l2_sim.cpu(),  data['labels'], 'l2')
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
