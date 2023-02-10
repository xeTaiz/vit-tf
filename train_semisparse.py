from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex
from tqdm import trange

from torchvtk.utils import make_4d, make_5d

import wandb
from argparse import ArgumentParser

from utils import *
from semisparseconv import gather_receiptive_fields2 as gather_receiptive_fields
from models import FeatureExtractor

if __name__ == '__main__':
    parser = ArgumentParser('Contrastive TF Design (Sparse)')
    parser.add_argument('--background-class', type=str, default='background', help='Name of the background class')
    parser.add_argument('--vol-scaling-factor', type=float, default=0.25, help='Scaling factor to reduce spatial resolution of volumes')
    parser.add_argument('--label-scaling-factor', type=float, default=0.25, help='Scaling factor at which labels are given compared to full volume')
    parser.add_argument('--samples-per-iteration', type=int, default=8, help='Number of samples per class used in each iteration')
    parser.add_argument('--cnn-layers', type=str, help='Number of features per CNN layer')
    parser.add_argument('--linear-layers', type=str, help='Number of features for linear layers after convs (per voxel)')
    parser.add_argument('--residual', type=str, choices=['true', 'false'], default='false', help='Use skip connections in network')
    parser.add_argument('--lambda-std', type=float, default=0.0, help='Weighting of standard deviation loss')
    parse_basics(parser)
    args = parser.parse_args()
    setup_seed_and_debug(args)

    # Setup
    BS = args.samples_per_iteration
    BG_CLASS = args.background_class
    NEG_COUNT = int(2**16)
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
    assert args.vol_scaling_factor >= args.label_scaling_factor
    data = torch.load(args.data)
    vol  = F.interpolate(make_5d(data['vol']).float(),  scale_factor=args.vol_scaling_factor, mode='nearest').squeeze(0)
    mask = F.interpolate(make_5d(data['mask']).float(), scale_factor=args.vol_scaling_factor, mode='nearest').squeeze()
    # Get downsampled volume for validation
    lowres_vol  = F.interpolate(make_5d(data['vol']).float(), scale_factor=args.label_scaling_factor, mode='nearest').squeeze()
    lowres_mask = F.interpolate(make_5d(data['mask']),        scale_factor=args.label_scaling_factor, mode='nearest').squeeze()
    vol_u8 = (255.0 * norm_minmax(lowres_vol)).squeeze().cpu().numpy().astype(np.uint8)
    lowres_vol = lowres_vol.to(typ).to(dev)

    IDX = min(lowres_vol.shape[-3:]) // 2
    num_classes = len(data['labels'])
    IDX_UP = get_index_upscale_function(args.label_scaling_factor / args.vol_scaling_factor, device=dev)
    label_dict = {i: n for i,n in enumerate(data['labels'])}
    label2idx =  {n: i for i,n in label_dict.items()}

    # Drop labels from GT to simulate sparse annotations
    if args.label_percentage < 1.0:
        non_bg_indices = [(lowres_mask == i).nonzero() for i in range(len(data['labels']))]
        # Choose 1 - label_pct non-bg samples to set to background
        to_drop = torch.cat([clas_idcs[torch.multinomial(
            ONE.cpu().expand(clas_idcs.size(0)),
            int((1.0 - args.label_percentage) * clas_idcs.size(0))
        )] for clas_idcs in non_bg_indices if clas_idcs.size(0) > 0], dim=0)
        mask_reduced = lowres_mask.clone()
        mask_reduced[split_squeeze3d(to_drop)] = 0
    else:
        mask_reduced = lowres_mask

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
    C = num_classes -1

    if NORMALIZE:  # Input Normalization
        vol        = norm_mean_std(       vol.float()).to(typ).to(dev)
        lowres_vol = norm_mean_std(lowres_vol.float()).to(typ).to(dev)
    else:
        vol = vol.to(typ).to(dev)
        lowres_vol = lowres_vol.to(typ).to(dev)

    if POS_ENCODING:  # Positional Encoding
        x = torch.linspace(-1, 1, vol.size(-1), device=dev, dtype=typ)
        y = torch.linspace(-1, 1, vol.size(-2), device=dev, dtype=typ)
        z = torch.linspace(-1, 1, vol.size(-3), device=dev, dtype=typ)
        z, y, x = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack((z, y, x)) * 1.7185
        vol = torch.cat([make_4d(vol), coords], dim=0)
        lowres_coords = F.interpolate(make_5d(coords), scale_factor=args.label_scaling_factor / args.vol_scaling_factor, mode='nearest').squeeze(0)
        lowres_vol = torch.cat([make_4d(lowres_vol), lowres_coords], dim=0)
    else:
        vol = make_4d(vol)

    ic(vol_u8)
    ic(vol,)
    ic(lowres_vol)

    # Model
    args.cnn_layers    = [int(n.strip()) for n in    args.cnn_layers.replace('[', '').replace(']', '').split(' ')] if args.cnn_layers    else [8, 16, 32]
    args.linear_layers = [int(n.strip()) for n in args.linear_layers.replace('[', '').replace(']', '').split(' ')] if args.linear_layers else [32]
    NF = args.linear_layers[-1]
    model = FeatureExtractor(
        in_dim=vol.size(0), 
        n_features=args.cnn_layers, 
        n_linear=args.linear_layers, 
        residual=args.residual == 'true'
    ).to(dev)
    # model = create_cnn(in_dim=vol.size(0), n_features=args.cnn_layers, n_linear=args.linear_layers).to(dev)
    REC_FIELD = len(args.cnn_layers) * 2 + 1

    # Logging
    group = 'Contrastive Sparse'
    tags = [f'{args.label_percentage} Labels', 'NormalizedData' if args.normalize else 'RawData', 'SemiSparse', *(args.wandb_tags if args.wandb_tags else [])]
    wandb.init(project='ntf', entity='viscom-ulm', tags=tags, config=vars(args),
        mode='offline' if args.debug else 'online', group=group)
    wandb.watch(model)
    ic(model)
    print(f'Network uses receiptive field of {REC_FIELD}x{REC_FIELD}x{REC_FIELD}.')
    jaccard = JaccardIndex(num_classes=num_classes, average=None)
    best_logit_iou, best_cosine_iou, best_mlp_iou = torch.zeros(num_classes), torch.zeros(num_classes), torch.zeros(num_classes)

    sample_idxs = { n: l.size(0) for n, l in class_indices.items() }
    ic(sample_idxs)

    # Dictionary that maps class to indices that are of a DIFFERENT class, for picking negatives
    different_class_indices = { n: torch.cat([v for m,v in class_indices.items() if m != n], dim=0) for n in class_indices.keys() }
    different_sample_idxs = { n: v.size(0) for n,v in different_class_indices.items() }
    if args.debug:
        ic('Number of negative samples for each class:')
        ic(different_sample_idxs)

    close_plot = False

    # Training
    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = LR_SCHED(opt)

    for i in trange(args.iterations):
        # Draw voxel samples to work with
        with torch.no_grad():
            pos_samples = { # Pick samples_per_class indices to `class_indices`
                n: torch.multinomial(ONE.expand(v), BS*2)
                for n,v in sample_idxs.items() if v >= BS*2 and n != BG_CLASS
            }
            neg_samples = { # Pick samples_per_class indices to `class_indices`
                n: torch.multinomial(ONE.expand(v), NEG_COUNT)
                for n,v in different_sample_idxs.items() if n != BG_CLASS
            } #                                                             AABBCCDDEEFF   +    NNN....NNNOOO...OOOPPP...PPPQQQ...QQQ...  (NEG_COUNTxN  NEG_COUNTxO  ...)
            pos_crops = gather_receiptive_fields(make_4d(vol), torch.cat(list({n:           IDX_UP(class_indices[n][v]) for n,v in pos_samples.items()}.values()), dim=0), ks=REC_FIELD) # (C*2, 1, Z,Y,X)
            neg_crops = gather_receiptive_fields(make_4d(vol), torch.cat(list({n: IDX_UP(different_class_indices[n][v]) for n,v in neg_samples.items()}.values()), dim=0), ks=REC_FIELD) # (C*N, 1, Z,Y,X)
            crops = torch.cat([pos_crops, neg_crops], dim=0) # (C*2 + C*N, IN, Z,Y,X)
        opt.zero_grad()
        # 1 Feed volume thru networks
        with torch.autocast('cuda', enabled=True, dtype=typ):
            features = model(crops).squeeze() # (C*BS*2 + C*N, F, 1,1,1)
            pos_feat = features[:2*BS*C ].reshape(C, 2, BS, NF)
            neg_feat = features[ 2*BS*C:].reshape(C, NEG_COUNT, 1, NF)
            pos_q, neg_q = F.normalize(pos_feat, dim=-1), F.normalize(neg_feat, dim=-1)
        # 2 Choose samples from classes
        if args.debug:
            ic('pos_samples', {k: v.shape for k,v in pos_samples.items()})
            ic('neg_samples', {k: v.shape for k,v in neg_samples.items()})
            ic(pos_crops)
            ic(neg_crops)
            ic(pos_feat)
            ic(neg_feat)
            ic(pos_q)
            ic(neg_q)

        # Compute InfoNCE               (C, 1, BS, F)   x     (C, 1+N, BS, F)     -> (C, 1, BS, 1+N)  ->  (C*BS, 1+N)
        sim = torch.einsum('cpbf,cnbf->cpbn', [pos_q[:,[0]], torch.cat([pos_q[:,[1]], neg_q.expand(-1, -1, BS, -1)], dim=1)]).squeeze(1).reshape(C*BS, NEG_COUNT+1)
        labels = torch.zeros(sim.size(0), dtype=torch.long, device=dev)
        infonce_loss = F.cross_entropy(sim, labels)
        loss = infonce_loss

        # Minimize cluster center standard deviation
        if args.lambda_std > 0:
            with torch.autocast('cuda', enabled=False): # The following .mean() needs to be done in FP32
                std_loss = feature_std(pos_feat, reduce_dim=(1,2), feature_dim=-1)
            loss += args.lambda_std * std_loss.sum(0)
        else:
            std_loss = torch.zeros(1)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        # Validate
        with torch.no_grad():
            log_dict = {
                'Charts/Total_Loss': loss.cpu().item(),
                'Charts/InfoNCE_Loss': infonce_loss.cpu().item(),
                'Charts/StdDev_Loss': std_loss.sum(0).cpu().item(),
                'Charts/learning_rate': sched.get_last_lr()[0]
            }

            # Cluster standard deviations
            log_dict.update(
                { f'StdDev_Cos/{n}':    feature_std(pos_q[i],    reduce_dim=(0,1), feature_dim=-1).cpu() for i,n in enumerate(pos_samples.keys()) } |
                { f'StdDev_Logits/{n}': feature_std(pos_feat[i], reduce_dim=(0,1), feature_dim=-1).cpu() for i,n in enumerate(pos_samples.keys()) }
            )

            if (i == args.iterations-1) or (i % args.validation_every == 0 and not args.no_validation):
                with torch.autocast('cuda', enabled=True, dtype=typ):
                    full_feats = model(F.pad(make_5d(lowres_vol), tuple([REC_FIELD//2]*6)))
                    full_qs = F.normalize(full_feats, dim=1)
                    cc_l2  = torch.nan_to_num(torch.stack([ full_feats[split_squeeze(v, bs=1, f=NF)].mean(dim=(0,2)) for n,v in class_indices.items() ]))
                    cc_cos = torch.nan_to_num(torch.stack([    full_qs[split_squeeze(v, bs=1, f=NF)].mean(dim=(0,2)) for n,v in class_indices.items() ]))

                # Distance to cluster centers
                l2_center_distances = torch.pow(full_feats - cc_l2[:,:,None,None,None].expand(-1, -1, 1, 1, 1), 2.0).sum(dim=1).sqrt().squeeze(0)
                # Get closest (i.e. segmentation) cluster center class
                l2_closest = l2_center_distances.argmin(dim=0).to(torch.uint8)
                # Distance to cluster center map
                l2_distance_map = torch.exp(-l2_center_distances)
                # Cosine distances as distance map
                cos_center_distances = torch.clamp(torch.einsum('fdhw,nf->ndhw', (full_qs.squeeze(0), cc_cos)), 0, 1)
                # Get (cosine distance) closest cluster center segmentation
                cos_closest = cos_center_distances.argmax(dim=0)
                # Compute IoUs
                l2_iou = jaccard(l2_closest.cpu(), lowres_mask)
                jaccard.reset()
                best_logit_iou = torch.stack([best_logit_iou, l2_iou], dim=0).max(0).values
                cos_iou = jaccard(cos_closest.cpu(), lowres_mask)
                jaccard.reset()
                best_cosine_iou = torch.stack([best_cosine_iou, cos_iou], dim=0).max(0).values
                log_dict.update(
                    { f'IoU_logits/{n}':     v for n,v in zip(label_dict.values(), l2_iou) } |
                    { f'IoU_cosine/{n}':     v for n,v in zip(label_dict.values(), cos_iou) } | 
                    { f'BestIoU_cosine/{n}': v for n,v in zip(label_dict.values(), best_cosine_iou) } |
                    { f'BestIoU_logits/{n}': v for n,v in zip(label_dict.values(), best_logit_iou) } |
                    { f'Plots_Dist_L2/{n}/z': wandb.Image(l2_distance_map[i, IDX].cpu().float().numpy())
                        for i, n in enumerate(class_indices.keys())} |
                    { f'Plots_Dist_L2/{n}/y': wandb.Image(l2_distance_map[i, :, IDX].cpu().float().numpy())
                        for i, n in enumerate(class_indices.keys()) } |
                    { f'Plots_Dist_L2/{n}/x': wandb.Image(l2_distance_map[i, :, :, IDX].cpu().float().numpy())
                        for i, n in enumerate(class_indices.keys()) } |
                    { f'Plots_Dist_Cos/{n}/z': wandb.Image(cos_center_distances.cpu()[i, IDX])
                        for i, n in enumerate(class_indices.keys()) } |
                    { f'Plots_Dist_Cos/{n}/y': wandb.Image(cos_center_distances.cpu()[i, :, IDX])
                        for i, n in enumerate(class_indices.keys()) } |
                    { f'Plots_Dist_Cos/{n}/x': wandb.Image(cos_center_distances.cpu()[i, :, :, IDX])
                        for i, n in enumerate(class_indices.keys()) }
                )
                log_dict['Plots_Seg/L2dist/z'] = wandb.Image(vol_u8[IDX], masks={
                    'predictions':  {'mask_data': l2_closest[IDX].cpu().numpy(),   'class_labels': label_dict },
                    'ground_truth': {'mask_data': lowres_mask[IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg/L2dist/y'] = wandb.Image(vol_u8[:, IDX], masks={
                    'predictions':  {'mask_data': l2_closest[:, IDX].cpu().numpy(),   'class_labels': label_dict },
                    'ground_truth': {'mask_data': lowres_mask[:, IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg/L2dist/x'] = wandb.Image(vol_u8[:, :, IDX], masks={
                    'predictions':  {'mask_data': l2_closest[:, :, IDX].cpu().numpy(),   'class_labels': label_dict },
                    'ground_truth': {'mask_data': lowres_mask[:, :, IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg/CosDist/z'] = wandb.Image(vol_u8[IDX], masks={
                    'predictions':  {'mask_data': cos_closest[IDX].cpu().numpy(),  'class_labels': label_dict },
                    'ground_truth': {'mask_data': lowres_mask[IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg/CosDist/y'] = wandb.Image(vol_u8[:, IDX], masks={
                    'predictions':  {'mask_data': cos_closest[:, IDX].cpu().numpy(),  'class_labels': label_dict },
                    'ground_truth': {'mask_data': lowres_mask[:, IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg/CosDist/x'] = wandb.Image(vol_u8[:, :, IDX], masks={
                    'predictions':  {'mask_data': cos_closest[:, :, IDX].cpu().numpy(),  'class_labels': label_dict },
                    'ground_truth': {'mask_data': lowres_mask[:, :, IDX].numpy(), 'class_labels': label_dict }
                })

                # Similarity / ClusterDistance matrices
                cos_sim = similarity_matrix(cc_cos.float(), 'cosine')
                l2_sim  = similarity_matrix(cc_l2.float(),   'l2')
                cos_sim_fig = plot_similarity_matrix(cos_sim.cpu(), list(label_dict.values()), 'cosine')
                l2_sim_fig  = plot_similarity_matrix(l2_sim.cpu(),  list(label_dict.values()), 'l2')
                close_plot = True
                log_dict.update({
                    'Plots_CC_similarity/cosine': cos_sim_fig,
                    'Plots_CC_similarity/logits': l2_sim_fig
                })

            wandb.log(log_dict)
            if close_plot:
                plt.close(cos_sim_fig)
                plt.close(l2_sim_fig)
