from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex, ConfusionMatrix
from tqdm import trange

from torchvtk.utils import make_4d, make_5d

import wandb
from argparse import ArgumentParser

from lars import LARS
from utils import *
from models import PAWSNet
from semisparseconv import gather_receiptive_fields2 as gather_receiptive_fields
from paws import paws_loss, transform_paws_crops

if __name__ == '__main__':
    parser = ArgumentParser('Contrastive TF Design')
    parser.add_argument('--unlabeled-class', type=str, default='unlabeled', help='Name of the background class')
    parser.add_argument('--vol-scaling-factor', type=float, default=1.0, help='Scaling factor to reduce spatial resolution of volumes')
    parser.add_argument('--samples-per-iteration', type=int, default=4096, help='Number of samples per class used in each iteration')
    parser.add_argument('--supports-per-class', type=int, default=256, help='Number of support samples per class')
    parser.add_argument('--cnn-layers', type=str, help='Number of features per CNN layer')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden feature dim used in projection and prediction heads')
    parse_basics(parser)
    args = parser.parse_args()
    setup_seed_and_debug(args)

    # Setup
    BS = args.samples_per_iteration
    M = args.supports_per_class
    NO_CLASS = args.unlabeled_class
    DOWNSAMPLE = args.vol_scaling_factor != 1.0
    NORMALIZE = args.normalize == 'true'
    POS_ENCODING = args.pos_encoding == 'true'
    if args.lr_schedule.lower() == 'onecycle':
        LR_SCHED = partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=args.learning_rate, total_steps=args.iterations, cycle_momentum=False)
    elif args.lr_schedule.lower() == 'cosine':
        LR_SCHED = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=args.iterations)
    else:
        LR_SCHED = partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    typ = torch.float16 if args.fp16 == 'true' else torch.float32

    # Data
    data = torch.load(args.data)
    if DOWNSAMPLE:
        vol  = F.interpolate(make_5d(data['vol']).float(), scale_factor=args.vol_scaling_factor, mode='nearest').squeeze()
        mask = F.interpolate(make_5d(data['mask']),        scale_factor=args.vol_scaling_factor, mode='nearest').squeeze()
    else:
        vol  = data['vol']
        mask = data['mask']
    ic(vol)
    num_classes = len(data['labels'])
    label_dict = {i: n for i,n in enumerate(data['labels'] + ['unlabeled'])}
    label2idx =  {n: i for i,n in label_dict.items()}

    if args.label_percentage < 1.0:
        non_bg_indices = [(mask == i).nonzero() for i in range(len(data['labels']))]
        # Choose 1 - label_pct non-bg samples to set to background
        to_drop = torch.cat([clas_idcs[torch.from_numpy(np.random.choice(clas_idcs.size(0),
            int((1.0 - args.label_percentage) * clas_idcs.size(0))
        ))] for clas_idcs in non_bg_indices if clas_idcs.size(0) > 0], dim=0)
        mask_reduced = mask.clone()
        mask_reduced[split_squeeze3d(to_drop)] = num_classes
    else:
        mask_reduced = mask
    # Get downsampled volume for validation
    lowres_vol  = F.interpolate(make_5d(data['vol']).float(), scale_factor=0.25, mode='nearest')
    lowres_mask = F.interpolate(make_5d(data['mask']),        scale_factor=0.25, mode='nearest').squeeze()
    vol_u8 = (255.0 * norm_minmax(lowres_vol)).squeeze().cpu().numpy().astype(np.uint8)
    ic(vol_u8)
    lowres_vol = lowres_vol.to(typ).to(dev)
    IDX = min(lowres_vol.shape[-3:]) // 2

    class_indices = {
        n: (mask_reduced == i).nonzero()
        for i, n in enumerate(data['labels'] + ['unlabeled'])
    }

    to_remove = [(i,n) for i,n in enumerate(class_indices.keys()) if class_indices[n].size(0) < 3]
    for i,n in to_remove: # Remove classes with too few samples
        del class_indices[n]
        del label_dict[i]
        del label2idx[n]
        num_classes -= 1

    if NORMALIZE:
        vol = (norm_mean_std(vol)).to(typ).to(dev)
    else:
        vol = vol.to(typ).to(dev)
    ic(vol)

    if POS_ENCODING:
        x = torch.linspace(-1, 1, vol.size(-1), device=dev, dtype=typ)
        y = torch.linspace(-1, 1, vol.size(-2), device=dev, dtype=typ)
        z = torch.linspace(-1, 1, vol.size(-3), device=dev, dtype=typ)
        z, y, x = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack((z, y, x)) * 1.7185
        vol = torch.cat([vol[None], coords], dim=0)
        lowres_vol = torch.cat([lowres_vol, F.interpolate(make_5d(coords), scale_factor=0.25/args.vol_scaling_factor, mode='nearest')], dim=1)
    else:
        vol = vol[None]

    args.cnn_layers    = [int(n.strip()) for n in    args.cnn_layers.replace('[', '').replace(']', '').split(' ')] if args.cnn_layers    else [8, 16, 32, 64]
    NF = args.cnn_layers[-1]
    model = PAWSNet(in_dim=vol.size(0), conv_layers=args.cnn_layers, hidden_sz=args.hidden_size, out_classes=num_classes).to(dev)
    REC_FIELD = len(args.cnn_layers) * 2 + 1

    group = 'SemiSupervised Sparse'
    tags = [f'{args.label_percentage} Labels', 'NormalizedData' if args.normalize else 'RawData', 'SemiSparse', *(args.wandb_tags if args.wandb_tags else [])]
    wandb.init(project='ntf', entity='viscom-ulm', tags=tags, config=vars(args), 
        mode='offline' if args.debug else 'online', group=group)
    wandb.watch(model)
    jaccard = JaccardIndex(num_classes=num_classes, average=None)
    best_iou = torch.zeros(num_classes)

    sample_idxs = { n: l.size(0) for n, l in class_indices.items() }
    if args.debug:
        ic(model)
        ic(sample_idxs)

    close_plot = False

    # Training
    scaler = torch.cuda.amp.GradScaler()
    param_groups = [
        {'params': (p for n,p in model.named_parameters() if ('bias' not in n) and ('bn' not in n) and ('norm' not in n))},
        {'params': (p for n,p in model.named_parameters() if ('bias'     in n) or  ('bn'     in n) or  ('norm'     in n)),
            'LARS_exclude': True, 'weight_decay': 0}
    ]
    opt = torch.optim.SGD(param_groups, weight_decay=args.weight_decay, momentum=0.9, lr=args.learning_rate)
    opt = LARS(opt, trust_coefficient=0.001)
    # opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = LR_SCHED(opt)
    label_eye = torch.eye(num_classes)
    # label = torch.cat([label_eye[None, i].expand(2*BS, num_classes-1) for i in range(num_classes-1)]).to(typ).to(dev)
    label = label_eye[[i for i in range(num_classes)]*M].to(dev)

    for i in trange(args.iterations):
        # Draw voxel samples to work with
        with torch.no_grad():
            sup_samples = { # Support samples
                n: torch.from_numpy(np.random.choice(v, M))
                for n,v in sample_idxs.items() if v >= M and n != NO_CLASS
            }
            # Anchor n' Positives
            anp_samples = torch.from_numpy(np.random.choice(sample_idxs[NO_CLASS], BS))
            sup_crops = gather_receiptive_fields(make_4d(vol), torch.cat(list({n: class_indices[n][v] for n,v in sup_samples.items()}.values()), dim=0).to(dev), ks=REC_FIELD) # (M*C*2, 1, Z,Y,X)
            anp_crops = gather_receiptive_fields(make_4d(vol), class_indices[NO_CLASS][anp_samples].to(dev), ks=REC_FIELD) # (BS, 1, Z,Y,X)
            anp_crops = transform_paws_crops(anp_crops)
            crops = torch.cat([sup_crops, anp_crops], dim=0) # (C*2 + C*N, IN, Z,Y,X)
        opt.zero_grad()
        # 1 Feed volume thru networks
        with torch.autocast('cuda', enabled=True, dtype=typ):
            h, z, clas_pred = model(crops, return_class_pred=True) # (C*2 + C*N, F, 1,1,1)
            with torch.autocast('cuda', enabled=True, dtype=torch.float32):
                # h, z = h.float(), z.float()
                sup_anc_feat = z[:sup_crops.size(0)] # (M*C, NF)
                anc_feat     = z[sup_crops.size(0):] # (BS, NF)
                sup_pos_feat = h[:sup_crops.size(0)].detach() # (M*C, NF)
                pos_feat     = h[sup_crops.size(0):].detach() # (BS, NF)
                # Swap the first and second half of batch (first and second augmentation)
                pos_feat = torch.cat([pos_feat[BS:], pos_feat[:BS]], dim=0)
                ploss, me_max, clas_loss = paws_loss(anc_feat, sup_anc_feat, label, pos_feat, sup_pos_feat, label, clas_pred=clas_pred)
                loss = ploss + me_max + clas_loss
        # 2 Choose samples from classes
        if args.debug:
            ic(sup_crops)
            ic(anp_crops)
            ic(sup_anc_feat)
            ic(sup_pos_feat)
            ic(anc_feat)
            ic(pos_feat)
            print('sup_samples:')
            ic({k: v.shape for k,v in sup_samples.items()})

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        # Validate
        with torch.no_grad():
            log_dict = {
                'Charts/ce_loss': ploss,
                'Charts/me_max': me_max,
                'Charts/clas_loss': clas_loss,
                'Charts/loss': loss,
                'Charts/learning_rate': sched.get_last_lr()[0]
            }

            if (i == args.iterations-1) or (i % args.validation_every == 0 and not args.no_validation):
                with torch.autocast('cuda', enabled=True, dtype=typ):
                    full_feats = model.forward_fullvol(F.pad(lowres_vol, tuple([REC_FIELD//2]*6))).squeeze(0)
                    pred = full_feats.argmax(dim=0)

                # Compute IoUs
                iou = jaccard(pred.cpu(), lowres_mask)
                jaccard.reset()

                best_iou = torch.stack([best_iou, iou], dim=0).max(0).values
                log_dict.update(
                    { f'IoU/{n}':     v for n,v in zip(label_dict.values(), iou) } |
                    { f'BestIoU/{n}': v for n,v in zip(label_dict.values(), best_iou) }
                )

                log_dict['Plots_Seg/z'] = wandb.Image(vol_u8[IDX], masks={
                    'predictions':  {'mask_data': pred[IDX].cpu().numpy(),  'class_labels': label_dict },
                    'ground_truth': {'mask_data': lowres_mask[IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg/y'] = wandb.Image(vol_u8[:, IDX], masks={
                    'predictions':  {'mask_data': pred[:, IDX].cpu().numpy(),  'class_labels': label_dict },
                    'ground_truth': {'mask_data': lowres_mask[:, IDX].numpy(), 'class_labels': label_dict }
                })
                log_dict['Plots_Seg/x'] = wandb.Image(vol_u8[:, :, IDX], masks={
                    'predictions':  {'mask_data': pred[:, :, IDX].cpu().numpy(),  'class_labels': label_dict },
                    'ground_truth': {'mask_data': lowres_mask[:, :, IDX].numpy(), 'class_labels': label_dict }
                })

                # Similarity / ClusterDistance matrices
                confusion = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(dev)
                conf_mat = confusion(pred, lowres_mask.to(dev))
                confusion_fig = plot_confusion_matrix(conf_mat.cpu(), list(label_dict.values()))
                close_plot = True
                log_dict.update({
                    'Plots_ConfusionMatrix': confusion_fig,
                })

            wandb.log(log_dict)
            if close_plot:
                plt.close(confusion_fig)
