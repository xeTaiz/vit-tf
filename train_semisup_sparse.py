import pprint
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex, ConfusionMatrix
import os
from tqdm import trange
from collections import OrderedDict
from itertools import count
import random

from torchvtk.utils import make_4d, make_5d

from rle_shit import decode_from_annotation

import wandb
from argparse import ArgumentParser

from utils import *
from lars import LARS
from semisparseconv import gather_receiptive_fields2 as gather_receiptive_fields
from paws import paws_loss

pltkwargs = {
    'dpi':  200,
    'tight_layout': True
}

def log_tensor(t, name):
    print(f'{name}: {tuple(t.shape)} in value range [{t.min().item():.3f}, {t.max().item():.3f}] and of type {t.dtype}')

class CenterCrop(nn.Module):
    def __init__(self, ks=3):
        super().__init__()
        self.pad = ks // 2

    def forward(self, x):
        i = self.pad
        out = x[..., i:-i, i:-i, i:-i]
        return out

class PrintLayer(nn.Module):
    def __init__(self, name=''):
        super().__init__()
        self.name = name
    def forward(self, x):
        log_tensor(x, self.name)
        return x

def conv_layer(n_in, n_out, Norm, Act, suffix=''):
    return nn.Sequential(OrderedDict([
        (f'conv{suffix}', nn.Conv3d(n_in, n_out, kernel_size=3, stride=1, padding=0)),
        (f'norm{suffix}', Norm(n_out // 4, n_out)),
        (f'act{suffix}', Act(inplace=True)),
    ]))

def create_cnn(in_dim, n_features=[8, 32, 16], Act=nn.Mish, Norm=nn.GroupNorm):
    feats = [in_dim] + n_features[:-1]
    layers = [(f'conv_block{i}', conv_layer(n_in, n_out, Norm=Norm, Act=Act, suffix=i))
            for i, n_in, n_out in zip(count(1), feats, feats[1:])]
    last = nn.Conv3d(n_features[-2], n_features[-1], kernel_size=3, stride=1, padding=0)
    return nn.Sequential(OrderedDict([*layers, ('last_conv', last)]))

class NTF(nn.Module):
    def __init__(self, in_dim, conv_layers, hidden_sz, out_classes, head_bottleneck=4):
        super().__init__()
        self.encoder = create_cnn(in_dim=in_dim, n_features=conv_layers)
        NF = conv_layers[-1]
        NH = hidden_sz
        self.head = nn.Sequential(OrderedDict([
            ('bn0',   nn.BatchNorm1d(NF)),
            ('fc1',   nn.Linear(NF, NH//head_bottleneck)),
            ('bn1',   nn.BatchNorm1d(NH//head_bottleneck)),
            ('mish1', nn.Mish(True)),
            ('fc2',   nn.Linear(NH//head_bottleneck, NF))
        ]))
        self.proj = nn.Sequential(OrderedDict([
            ('bn0',   nn.BatchNorm1d(NF)),
            ('fc1',   nn.Linear(NF, NH)),
            ('bn1',   nn.BatchNorm1d(NH)),
            ('mish1', nn.Mish(True)),
            ('fc2',   nn.Linear(NH, NH)),
            ('bn2',   nn.BatchNorm1d(NH)),
            ('mish2', nn.Mish(True)),
            ('fc3',   nn.Linear(NH, NF))
        ]))
        self.predict = nn.Sequential(OrderedDict([
            ('bn0',   nn.BatchNorm1d(NF)),
            ('fc1',   nn.Linear(NF, NH)),
            ('bn1',   nn.BatchNorm1d(NH)),
            ('mish1', nn.Mish(True)),
            ('fc2',   nn.Linear(NH, out_classes))
        ]))

    def forward(self, x, return_class_pred=False):
        z = self.encoder(x).squeeze() # BS, F, D,H,W -> BS, F
        feat = self.proj(z)
        pred = self.head(feat)
        if return_class_pred:
            clas = self.predict(z.detach())
            return feat, pred, clas
        else:
            return feat, pred

    def forward_fullvol(self, x):
        z = self.encoder(x).permute(0, 2,3,4, 1)
        shap = z.shape # BS, D,H,W, F
        clas = self.predict(z.reshape(-1, z.size(-1)))
        return clas.view(*shap[:4], -1).permute(0, 4, 1,2,3).contiguous()


if __name__ == '__main__':
    parser = ArgumentParser('Contrastive TF Design')
    parser.add_argument('--data', required=True, type=str, help='Path to Data with {vol, mask, labels} keys in .pt file')
    parser.add_argument('--unlabeled-class', type=str, default='unlabeled', help='Name of the background class')
    parser.add_argument('--vol-scaling-factor', type=float, default=1.0, help='Scaling factor to reduce spatial resolution of volumes')
    parser.add_argument('--no-pos-encoding', action='store_true', help='Use positional encoding with input (3D coordinate)')
    parser.add_argument('--raw-data', action='store_true', help='Use raw data and do not normalize input to 0-mean 1-std')
    parser.add_argument('--fp32', action='store_true', help='Use full 32-bit precision')
    parser.add_argument('--learning-rate', type=float, default=5e-1, help='Learning rate for optimization')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='Weight decay for optimizer')
    parser.add_argument('--lr-schedule', type=str, default='onecycle', help='Learning rate schedule')
    parser.add_argument('--iterations', type=int, default=10001, help='Number of optimization steps')
    parser.add_argument('--samples-per-iteration', type=int, default=4096, help='Number of samples per class used in each iteration')
    parser.add_argument('--supports-per-class', type=int, default=256, help='Number of support samples per class')
    parser.add_argument('--label-percentage', type=float, default=1.0, help='Percentage of labels to use for optimization')
    parser.add_argument('--wandb-tags', type=str, nargs='*', help='Additional tags to use for W&B')
    parser.add_argument('--no-validation', action='store_true', help='Skip validation')
    parser.add_argument('--cnn-layers', type=int, nargs='*', help='Number of features per CNN layer')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden feature dim used in projection and prediction heads')
    parser.add_argument('--debug', action='store_true', help='Turn of WandB, some more logs')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for experiment')
    args = parser.parse_args()

    # Determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        torch.autograd.set_detect_anomaly(True)

    # Setup
    BS = args.samples_per_iteration
    M = args.supports_per_class
    NO_CLASS = args.unlabeled_class
    DOWNSAMPLE = args.vol_scaling_factor != 1.0
    NORMALIZE = not args.raw_data
    POS_ENCODING = not args.no_pos_encoding
    if args.lr_schedule.lower() == 'onecycle':
        LR_SCHED = partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=args.learning_rate, total_steps=args.iterations, cycle_momentum=False)
    elif args.lr_schedule.lower() == 'cosine':
        LR_SCHED = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=args.iterations)
    else:
        LR_SCHED = partial(torch.optim.lr_scheduler.ConstantLR, factor=1.0)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    typ = torch.float32 if args.fp32 else torch.float16

    # Data
    data = torch.load(args.data)
    if DOWNSAMPLE:
        vol  = F.interpolate(make_5d(data['vol']).float(), scale_factor=args.vol_scaling_factor, mode='nearest').squeeze()
        mask = F.interpolate(make_5d(data['mask']),        scale_factor=args.vol_scaling_factor, mode='nearest').squeeze()
    else:
        vol  = data['vol']
        mask = data['mask']
    log_tensor(vol, 'Loaded volume data')
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
        mask_reduced[split_squeeze3d(to_drop)] = num_classes
    else:
        mask_reduced = mask
    # Get downsampled volume for validation
    lowres_vol  = F.interpolate(make_5d(data['vol']).float(), scale_factor=0.25, mode='nearest')
    lowres_mask = F.interpolate(make_5d(data['mask']),        scale_factor=0.25, mode='nearest').squeeze()
    vol_u8 = (255.0 * (lowres_vol - lowres_vol.min()) / (lowres_vol.max() - lowres_vol.min())).squeeze().cpu().numpy().astype(np.uint8)
    log_tensor(vol_u8, 'vol_u8')
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
        vol = ((vol.float() - vol.float().mean()) / vol.float().std()).to(typ).to(dev)
    else:
        vol = vol.to(typ).to(dev)
    log_tensor(vol, 'Normalized volume')

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

    args.cnn_layers = args.cnn_layers if args.cnn_layers else [8, 16, 32, 64]
    NF = args.cnn_layers[-1]
    model = NTF(in_dim=vol.size(0), conv_layers=args.cnn_layers, hidden_sz=args.hidden_size, out_classes=num_classes).to(dev)
    REC_FIELD = len(args.cnn_layers) * 2 + 1

    tags = [f'{args.label_percentage} Labels', 'RawData' if args.raw_data else 'NormalizedData', 'SemiSparse', *(args.wandb_tags if args.wandb_tags else [])]
    wandb.init(project='ntf', entity='viscom-ulm', tags=tags, config=vars(args), mode='offline' if args.debug else 'online')
    wandb.watch(model)
    jaccard = JaccardIndex(num_classes=num_classes, average=None)
    best_iou = torch.zeros(num_classes)

    sample_idxs = { n: l.size(0) for n, l in class_indices.items() }
    if args.debug:
        print(model)
        print('Volume Indices for each class:')
        pprint.pprint({n: v.shape for n, v in class_indices.items()})
        print('Number of annotations per class:')
        pprint.pprint(sample_idxs)

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
            anp_crops = gather_receiptive_fields(make_4d(vol), class_indices[NO_CLASS][anp_samples].to(dev), ks=REC_FIELD) # (2*BS, 1, Z,Y,X)
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
            log_tensor(sup_crops, 'sup_crops')
            log_tensor(anp_crops, 'anp_crops')
            log_tensor(sup_anc_feat, 'sup_anc_feat')
            log_tensor(sup_pos_feat, 'sup_pos_feat')
            log_tensor(anc_feat, 'anc_feat')
            log_tensor(pos_feat, 'pos_feat')
            print('sup_samples:')
            pprint.pprint({k: v.shape for k,v in sup_samples.items()})

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

            if (i == args.iterations-1) or (i % 1000 == 0 and not args.no_validation):
                with torch.autocast('cuda', enabled=True, dtype=typ):
                    full_feats = model.forward_fullvol(F.pad(lowres_vol, tuple([REC_FIELD//2]*6))).squeeze(0)
                    pred = full_feats.argmax(dim=0)

                # Compute IoUs
                iou = jaccard(pred.cpu(), lowres_mask)
                jaccard.reset()

                best_iou = torch.stack([best_iou, iou], dim=0).max(0).values
                log_dict.update({
                    f'IoU/{n}': v for n,v in zip(label_dict.values(), iou)
                })
                log_dict.update({
                    f'BestIoU/{n}': v for n,v in zip(label_dict.values(), best_iou)
                })

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
