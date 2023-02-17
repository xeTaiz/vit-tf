import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from torchmetrics import JaccardIndex
from itertools import count
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import sys

import icecream as ic

from torchvtk.utils import make_4d, make_5d
from domesutils import *

in_mean = [0.485, 0.456, 0.406]
in_std = [0.229, 0.224, 0.225]

dino_archs = ['vits16', 'vits8', 'vitb16', 'vitb8',
    'xcit_small_12_p16', 'xcit_small_12_p8',
    'xcit_medium_24_p16', 'xcit_medium_24_p8',
    'resnet50']

def get_dino_model(name):
    return torch.hub.load('facebookresearch/dino:main', f'dino_{name}')

def sample_features2d(feat_vol, abs_coords, rel_coords, mode='nearest'):
    ''' Samples features at given coords from `feat_vol` by first indexing the un-reduced dimension, then interpolating in the reduced dimensions

    Args:
        feat_vol (Tensor): Shape (1, F, W, H, D)
        abs_coords (Tensor): Shape (C, A, 3)
        rel_coords (Tensor): Shape (C, A, 3)
        mode (str): Interpolation along 2D slice. One of: nearest, bilinear

    Returns:
        Tensor: Shape (C, A, F)
    '''
    slices = feat_vol.squeeze(0).permute(3,0,1,2)[abs_coords.view(-1,3)[:, 2]] # C*A, F, W, H
    grid_idx = rel_coords.view(-1,3)[:, None, None, [1,0]].to(dev).to(typ) # (C*A, 1,1, 2)
    queried_feats = F.grid_sample(slices, grid_idx, mode=mode, align_corners=False) # (C*A, F, 1, 1)
    return queried_feats.reshape(abs_coords.size(0), abs_coords.size(1), feat_vol.size(1))

def sample_features3d(feat_vol, rel_coords, mode='nearest'):
    '''Samples features at given coords from `feat_vol` by interpolating the feature maps in all dimensions. This should result in nearest interpolation along the un-reduced dimension

    Args:
        feat_vol (Tensor): Shape (F, W, H, D)
        rel_coords (Tensor): Shape (C, A, 3)
        mode (str): Interpolation along 3D volume. One of: nearest, bilinear

    Returns:
        Tensor: Shape (C, A, F)

    '''
    # Flip dims to get X,Y,Z -> Z,Y,X
    grid_idx = make_5d(rel_coords.flip(dims=(-1,))).to(feat_vol.dtype).to(feat_vol.device) # (1, 1, C, A, 3)
    feats = F.grid_sample(make_5d(feat_vol), grid_idx, mode=mode, align_corners=False)
    return feats.squeeze(0).squeeze(1).permute(1,2,0).contiguous()

def plot_sims(raw_slice, sims, mask_slice, pred_mask_slice, coord, label=''):
    X, Y, Z = coord
    max_idxs = torch.argwhere(sims == sims.max()).squeeze().cpu()
    if max_idxs.numel() > 2: 
        mX, mY = max_idxs[0]
    else:
        mX, mY = max_idxs
    fig, ax = plt.subplots(2,2, figsize=(10,11), tight_layout=True)
    for x in ax.reshape(-1): x.set_axis_off()
    ax[0,0].imshow(raw_slice.cpu())
    ax[0,0].set_title('Raw Data')
    ax[0,1].imshow(sims.cpu())
    ax[0,1].set_title('Similarity')
    ax[1,0].imshow(mask_slice.cpu())
    ax[1,0].set_title('GT Segmentation')
    ax[1,1].imshow(pred_mask_slice.cpu())
    ax[1,1].set_title('Predicted Segmentation')
    ax[0,0].plot(Y, X, marker='x', color="red")
    ax[0,1].plot(Y, X, marker='x', color="red")
    ax[0,1].plot(mY, mX, marker='x', color='blue')
    ax[1,0].plot(Y, X, marker='x', color='white')
    ax[1,0].plot(Y, X, marker='+', color='black')
    ax[1,1].plot(Y, X, marker='x', color='white')
    ax[1,1].plot(Y, X, marker='+', color='black')
    fig.suptitle(f'Coord: {X, Y}, Slice: {Z}, {label}')
    return fig

def plot_sims_xyz(raw_slices, sims, asims, mask_slices, pred_mask_slices, coord, label=''):
    X, Y, Z = coord
    fig, ax = plt.subplots(3,5, figsize=(25,17), tight_layout=True)
    fig.suptitle(f'Coord: {X, Y, Z}, {label}')
    raw_marker_pos = [(Z,Y), (Z,X), (Y,X)]
    for i,n,r in zip(count(), ['x', 'y', 'z'], raw_marker_pos):
        raw_slice = raw_slices[i]
        sim, asim, mask_slice, pred_mask_slice = sims[i], asims[i], mask_slices[i], pred_mask_slices[i]
        

        max_idxs = torch.argwhere(sim == sim.max()).squeeze().cpu()
        if max_idxs.numel() > 2: 
            mX, mY = max_idxs[0]
        else:
            mX, mY = max_idxs
        for x in ax.reshape(-1): x.set_axis_off()
        # Raw Slices
        ax[i,0].imshow(raw_slice.cpu())
        ax[i,0].set_title(f'Raw Data along {n}')
        ax[i,0].plot(r[0], r[1], marker='x', color="red")
        # Similarity for query point
        ax[i,1].imshow(sim.cpu())
        ax[i,1].set_title(f'Similarity along {n}')
        ax[i,1].plot(r[0], r[1], marker='x', color="red")
        ax[i,1].plot(mY, mX, marker='x', color='blue')
        # Average similarity over all query points
        ax[i,2].imshow(asim.cpu())
        ax[i,2].set_title(f'AVG Similarity along {n}')
        ax[i,2].plot(r[0], r[1], marker='x', color="red")
        ax[i,2].plot(mY, mX, marker='x', color='blue')
        # GT Segmentation
        ax[i,3].imshow(mask_slice.cpu())
        ax[i,3].set_title(f'GT Segmentation along {n}')
        ax[i,3].plot(r[0], r[1], marker='x', color='white')
        ax[i,3].plot(r[0], r[1], marker='+', color='black')
        # Predicted Segmentation
        ax[i,4].imshow(pred_mask_slice.cpu())
        ax[i,4].set_title(f'Pred Segmentation along {n}')
        ax[i,4].plot(r[0], r[1], marker='x', color='white')
        ax[i,4].plot(r[0], r[1], marker='+', color='black')
    return fig

def compute_and_save_qkv(args, dev=torch.device('cpu'), typ=torch.float32):
    print(f'Computing feature maps for {data_path.name}')
    patch_size = 8 if args.dino_model[-1] == '8' else 16
    data = torch.load(data_path)

    model = get_dino_model(args.dino_model).to(dev)
    model.eval()
    feat_out = defaultdict(list)
    def hook_fn_forward_qkv(module, input, output):
        feat_out['qkv'].append(output.cpu())
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    vol = data['vol'].float().squeeze()
    slice_along = {
        'z': ((3, 0, 1, 2), (1,2,3,0)),
        'y': ((2, 0, 1, 3), (1,2,0,3)),
        'x': ((1, 0, 2, 3), (1,0,2,3))
    }
    permute_in, permute_out = slice_along[args.slice_along]
    ic(vol)
    image = make_4d(vol).permute(*permute_in).expand(-1, 3, -1, -1)
    image = normalize(norm_minmax(image), in_mean, in_std)
    ic(image.mean())
    ic(image.std())
    im_sz = image.shape[-2:]

    # forward pass
    out = []
    with torch.cuda.amp.autocast(enabled=True, dtype=typ):
        with torch.no_grad():
            im_in = image.to(dev)
            ic(im_in)
            for batch in tqdm(torch.arange(im_in.size(0)).split(args.batch_size)):
                _ = model(im_in[batch])

            attentions = model.get_last_selfattention(im_in[[0]])

    # Scaling factor
    scales = [patch_size, patch_size]

    # Dimensions
    nb_im = im_in.size(0) # Batch sizemonitor
    nh = attentions.shape[1]  # Number of heads
    nb_tokens = attentions.shape[2]  # Number of tokens

    # Extract the qkv features of the last attention layer
    qkv = (
        torch.cat(feat_out["qkv"][:-1])
        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    f_sz = im_sz[0] // patch_size , im_sz[1] // patch_size
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    k = k[:, 1:].reshape(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    q = q[:, 1:].reshape(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    v = v[:, 1:].reshape(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
    ic(q)
    ic(k)
    ic(v)
    qkv = {
        'q': q.permute(*permute_out).contiguous(),
        'k': k.permute(*permute_out).contiguous(),
        'v': v.permute(*permute_out).contiguous()
    }
    torch.save(qkv, cache_path)
    print(f'Saved feature maps: {cache_path.name}')

    del model
    return qkv

if __name__ == '__main__':
    parser = ArgumentParser('DINO segmentor')
    parser.add_argument('--dino-model', type=str, choices=dino_archs, required=True, help='DINO model to use')
    parser.add_argument('--slice-along', type=str, choices=['x', 'y', 'z'], default='z', help='Along which axis to slice volume, as it is fed slice-wise to DINO')
    parser.add_argument('--batch-size', type=int, default=2, help='Feed volume through network in batches')
    parser.add_argument('--annotations-per-class', type=int, default=8, help='Number of annotated features to use per class')
    parser.add_argument('--feature-interpolation-mode', type=str, default='nearest', choices=['nearest', 'bilinear'], help='Interpolation mode for features (in reduced dimensions)')
    parser.add_argument('--attention-features', type=str, default='k', choices=['q', 'k', 'v'], help='Which of the attention features to use.')
    parser.add_argument('--similarity-exponent', type=float, default=2.0, help='Raise similarity to the given power to suppress non-similars')
    parser.add_argument('--minmax-norm-similarities', type=str, default='false', choices=['true', 'false'], help='Whether to normalize averaged similarity maps to [0,1] range')
    parser.add_argument('--drop-sims-below', type=float, default=0.0, help='Sets low similarities to 0 (except for background class)')
    parser.add_argument('--background-class', type=str, default='background', help='Name of background class for dropping low similarities')
    parse_basics(parser)
    args = parser.parse_args()
    setup_seed_and_debug(args)


    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    typ = torch.float16 if args.fp16 == 'true' else torch.float32
    BG_CLASS = args.background_class

    cache_path = Path(args.data.replace('volume', f'{args.dino_model}_feats_along_{args.slice_along}'))
    data_path = Path(args.data)
    patch_size = 8 if args.dino_model[-1] == '8' else 16

    with torch.no_grad(): # Disable all autograd stuff
        # Load cached features
        if Path(cache_path).exists():
            print(f'Found existing feature maps: {cache_path.name}')
            qkv = torch.load(cache_path)
        else: # Compute and save features
            qkv = compute_and_save_qkv(args, dev=dev, typ=typ)

        feat_vol = qkv[args.attention_features].to(dev)
        # Load Data, Mask
        data = torch.load(data_path)
        mask = data['mask']
        ic(mask)
        ic(feat_vol)

        # Choose some annotated voxels randomly
        non_bg_indices = [(mask == i).nonzero() for i in range(len(data['labels']))]
        abs_coords = torch.stack([
            clas_idcs[torch.from_numpy(np.random.choice(clas_idcs.size(0), args.annotations_per_class))]
            for clas_idcs in non_bg_indices if clas_idcs.size(0) > 0], dim=0)
        ic(abs_coords) # (C, A, 3)
        num_classes = abs_coords.size(0)
        label_dict = {i: n for i, n in enumerate(data['labels'])}
        BG_IDX = list(label_dict.values()).index(BG_CLASS)
        NON_BG_IDX = torch.tensor([i for i in range(num_classes) if i != BG_IDX])
        vol_u8 = (norm_minmax(data['vol'].float()) * 255.0).to(torch.uint8)

        # Compute query point coordinates
        vol_extent = torch.tensor([[[*mask.shape]]], dtype=torch.float32, device=abs_coords.device)
        rel_coords = (abs_coords.float() + 0.5) / vol_extent * 2.0 - 1.0
        ic(rel_coords)

        # Normalize Feature Volume
        feat_vol = F.normalize(feat_vol, dim=0) # (F, W, H, D)
        ic(feat_vol)

        # Sample Feature vectors for annotated voxels
        qf = sample_features3d(feat_vol, rel_coords, mode='nearest')
        ic(qf)
        # qf2d = sample_features2d(k_vol, abs_coords, rel_coords, mode='nearest')
        # ic(qf2d)
        # qf = F.normalize(qf3d, dim=-1) # already done above with feat_vol

        # Compute Similarities for each annotation a in each class c
        sims = torch.einsum('fwhd,caf->cawhd', (feat_vol, qf)).clamp(0, 1) ** args.similarity_exponent
        avg_sims = sims.mean(dim=1) # Average similarity per class over its annotations
        ic(feat_vol)
        ic(sims)
        ic(avg_sims)

        # Compute IoUs
        jaccard = JaccardIndex(num_classes=num_classes, average=None).to(dev)
        lowres_mask = F.interpolate(make_5d(mask.to(dev)), avg_sims.shape[-3:], mode='nearest').squeeze()
        if args.drop_sims_below > 0.0: # Discard low similarities
            low_sim_mask = avg_sims[NON_BG_IDX] < args.drop_sims_below
            avg_sims[NON_BG_IDX][low_sim_mask] = 0.0
            ic(avg_sims[NON_BG_IDX])
        pred_mask = avg_sims.argmax(0)
        ic(pred_mask)
        ious =  jaccard(pred_mask, lowres_mask)
        iou_dict = {label_dict[i]: v.cpu().item() for i,v in enumerate(ious)}
        ic(iou_dict)

        # Plotting
        if args.minmax_norm_similarities == 'true':
            avg_sims = avg_sims / avg_sims.max(dim=0, keepdim=True).values
        for i in range(num_classes):
            for a in range(args.annotations_per_class)[:4]:
                X,Y,Z = abs_coords[i,a].tolist()
                aX = X if args.slice_along == 'x' else X // patch_size
                aY = Y if args.slice_along == 'y' else Y // patch_size
                aZ = Z if args.slice_along == 'z' else Z // patch_size
                sim_x  = F.interpolate(make_4d(    sims[i, a, aX, :, :]), (mask.size(1), mask.size(2)), mode='bilinear').squeeze()
                asim_x = F.interpolate(make_4d(avg_sims[i,    aX, :, :]), (mask.size(1), mask.size(2)), mode='bilinear').squeeze()
                pmask_x = F.interpolate(make_4d(pred_mask[aX,:,:]).float(), (mask.size(1), mask.size(2)), mode='bilinear').squeeze().round().long()
                sim_y  = F.interpolate(make_4d(    sims[i, a, :, aY, :]), (mask.size(0), mask.size(2)), mode='bilinear').squeeze()
                asim_y = F.interpolate(make_4d(avg_sims[i,    :, aY, :]), (mask.size(0), mask.size(2)), mode='bilinear').squeeze()
                pmask_y = F.interpolate(make_4d(pred_mask[:,aY,:]).float(), (mask.size(0), mask.size(2)), mode='bilinear').squeeze().round().long()
                sim_z  = F.interpolate(make_4d(    sims[i, a, :, :, aZ]), (mask.size(0), mask.size(1)), mode='bilinear').squeeze()
                asim_z = F.interpolate(make_4d(avg_sims[i,    :, :, aZ]), (mask.size(0), mask.size(1)), mode='bilinear').squeeze()
                pmask_z = F.interpolate(make_4d(pred_mask[:,:,aZ]).float(), (mask.size(0), mask.size(1)), mode='bilinear').squeeze().round().long()

                fig = plot_sims_xyz(
                    (vol_u8[X,:,:], vol_u8[:,Y,:], vol_u8[:,:,Z]), 
                    (sim_x, sim_y, sim_z), 
                    (asim_x, asim_y, asim_z), 
                    (mask[X,:,:], mask[:,Y,:], mask[:,:,Z]), 
                    (pmask_x, pmask_y, pmask_z), coord=(X,Y,Z), label=f'Label: {label_dict[i]}')
                fig.savefig(f'figures_annotations/{label_dict[i]}_{a}.png')
                plt.close(fig)
