import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
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

def sample_features(feat_vol, abs_coords, rel_coords):
    grid_idx = make_4d(rel_coords[...,:2]).to(dev).to(typ) # (1, 1, C, A, 3)
    queried_feats = F.grid_sample(k_vol[...,abs_coords[2]], grid_idx, mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0).contiguous()
    return F.normalize(queried_feats, dim=-1)

def plot_sims(raw_slice, sims, mask_slice, coord, label=''):
    X, Y, Z = coord
    mX, mY = torch.argwhere(sims == sims.max()).squeeze().cpu()
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    for x in ax.reshape(-1): x.set_axis_off()
    ax[0].imshow(raw_slice)
    ax[0].set_title('Raw Data')
    ax[1].imshow(sims.cpu())
    ax[1].set_title('Similarity')
    ax[2].imshow(mask_slice)
    ax[2].set_title('GT Segmentation')
    ax[0].plot(Y, X, marker='x', color="red")
    ax[1].plot(Y, X, marker='x', color="red")
    ax[1].plot(mY, mX, marker='x', color='blue')
    ax[2].plot(Y, X, marker='x', color='white')
    fig.suptitle(f'Coord: {X, Y}, Slice: {Z}, {label}')
    return fig

if __name__ == '__main__':
    parser = ArgumentParser('DINO segmentor')
    parser.add_argument('--dino-model', type=str, choices=dino_archs, required=True, help='DINO model to use')
    parser.add_argument('--batch-size', type=int, default=8, help='Feed volume through network in batches')
    parser.add_argument('--annotations-per-class', type=int, default=8, help='Number of annotated features to use per class')
    parse_basics(parser)
    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    typ = torch.float16 if args.fp16 == 'true' else torch.float32

    cache_path = Path(args.data.replace('volume', f'{args.dino_model}_feats'))
    data_path = Path(args.data)
    patch_size = 8 if args.dino_model[-1] == '8' else 16
    # Load cached features
    if Path(cache_path).exists():
        print(f'Found existing feature maps: {cache_path.name}')
        data = torch.load(cache_path)
        q, k, v = data['q'], data['k'], data['v']
    else: # Compute and save features
        print(f'Computing feature maps for {data_path.name}')
        data = torch.load(data_path)

        model = get_dino_model(args.dino_model).to(dev)
        feat_out = defaultdict(list)
        def hook_fn_forward_qkv(module, input, output):
            feat_out['qkv'].append(output.cpu())
        model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

        vol = data['vol'].float().squeeze()

        ic(vol)
        image = make_4d(vol).permute(3, 0, 1, 2).expand(-1, 3, -1, -1)
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
        k = k[:, 1:].reshape(nb_im, f_sz[0], f_sz[1], -1).permute(0,3,1,2)
        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q = q[:, 1:].reshape(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v = v[:, 1:].reshape(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
        ic(q)
        ic(k)
        ic(v)
        torch.save({'q': q, 'k': k, 'v': v}, cache_path)
        print(f'Saved feature maps: {cache_path.name}')
        del model

    # Load Data, Mask
    data = torch.load(data_path)
    mask = data['mask'][...,:500]
    ic(mask)
    ic(k)
    k = k[:500]
    # Choose some annotated voxels randomly
    non_bg_indices = [(mask == i).nonzero() for i in range(len(data['labels']))]
    class_queries = torch.stack([
        clas_idcs[torch.from_numpy(np.random.choice(clas_idcs.size(0), args.annotations_per_class))]
        for clas_idcs in non_bg_indices if clas_idcs.size(0) > 0], dim=0)
    ic(class_queries) # (C, A, 3)
    num_classes = class_queries.size(0)
    label_dict = {i: n for i, n in enumerate(data['labels'])}
    vol_u8 = (norm_minmax(data['vol'][...,:500].float()) * 255.0).to(torch.uint8)

    # Samplpe query points
    vol_extent = torch.tensor([[[*mask.shape]]], dtype=torch.float32, device=class_queries.device)-1.0
    rel_coords = (class_queries.float() / vol_extent) * 2.0 - 1.0
    # Reshape features to volume
    k_vol = k.permute(1,2,3,0).unsqueeze(0).to(dev) # (1, F, W, H, D)
    grid_idx = make_5d(rel_coords).to(dev).to(typ) # (1, 1, C, A, 3)
    queried_feats = F.grid_sample(k_vol, grid_idx, mode='bilinear', align_corners=True).squeeze(0).squeeze(1).permute(1,2,0).contiguous()
    queried_feats = F.normalize(queried_feats, dim=-1)
    qf2 = sample_features(k_vol, class_queries[1,0], rel_coords[1,0])
    ic(queried_feats)
    print(queried_feats[1,0,:10])
    ic(qf2)
    print(qf2.squeeze()[:10])
    print(torch.allclose(queried_feats[[1],[0]], qf2))

    for i in range(num_classes):
        feat_vol = F.normalize(k_vol.squeeze(0), dim=0) # (F, H, W, D)
        sims = torch.einsum('fhwd,af->ahwd', (feat_vol, queried_feats[i])) ** 2
        sims = sims.mean(dim=0)
        query_pos = class_queries[i,0].tolist()
        ic(sims)
        ic(query_pos[2])
        sim = sims[None, None, :, :, query_pos[2]]
        ic(sim)
        sim = F.interpolate(sim, scale_factor=float(patch_size), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
        ic(sim)
        fig = plot_sims(vol_u8[:,:,query_pos[2]], sim, mask[:,:,query_pos[2]], coord=query_pos, label=f'Label: {label_dict[i]}')
        fig.savefig(f'figures3/{label_dict[i]}.png')
        plt.close(fig)
    sys.exit(0)
    # queried_feats = F.normalize()
    #######
    coord_abs = torch.tensor([256,256,250])# class_queries[1,0]
    coord_rel = torch.tensor([0,0,250.0/501.0 *2.0 - 1.0])#rel_coords[1,0]
    print('feature indexing', coord_abs[2].item(), ':10', coord_abs[0].item() // 8,
                  coord_abs[1].item() // 8)
    feats_abs = k[coord_abs[2].item(), :, coord_abs[0].item() // 8,
                  coord_abs[1].item() // 8].float()
    print(feats_abs[:10])
    feats_ups = F.interpolate(make_4d(k[coord_abs[2].item()]).float(), scale_factor=8.0, mode='bilinear', align_corners=True)
    feat_up = feats_ups[0, :, coord_abs[0].item(), coord_abs[1].item()]
    ic(feat_up)
    print(feat_up[:10])
    # rel_coords2 = rel_coords[None, [1], [0], None, :]
    rel_coords2 = make_5d(coord_rel)
    k_vol = k.permute(1,2,3,0).unsqueeze(0) # (1, F, W, H, D)
    ic(k_vol)
    ic(make_5d(rel_coords2))
    print(rel_coords2.squeeze())
    query_features = F.grid_sample(k_vol.float(), make_5d(rel_coords2), mode='bilinear', align_corners=True)
    ic(query_features)   #   (1, F, 1, 1, 1) -> (W, H, F)
    query_features = query_features.squeeze(4).squeeze(0).permute(1,2,0).contiguous()
    ic(query_features)
    print(query_features.squeeze()[:10])
    sys.exit(0)

    ######
    print(rel_coords)
    query_features = F.grid_sample(k.permute(1,2,3,0)[None].float(), rel_coords[None, :, :, None], mode='bilinear')
    ic(query_features)
    #            (1, F, C, A, 1)            -> (F, C, A) -> (C, A, F)
    query_features = query_features.squeeze(4).squeeze(0).permute(1,2,0).to(typ).to(dev)
    query_features = F.normalize(query_features, dim=-1)
    ic(query_features)

    # Compute Similarity Maps
    ic(num_classes)
    ic(class_queries)
    
    # for i in range(num_classes):
    #     for X,Y,Z in class_queries[i].tolist():
    #         feat_map = F.normalize(F.interpolate(k[[Z]].to(dev), scale_factor=float(patch_size), mode='bilinear').squeeze(0), dim=0)
    #         #query_feat = feat_map[..., X,Y]
    #         label_idx = mask[X, Y, Z].item()
    #         query_clas = label_dict[label_idx]
    #         sims = torch.mean(torch.einsum('fhw,af->ahw', (feat_map, F.normalize(query_features[label_idx], dim=-1)))**2, dim=0)
    #         fig = plot_sims(vol_u8[:,:,Z], sims, mask[:,:,Z], coord=(X,Y,Z), label=f'Label: {query_clas}')
    #         fig.savefig(f'figures2/similarity_{X}_{Y}_{Z}.png')
    #         plt.close(fig)

    # Get Mask, find features corresponding to labels
    # for Z in range(100, 401, 40):
    #     feat_map = F.interpolate(k[[Z]].to(dev), scale_factor=float(patch_size), mode='bilinear')
    #     for X in range(100, 401, 40):
    #         for Y in range(100, 401, 40):
    #             query_feat = feat_map[...,X,Y]
    #             query_clas = label_dict[mask[X,Y,Z].item()]
    #             sims = torch.einsum('chw,c->hw', (F.normalize(feat_map[0], dim=0), F.normalize(query_feat[0], dim=0)))
    #             fig = plot_sims(vol_u8[:,:,Z], sims, mask[:,:,Z], coord=(X,Y,Z), label=f'Label: {query_clas}')
    #             fig.savefig(f'figures/similarity_{X}_{Y}_{Z}.png')
    #             plt.close(fig)
