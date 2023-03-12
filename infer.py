import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import normalize
from collections import defaultdict

def make_nd(t, n):
    '''  Prepends singleton dimensions to `t` until n-dimensional '''
    if n < t.ndim:
        raise Exception(f'make_nd cannot reduce cardinality. Your Tensor.ndim={t.ndim} > n={n}.')
    elif n == t.ndim: 
        return t
    else:
        nons = [None]*(n-t.ndim)
        return t[nons]

def make_4d(t):
    '''  Prepends singleton dimensions to `t` until 4D '''
    return make_nd(t, 4)

def make_5d(t):
    '''  Prepends singleton dimensions to `t` until 5D '''
    return make_nd(t, 5)

def norm_minmax(t):
    mi, ma = t.min(), t.max()
    return (t - mi) / (ma - mi)

def norm_mean_std(t, mu=0, std=1):
    return (t.float() - t.float().mean()) * std / t.float().std() + mu

in_mean = [0.485, 0.456, 0.406]
in_std = [0.229, 0.224, 0.225]

def get_dino_model(name):
    return torch.hub.load('facebookresearch/dino:main', f'dino_{name}')

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

def resample_topk(feat_vol, sims, K=8, similarity_exponent=2.0, feature_sampling_mode='nearest'):
    ''' Re-samples the feature volume at the `K` most similar locations.
    
    Args:
        feat_vol (Tensor): (Normalized) Feature Volume. Shape (F, W, H, D)
        sims (Tensor): Similarity Volume for classes C with A annotations. Shape (C, A, W, H, D)
        K (int): Number of new samples to draw per annotation (and per class)
        similarity_exponent (float): Exponent to sharpen similarity maps (sim ** exponent)
        feature_sampling_mode (str): PyTorch grid_sample interpolation mode for sampling features

    Returns:
        Tensor: Similarity Volume of shape (C, A, W, H, D)
    '''
    if K > 4: 
        dev, typ = torch.device('cpu'), torch.float32
    else:
        dev, typ = feat_vol.device, feat_vol.dtype
    top_ks = []
    for s in sims.reshape(-1, sims.size(2), sims.size(3), sims.size(4)):
        top_idxs = torch.topk(s.flatten(), K, largest=True, sorted=True).values[-1]
        top_idxs_nd = (s >= top_idxs).nonzero()[:K]
        top_ks.append(top_idxs_nd)
    top_ks = torch.stack(top_ks).reshape(*sims.shape[:2], K, 3)
    rel_top_ks = (top_ks.float() + 0.5) / torch.tensor([[[sims.shape[-3:]]]], device=top_ks.device) * 2.0 - 1.0
    c, a, k, _ = top_ks.shape
    qf2 = sample_features3d(feat_vol, rel_top_ks.view(c, a*k, 3), mode=feature_sampling_mode)
    qf2 = qf2.reshape(c, a, k, qf2.size(-1))
    sims = torch.einsum('fwhd,cakf->cakwhd', (feat_vol.to(dev).to(typ), qf2.to(dev).to(typ))).clamp(0, 1) ** similarity_exponent
    return sims.mean(dim=2).to(feat_vol.dtype).to(feat_vol.device)

def compute_qkv(vol, batch_size=1, slice_along='z', dev=torch.device('cpu'), typ=torch.float32):
    patch_size = 8

    model = get_dino_model('vits8').to(dev)
    model.eval()
    feat_out = []
    def hook_fn_forward_qkv(module, input, output):
        feat_out.append(output.cpu().half())
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    vol = vol.float().squeeze()
    slice_along_permutes = {
        'z': ((3, 0, 1, 2), (1,2,3,0)),
        'y': ((2, 0, 1, 3), (1,2,0,3)),
        'x': ((1, 0, 2, 3), (1,0,2,3))
    }
    permute_in, permute_out = slice_along_permutes[slice_along]
    image = make_4d(vol).permute(*permute_in).expand(-1, 3, -1, -1)
    image = normalize(norm_minmax(image), in_mean, in_std)
    im_sz = image.shape[-2:]

    print('Network Input:', image.shape, image.dtype, image.min(), image.max())

    # forward pass
    out = []
    with torch.cuda.amp.autocast(enabled=True, dtype=typ):
        with torch.no_grad():
            im_in = image.to(dev)
            for batch in torch.arange(im_in.size(0)).split(batch_size):
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
        torch.cat(feat_out[:-1])
        .view(nb_im, nb_tokens, 3, nh, -1 // nh)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0].cpu(), qkv[1].cpu(), qkv[2].cpu()
    f_sz = im_sz[0] // patch_size , im_sz[1] // patch_size
    k = k.transpose(1, 2).view(nb_im, nb_tokens, -1)
    k = k[:, 1:].view(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
    q = q.transpose(1, 2).view(nb_im, nb_tokens, -1)
    q = q[:, 1:].view(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
    v = v.transpose(1, 2).view(nb_im, nb_tokens, -1)
    v = v[:, 1:].view(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
    qkv = {
        'q': q.permute(*permute_out),
        'k': k.permute(*permute_out),
        'v': v.permute(*permute_out)
    }
    return qkv

if __name__ == '__main__':
    dino_archs = ['vits16', 'vits8', 'vitb16', 'vitb8',
        'xcit_small_12_p16', 'xcit_small_12_p8',
        'xcit_medium_24_p16', 'xcit_medium_24_p8',
        'resnet50']
    from pathlib import Path
    from argparse import ArgumentParser
    import os, sys

    def is_path_creatable(pathname: str) -> bool:
        '''
        `True` if the current user has sufficient permissions to create the passed
        pathname; `False` otherwise.
        '''
        # Parent directory of the passed path. If empty, we substitute the current
        # working directory (CWD) instead.
        dirname = os.path.dirname(pathname) or os.getcwd()
        return os.access(dirname, os.W_OK)

    parser = ArgumentParser('Infer DINO features from saved volume')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the saved volume')
    parser.add_argument('--cache-path', type=str, required=True, help='Path to save computed qkv features to.')
    parser.add_argument('--dino-model', type=str, choices=dino_archs, default='vits8', help='DINO model to use')
    parser.add_argument('--slice-along', type=str, choices=['x', 'y', 'z', 'all'], default='z', help='Along which axis to slice volume, as it is fed slice-wise to DINO')
    parser.add_argument('--batch-size', type=int, default=2, help='Feed volume through network in batches')
    parser.add_argument('--feature-downsize-factor', type=int, default=4, help='Rescales the feature map to input volume extent // this factor. With a ViTs8 this needs to be 8 to prevent upscaling of feature maps. Only if --slice-along ALL')
    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    typ = torch.float16 if dev == torch.device('cuda') else torch.float32
    patch_size = 8 if args.dino_model[-1] == '8' else 16

    data_path = Path(args.data_path)
    cache_path = Path(args.cache_path)

    if not data_path.exists():
        print(f'Invalid argument for --data-path (File does not exist): {args.data_path}')
        sys.exit(1)
    if not is_path_creatable(args.cache_path):
        print(f'Invalid argument for --cache-path (Cannot write to location): {args.cache_path}')
        sys.exit(1)

    with torch.no_grad():
        if data_path.suffix in ['.pt', '.pth']:
            data = torch.load(data_path)
            if type(data) == dict:
                vol = data['vol']
            else:
                vol = data
            print(f'Loaded volume: {vol.shape} of type {vol.dtype}.')
            assert vol.ndim == 3
        elif data_path.suffix == '.npy':
            data = np.load(data_path, allow_pickle=True)
            if data.dtype == "O":
                vol = torch.from_numpy(data[()]['vol'].astype(np.float32))
            else:
                vol = torch.from_numpy(data.astype(np.float32))
            print(f'Loaded volume: {vol.shape} of type {vol.dtype}.')
            assert vol.ndim == 3
        else:
            print(f'Unsupported file extension: {data_path.suffix}')
        if args.slice_along in ['x', 'y', 'z']:
            qkv = compute_qkv(vol, batch_size=args.batch_size, slice_along=args.slice_along, dev=dev, typ=typ)
        elif args.slice_along == 'all':
            feat_res = tuple(map(lambda d: int(d // args.feature_downsize_factor), vol.shape))
            # feat_res = (args.feature_resolution, args.feature_resolution, args.feature_resolution)
            qkv = {'q': 0, 'k': 0, 'v': 0}
            for ax in ['x', 'y', 'z']:
                for k,v in compute_qkv(vol, batch_size=args.batch_size, slice_along=ax, dev=dev, typ=typ).items():
                    qkv[k] += (F.interpolate(make_5d(v).float(), feat_res, mode='trilinear', align_corners=False).squeeze(0) / 3.0).half()
        else:
            raise Exception(f'Invalid argument for --slice-along: {args.slice_along}. Must be x,y,z or all')
        print(f'Computed qkv, saving now to: {cache_path}')
        if cache_path.suffix in ['.pt', '.pth']:
            torch.save(qkv, cache_path)
        elif cache_path.suffix == '.npy':
            np.save(cache_path, {k: v.numpy() for k,v in qkv.items()})
    
    sys.exit(0)
