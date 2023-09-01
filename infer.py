import torch
import torch.nn.functional as F
import numpy as np
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

def make_3d(t):
    '''  Prepends singleton dimensions to `t` until 3D '''
    return make_nd(t, 3)

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

def get_dinov2_model(name):
    return torch.hub.load('facebookresearch/dinov2', f'dinov2_{name}')
def sample_features3d(feat_vol, rel_coords, mode='nearest'):
    '''Samples features at given coords from `feat_vol` by interpolating the feature maps in all dimensions. This should result in nearest interpolation along the un-reduced dimension

    Args:
        feat_vol (Tensor): Shape ([M,] F, W, H, D), M being modality of the feature map
        rel_coords (Tensor): Shape ([M,] C, A, 3)
        mode (str): Interpolation along 3D volume. One of: nearest, bilinear

    Returns:
        Tensor: Shape ([M,] C, A, F)

    '''
    # Flip dims to get X,Y,Z -> Z,Y,X
    if feat_vol.ndim == 4: feat_vol = make_5d(feat_vol)     # Ensure 5D
    if rel_coords.ndim == 2: rel_coords = make_4d(rel_coords) # Ensure 4D
    if rel_coords.ndim == 3: rel_coords = make_4d(rel_coords) # Ensure 4D
    if rel_coords.size(0) != feat_vol.size(0):        # Expand M dimension to feat_vol's
        rel_coords = rel_coords.expand(feat_vol.size(0),-1,-1,-1)
    rel_coords.unsqueeze_(-2) # Make 5D
    grid_idx = rel_coords.flip(dims=(-1,)).to(feat_vol.dtype).to(feat_vol.device) # (1, 1, C, A, 3)
    # print('sample_features3d: feat_vol:', make_5d(feat_vol).shape, 'grid_idx:', grid_idx.shape)
    # Maybe do make_5d(grid_idx) in the line below
    feats = F.grid_sample(make_5d(feat_vol), grid_idx, mode=mode, align_corners=False)
    # (M, F, C*A, K, 1) -> (M, C, A, F)
    return feats.squeeze(-1).permute(0,2,3,1).contiguous()


def resample_topk(feat_vol, sims, K=8, similarity_exponent=2.0, feature_sampling_mode='nearest'):
    ''' Re-samples the feature volume at the `K` most similar locations.

    Args:
        feat_vol (Tensor): (Normalized) Feature Volume. Shape ([M,] F, W, H, D)
        sims (Tensor): Similarity Volume for classes C with A annotations. Shape ([M,] C, A, W, H, D)
        K (int): Number of new samples to draw per annotation (and per class)
        similarity_exponent (float): Exponent to sharpen similarity maps (sim ** exponent)
        feature_sampling_mode (str): PyTorch grid_sample interpolation mode for sampling features

    Returns:
        Tensor: Similarity Volume of shape ([M,] C, A, W, H, D)
    '''
    if sims.ndim == 5: sims.unsqueeze_(0) # add empty M dimension
    if K > 4:
        dev, typ = torch.device('cpu'), torch.float32
    else:
        dev, typ = feat_vol.device, feat_vol.dtype
    top_ks = []
    for s in sims.reshape(-1, *sims.shape[-3:]):
        top_idxs = torch.topk(s.flatten(), K, largest=True, sorted=True).values[-1]
        top_idxs_nd = (s >= top_idxs).nonzero()[:K]
        top_ks.append(top_idxs_nd)
    top_ks = torch.stack(top_ks).reshape(*sims.shape[:-3], K, 3)
    rel_top_ks = (top_ks.float() + 0.5) / torch.tensor([[[[sims.shape[-3:]]]]], device=top_ks.device) * 2.0 - 1.0
    m, c, a, k, _ = top_ks.shape
    qf2 = sample_features3d(feat_vol, rel_top_ks.view(m, c, a*k, 3), mode=feature_sampling_mode)
    qf2 = qf2.reshape(m, c, a, k, qf2.size(-1))
    print('resample_topk() qf2:', qf2.shape)
    sims = torch.einsum('mfwhd,mcakf->mcakwhd', (feat_vol.to(dev).to(typ), qf2.to(dev).to(typ))).clamp(0, 1) ** similarity_exponent
    print('resample_topk() sims:', sims.shape)
    return sims.mean(dim=3).to(feat_vol.dtype).to(feat_vol.device)

def take_most_dissimilar(features, num_prototypes=35, measure='cosine'):
    ''' Takes the most dissimilar features from the given `features` (N, F)
        Args:
            features (Tensor): Shape (N, F)
            num_prototypes (int): Number of prototypes to take
            measure (str): One of: cosine, euclidean
        Returns:
            Tensor: Shape (num_prototypes, F)
    '''
    if features.size(0) <= num_prototypes: return features
    if measure == 'cosine':
        dist = 1 - F.cosine_similarity(features.unsqueeze(0), features.unsqueeze(1), dim=-1).squeeze(0).mean(0)  # (N,)
    elif measure == 'euclidean':
        dist = torch.cdist(features.unsqueeze(0), features.unsqueeze(0)).squeeze(0).mean(0)  # (N,)
    else:
        raise ValueError(f'Unknown measure: {measure}')
    largest_dists, selected = torch.topk(dist, num_prototypes, largest=True, sorted=False)
    print(f'Smallest distances (min: {largest_dists.min().item():.4f} avg: {largest_dists.mean().item():.4f}) vs average distance ({dist.mean().item():.4f})')
    return features[selected]

def _noop(x, **kwargs): return x

def compute_qkv(vol, model, patch_size, im_sizes, pool_fn=_noop, batch_size=1, slice_along='z', return_keys=['q', 'k', 'v'], dev=torch.device('cpu'), typ=torch.float32):
    if isinstance(return_keys, str): return_keys = [return_keys]
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
    image_sizes = {
        'z': (im_sizes[0], im_sizes[1]),
        'y': (im_sizes[0], im_sizes[2]),
        'x': (im_sizes[1], im_sizes[2])
    }
    target_slices = {
        'z': im_sizes[2] // patch_size,
        'y': im_sizes[1] // patch_size,
        'x': im_sizes[0] // patch_size
    }
    permute_in, permute_out = slice_along_permutes[slice_along]
    image = make_4d(vol).permute(*permute_in).expand(-1, 3, -1, -1)
    image = normalize(norm_minmax(image), in_mean, in_std)
    im_sz = image_sizes[slice_along]
    num_slices = image.size(0)
    targ_slices = target_slices[slice_along]
    # print('Discarding slices to get from ', num_slices, ' to ', targ_slices)
    # slice_range = F.interpolate(torch.arange(num_slices).view(1,1,-1).float(), size=targ_slices, mode='nearest').squeeze().long()
    # print(slice_range)
    # offset = (num_slices - slice_range.max()) // 2
    # slice_range += offset
    # print(slice_range)
    # image = image[slice_range]
    # print('Indices of chosen slices:', slice_range)

    feat_out_sz = tuple(map(lambda d: d // patch_size, im_sizes))
    print('Network Input (unscaled):', image.shape, image.dtype, image.min(), image.max())
    print('-> Scaled to images of shape:', im_sz, ' to get feats of size ', feat_out_sz)

    # forward pass
    with torch.cuda.amp.autocast(enabled=True, dtype=typ):
        with torch.no_grad():
            im_in = image
            for batch in torch.arange(im_in.size(0)).split(batch_size):
                _ = model(F.interpolate(im_in[batch], size=im_sz, mode='nearest').to(dev))

    # Dimensions
    nh = model._modules["blocks"][-1]._modules["attn"].num_heads
    nb_im = image.size(0) # Batch sizemonitor
    f_sz = im_sz[0] // patch_size , im_sz[1] // patch_size

    merged_feats = torch.cat(feat_out)
    print('merged_feats:', merged_feats.shape, merged_feats.dtype, merged_feats.device)
    nb_tokens = merged_feats.shape[1]
    print('Model num heads:', nh, ' num tokens:', nb_tokens)
    # Extract the qkv features of the last attention layer
    qkv = (
        merged_feats
        .view(nb_im, nb_tokens, 3, nh, -1 // nh)
        .permute(2, 0, 3, 1, 4)
    )
    out = {}
    if 'q' in return_keys:
        q = qkv[0].transpose(1, 2).view(nb_im, nb_tokens, -1)
        q = q[:, 1:].view(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
        out['q'] = pool_fn(q.permute(*permute_out).to(dev)).cpu()
        del q
    if 'k' in return_keys:
        k = qkv[1].transpose(1, 2).view(nb_im, nb_tokens, -1)
        k = k[:, 1:].view(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
        out['k'] = pool_fn(k.permute(*permute_out).to(dev)).cpu()
        del k
    if 'v' in return_keys:
        v = qkv[2].transpose(1, 2).view(nb_im, nb_tokens, -1)
        v = v[:, 1:].view(nb_im, f_sz[0], f_sz[1], -1).permute(0, 3, 1, 2)
        out['v'] = pool_fn(v.permute(*permute_out).to(dev)).cpu()
        del v
    return out

if __name__ == '__main__':
    dino_archs = ['vits16', 'vits8', 'vitb16', 'vitb8']
    dino2_archs = ['vits14', 'vitb14', 'vitl14', 'vitg14']
    from torchvision.transforms.functional import normalize
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
    parser.add_argument('--cache-path', type=str, default=None, help='Path to save computed qkv features to.')
    parser.add_argument('--dino-model', type=str, choices=dino_archs, default=None, help='DINO model to use')
    parser.add_argument('--dino2-model', type=str, choices=dino2_archs, default=None, help='DINOv2 model to use')
    parser.add_argument('--slice-along', type=str, choices=['x', 'y', 'z', 'all'], default='all', help='Along which axis to slice volume, as it is fed slice-wise to DINO')
    parser.add_argument('--batch-size', type=int, default=1, help='Feed volume through network in batches')
    parser.add_argument('--feature-output-size', type=int, default=64, help='Produces a features map with aspect ratio of input volume with this value as y resolution. Only if --slice-along ALL')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing cache files')
    args = parser.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    typ = torch.float16 if dev == torch.device('cuda') else torch.float32
    # Determine DINO model and patch size
    if not args.dino_model and not args.dino2_model:
        print('No DINO/DINOv2 model specified, using default: vits8')
        dino_model = 'vits8'
        dino_model_fn = get_dino_model
        patch_size = 8
    elif args.dino_model and args.dino2_model:
        print(f'Both --dino-model and --dino2-model were set. Please only set one of them.')
        sys.exit(1)
    elif args.dino_model:
        dino_model = args.dino_model
        dino_model_fn = get_dino_model
        patch_size = 8 if dino_model[-1] == '8' else 16
    elif args.dino2_model:
        dino_model = args.dino2_model
        dino_model_fn = get_dinov2_model
        patch_size = 14
    else:
        print('Something weird happend with --dino-model / --dino2-model. Set exactly 1 of them.')
        sys.exit(1)

    data_path = Path(args.data_path)
    if not args.cache_path:
        args.cache_path = data_path.parent / f'{data_path.stem}_{dino_model}_{args.slice_along}_features{args.feature_output_size}{data_path.suffix}'
    cache_path = Path(args.cache_path)
    if cache_path.exists() and not args.overwrite:
        print(f'Cache file already exists: {cache_path}. Use --overwrite to overwrite.')
        sys.exit(1)

    if not data_path.exists():
        print(f'Invalid argument for --data-path (File does not exist): {args.data_path}')
        sys.exit(1)
    if not is_path_creatable(args.cache_path):
        print(f'Invalid argument for --cache-path (Cannot write to location): {args.cache_path}')
        sys.exit(1)

    with torch.no_grad():
        print(f'Attempting to load {args.data_path}.')
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

        # Compute Input Image Size
        ref_fact = sorted(vol.shape[-3:])[1] / args.feature_output_size
        im_sz       = tuple(map(lambda d: int(patch_size * (d // ref_fact)), vol.shape[-3:]))
        feat_out_sz = tuple(map(lambda d: d // patch_size, im_sz))
        print(f'Input image size: {im_sz}')

        # Compute Features
        model = dino_model_fn(dino_model).to(dev).eval()
        if args.slice_along in ['x', 'y', 'z']:
            qkv = compute_qkv(vol, model, patch_size, im_sz, batch_size=args.batch_size, return_keys='k', slice_along=args.slice_along, dev=dev, typ=typ)
        elif args.slice_along == 'all':
            qkv = defaultdict(float)
            avg_pool = torch.nn.AdaptiveAvgPool3d(output_size=feat_out_sz)
            for ax in ['z', 'y', 'x']:
                for k,v in compute_qkv(vol, model, patch_size, im_sz, pool_fn=avg_pool, batch_size=args.batch_size, return_keys='k', slice_along=ax, dev=dev, typ=typ).items():
                    qkv[k] = (torch.as_tensor(qkv[k]).to(dev) + v.to(dev).squeeze().half()).cpu()
                    print(k, ':', qkv[k].shape)
        else:
            raise Exception(f'Invalid argument for --slice-along: {args.slice_along}. Must be x,y,z or all')
        print(f'Computed qkv, saving now to: {cache_path}')
        if cache_path.suffix in ['.pt', '.pth']:
            torch.save(qkv, cache_path)
        elif cache_path.suffix == '.npy':
            np.save(cache_path, {k: v.numpy() for k,v in qkv.items()})

    sys.exit(0)
