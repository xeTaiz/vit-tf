import torch
import torch.nn.functional as F
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

def compute_and_save_qkv(data_path, slice_along='z', dev=torch.device('cpu'), typ=torch.float32):
    patch_size = 8
    cache_path = data_path.replace('volume', f'vits8_feats_along_{slice_along}')
    data = torch.load(data_path)

    model = get_dino_model('vits8').to(dev)
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
    permute_in, permute_out = slice_along[slice_along]
    image = make_4d(vol).permute(*permute_in).expand(-1, 3, -1, -1)
    image = normalize(norm_minmax(image), in_mean, in_std)
    im_sz = image.shape[-2:]

    # forward pass
    out = []
    with torch.cuda.amp.autocast(enabled=True, dtype=typ):
        with torch.no_grad():
            im_in = image.to(dev)
            for batch in torch.arange(im_in.size(0)).split(2):
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
    qkv = {
        'q': q.permute(*permute_out).contiguous(),
        'k': k.permute(*permute_out).contiguous(),
        'v': v.permute(*permute_out).contiguous()
    }
    torch.save(qkv, cache_path)
    print(f'Saved feature maps: {cache_path.name}')

    del model
    return qkv
