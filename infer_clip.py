import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from functools import partial
from argparse import ArgumentParser
from pathlib import Path
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from lavis.models import load_model_and_preprocess
from infer import make_4d, make_5d, load_data, handle_output_path

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

def load_lavis_model(method, arch, dev='cuda'):
    return load_model_and_preprocess(name=method, model_type=arch, is_eval=True, device=dev)

def load_medclip_model(arch, dev='cuda'):
    proc = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).from_pretrained().to(dev)
    return model, partial(proc, return_tensors="pt", padding=True)

def load_model(args, dev):
    method, arch = args.model.split('/')
    if method == 'medclip':
        pass
    elif method == 'openclip':
        pass
    else:
        model, vis_proc, txt_proc = load_lavis_model(method, arch, dev)
    return model, vis_proc, txt_proc

if __name__ == '__main__':
    clip_methods = ['medclip', 'openclip', 'clip', 'blip', 'blip2']
    from pathlib import Path
    from argparse import ArgumentParser
    import os, sys

    parser = ArgumentParser('Infer DINO features from saved volume')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the saved volume')
    parser.add_argument('--cache-path', type=str, default=None, help='Path to save computed qkv features to.')
    parser.add_argument('--model', type=str, default='clip/ViT-L-14', help='Method/Architecture to use')
    parser.add_argument('--slice-along', type=str, choices=['x', 'y', 'z', 'all'], default='all', help='Along which axis to slice volume, as it is fed slice-wise to DINO')
    parser.add_argument('--batch-size', type=int, default=1, help='Feed volume through network in batches')
    parser.add_argument('--feature-output-size', type=int, default=64, help='Produces a features map with aspect ratio of input volume with this value as y resolution. Only if --slice-along ALL')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing cache files')
    args = parser.parse_args()

    from lavis.models import model_zoo
    print(model_zoo)
    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    typ = torch.float16 if dev == torch.device('cuda') else torch.float32
    cache_path = handle_output_path(args)

    model, vis_proc, txt_proc = load_model(args, dev)
    print(model)

    feat_out = []
    def hook_fn_forward_qkv(module, input, output):
        print('output', output[0].shape)
        feat_out.append(output[0].cpu().half())
    model.visual_encoder.blocks[-1].mlp.register_forward_hook(hook_fn_forward_qkv)
    print(vis_proc['eval'].transform)
    inp = { "image": vis_proc['eval'].transform.transforms[-1](torch.randn(1,3,224,224)).to(dev),
            "text_input": [txt_proc['eval']("test")] }
    im_feat = model.extract_features(inp)
    print('proj', {k: v.shape for k,v in im_feat.items()})
    patch_size = 16
    sys.exit(0)
    with torch.no_grad():
        vol = load_data(args.data_path) # Load volume
        # Compute Input Image Size
        ref_fact = sorted(vol.shape[-3:])[1] / args.feature_output_size
        im_sz       = tuple(map(lambda d: int(patch_size * (d // ref_fact)), vol.shape[-3:]))
        feat_out_sz = tuple(map(lambda d: d // patch_size, im_sz))
        print(f'Input image size: {im_sz}')

        # Compute Features
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
