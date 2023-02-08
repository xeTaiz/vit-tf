import torch
import torch.nn.functional as F
import torch.nn as nn
from argparse import ArgumentParser

from torchvtk.utils import make_4d, make_5d
from semisparseconv import gather_receiptive_fields2 as gather_receiptive_fields
from train_semisparse import create_cnn
from utils import *

class Identity(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, x): return x

if __name__ == '__main__':
    parser = ArgumentParser('Compare Sprase Conv with standard')
    parser.add_argument('--data', type=str, default=None, help='Path to some .pt with vol data')
    parser.add_argument('--resolution', type=int, default=100, help='Resolution of test volume')
    parser.add_argument('--cnn-layers', type=str, default=None, help='Conv layers in test network')
    parser.add_argument('--linear-layers', type=str, default=None, help='Linear layers after convs (1x1 convs)')
    args = parser.parse_args()
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Get Data
    if args.data:
        vol = make_4d(torch.load(args.data)['vol'].float()).to(dev)
        vol = (vol - vol.mean()) / vol.std()
    else:
        vol = torch.randn(1, args.resolution, args.resolution, args.resolution).float().to(dev)
    # Create Model
    args.cnn_layers    = [int(n.strip()) for n in    args.cnn_layers.replace('[', '').replace(']', '').split(' ')] if args.cnn_layers    else [8, 16, 32]
    args.linear_layers = [int(n.strip()) for n in args.linear_layers.replace('[', '').replace(']', '').split(' ')] if args.linear_layers else [32]
    NF = args.cnn_layers[-1]
    model = create_cnn(in_dim=vol.size(0), n_features=args.cnn_layers, n_linear=args.linear_layers, Norm=Identity).to(dev)
    print(model)
    REC_FIELD = len(args.cnn_layers) * 2 + 1
    PAD = tuple([REC_FIELD // 2]*6)
    CENTER = REC_FIELD //2

    # Get all voxel locations
    indices = (vol.squeeze() < 1e10).nonzero()
    crops = gather_receiptive_fields(vol, indices, ks=REC_FIELD)
    log_tensor(indices, 'indices')
    log_tensor(crops, 'crops')
    log_tensor(vol, 'vol')
    # Compare
    with torch.no_grad():
        res_dense = model(F.pad(make_5d(vol), PAD)).squeeze(0)
        res_spars = model(make_5d(crops)).squeeze().view(*vol.shape[-3:], NF).permute(3,0,1,2).contiguous()
        log_tensor(res_dense, 'res_dense')
        log_tensor(res_spars, 'res_spars')
        print('All Close?', torch.allclose(res_dense, res_spars))

        print(res_dense[0, 50, 50, 50:55])
        print(res_spars[0, 50, 50, 50:55])

        # Reshape centers of crops to original volume
        crop_centers = crops[..., CENTER, CENTER, CENTER]
        log_tensor(crop_centers, 'crop_centers')
        recon = crop_centers.view(*vol.shape[-3:], 1).permute(3,0,1,2).contiguous()
        print('Reoncstruct from crops close?', torch.allclose(recon, make_4d(vol)))
