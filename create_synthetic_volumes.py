import torch
import torch.nn.functional as F
import numpy as np

from argparse import ArgumentParser
from pathlib import Path

def sdf_sphere(pos, r):
    return torch.norm(pos, dim=-1) - r

def sphere_thickness(pos, r, t):
    return (torch.abs(sdf_sphere(pos, r)) < t).float()

def sphere_filled(pos, r):
    return (sdf_sphere(pos, r) <= 0).float()

def sdf_torus(pos, r1, r2):
    q = torch.norm(pos[..., :2], dim=-1) - r1
    return torch.norm(torch.cat([q[..., None], pos[..., 2:]], dim=-1), dim=-1) - r2

def torus_thickness(pos, r1, r2, t):
    return (torch.abs(sdf_torus(pos, r1, r2)) < t).float()

def torus_filled(pos, r1, r2):
    return (sdf_torus(pos, r1, r2) <= 0).float()


def main():
    parser = ArgumentParser()
    parser.add_argument('outdir', type=Path, help='Output directory')
    parser.add_argument('--size', type=int, default=128, help='Volume size')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise standard deviation')
    parser.add_argument('--torch', action='store_true', help='Save as torch tensors (.pt files)')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    def noise_clamp(vol):
        return torch.clamp(vol + torch.rand_like(vol) * args.noise, 0, 1)

    def save_vol(vol, name):
        if args.torch:
            torch.save(vol.to(torch.float16), outdir / f'{name}.pt')
        else:
            np.save(outdir/f'{name}.npy', vol.numpy().astype(np.float16))

    def save_label(vol, name):
        if args.torch:
            torch.save((vol > 0.5).to(torch.uint8), outdir / f'{name}_label.pt')
        else:
            np.save(outdir/f'{name}_label.npy', (vol > 0.5).numpy().astype(np.uint8))

    ls = torch.linspace(-1, 1, args.size)
    pos = torch.stack(torch.meshgrid(ls, ls, ls, indexing='xy'), dim=-1)

    spheret = sphere_thickness(pos, 0.5, 0.05)
    spheref = sphere_filled(pos, 0.5)
    torust = torus_thickness(pos, 0.5, 0.2, 0.05)
    torusf = torus_filled(pos, 0.5, 0.2)

    save_vol(noise_clamp(spheret), 'sphere_thick')
    save_label(spheret, 'sphere_thick')
    save_vol(noise_clamp(spheref), 'sphere_filled')
    save_label(spheref, 'sphere_filled')
    save_vol(noise_clamp(torust), 'torus_thick')
    save_label(torust, 'torus_thick')
    save_vol(noise_clamp(torusf), 'torus_filled')
    save_label(torusf, 'torus_filled')

if __name__ == '__main__':
    main()
