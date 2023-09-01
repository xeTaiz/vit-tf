import numpy as np
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser('Resize volume')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--resolution', type=float, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=False)
    args = parser.parse_args()

    if not args.output:
        args.output = args.data.parent / f'{args.data.stem}_resized{args.data.suffix}'

    data = torch.from_numpy(np.load(args.data).astype(np.float32))
    res = tuple(int(r) if r > 1.0 else int(r * data.shape[i]) for i, r in enumerate(args.resolution))
    print(f'Resizing {args.data} to {res}')
    out = F.interpolate(data[None,None], size=res, mode='trilinear', align_corners=False)[0,0].numpy()
    np.save(args.output, out)

