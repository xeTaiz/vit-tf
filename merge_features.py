import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser("Merge features")
    parser.add_argument('--data', type=str, required=True, help='Directory containing all data')
    parser.add_argument('--name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--output', type=str, required=False, default=None, help='Output name')
    args = parser.parse_args()

    dir = Path(args.data)
    assert dir.exists() and dir.is_dir()
    files = sorted(list(filter(lambda n: '_x_' in n.name or '_y_' in n.name or '_z_' in n.name, dir.rglob(args.name))))
    assert len(files) == 3
    if args.output is None:
        args.output = files[0].name.replace('_x_', '_merged_')

    x, y, z = list(map(lambda f: torch.from_numpy(np.load(dir/f, allow_pickle=True)[()]['k']), files))
    print('x', x.shape)
    print('y', y.shape)
    print('z', z.shape)
    feat_sz = (z.size(1), z.size(2), x.size(3))
    f  = (F.adaptive_avg_pool3d(x[None].float(), feat_sz) / 3).half()
    f += (F.adaptive_avg_pool3d(y[None].float(), feat_sz) / 3).half()
    f += (F.adaptive_avg_pool3d(z[None].float(), feat_sz) / 3).half()

    np.save(dir/args.output, f.numpy())
