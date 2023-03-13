import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser('Convert .raw files to .pt')
    parser.add_argument('--data', type=str, required=True, help='Path to .raw data')
    args = parser.parse_args()
    assert args.data.endswith('.raw')

    raw_path = Path(args.data)
    dat_path = Path(args.data.replace('.raw', '.dat'))
    out_path = Path(args.data.replace('.raw', '.pt'))

    raw_file = open(raw_path, 'rb')
    if dat_path.exists():
        dat_file = open(dat_path, 'r')
        print('DAT File:')
        print(dat_file.read())

    array = np.fromfile(raw_file, dtype=np.uint8, count=512*512*1873*4)
    vol = torch.from_numpy(array.reshape(512,512,1873,4)).permute(3,0,1,2)
    vol = F.interpolate(vol[None], (512,512,468), mode='nearest').squeeze(0)

    print(f'Saving volume to {out_path}')
    print(vol.shape, vol.dtype, vol.min(), vol.max())

    torch.save({'vol': vol}, out_path)
