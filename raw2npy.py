import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = ArgumentParser('Convert .raw files to .pt')
    parser.add_argument('--data', type=str, required=True, help='Path to .raw data')
    args = parser.parse_args()
    assert args.data.endswith('.raw')

    raw_path = Path(args.data)
    dat_path = Path(args.data.replace('.raw', '.dat'))
    out_path = Path(args.data.replace('.raw', '.npy'))

    raw_file = open(raw_path, 'rb')
    if dat_path.exists():
        dat_file = open(dat_path, 'r')
        print('DAT File:')
        print(dat_file.read())

    array = np.fromfile(raw_file, dtype=np.uint8, count=512*512*1873*4)
    vol = torch.from_numpy(array.reshape(1873,512,512,4)).permute(3,2,1,0)
    vol = F.interpolate(vol[None], (512,512,468), mode='nearest').squeeze(0)

    print(f'Saving volume to {out_path}')
    print(vol.shape, vol.dtype, vol.min(), vol.max())
    vol_np = np.ascontiguousarray(vol.permute(1, 2, 3, 0).numpy())
    print(f'Contiguity:    C: {vol_np.flags["C_CONTIGUOUS"]}     F: {vol_np.flags["F_CONTIGUOUS"]}')

    # Plot to verity contiguity
    fig, ax = plt.subplots(1,3, dpi=300, tight_layout=True)
    ax[0].imshow(vol_np[256, :, :,:3])
    ax[1].imshow(vol_np[:, 256, :,:3])
    ax[2].imshow(vol_np[:, :, 234,:3])
    fig.savefig('raw2npy_sample.png')
    np.save(out_path, vol_np)
