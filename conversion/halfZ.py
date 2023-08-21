import numpy as np
import torch
import torch.nn.functional as F

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser("Half Z dimension")
    parser.add_argument('--data', type=str, help='Path to data')
    args = parser.parse_args()

    vol = np.load(args.data, allow_pickle=True)[()]
    print(f'Halfing Z for volume {args.data} from {vol.shape} to {(vol.shape[0], vol.shape[1], vol.shape[2]//2)}')
    assert(vol.ndim == 3)
    assert(vol.shape[2] > vol.shape[0] and vol.shape[2] > vol.shape[1])
    vol2 = F.interpolate(torch.as_tensor(vol)[None, None], size=(vol.shape[0], vol.shape[1], vol.shape[2]//2), mode='nearest').squeeze().numpy()
    np.save(args.data.replace('.npy', '_halfZ.npy'), vol2)
