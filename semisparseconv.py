import torch
import torch.nn.functional as F

def gather_receiptive_fields(volume, centers, ks=3):
    L = ks // 2
    R = L+1
    pad_vol = F.pad(volume, tuple([ks]*6))
    return torch.stack([pad_vol[...,
            coord[0]-L:coord[0]+R,
            coord[1]-L:coord[1]+R,
            coord[2]-L:coord[2]+R] for coord in centers + ks
    ]).contiguous()
