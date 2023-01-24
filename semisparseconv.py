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

def gather_receiptive_fields2(volume, centers, ks=3):
    L = ks // 2
    R = L+1
    offsets = [[i,j,k] for i in range(-L,L+1) for j in range(-L, L+1) for k in range(-L, L+1)]
    pad_vol = F.pad(volume, tuple([ks]*6))
    return torch.stack([pad_vol[...,
            centers + off[0],
            coord[1]-L:coord[1]+R,
            coord[2]-L:coord[2]+R] for coord in centers + ks
    ]).contiguous()
