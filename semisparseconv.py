import torch
import torch.nn.functional as F

def gather_receiptive_fields(volume, centers, ks=3):
    L = ks // 2
    R = L+1
    pad_vol = F.pad(volume, tuple([ks//2]*6))
    return torch.stack([pad_vol[...,
            coord[0]-L:coord[0]+R,
            coord[1]-L:coord[1]+R,
            coord[2]-L:coord[2]+R] for coord in centers + ks//2
    ]).contiguous()

def gather_receiptive_fields2(volume, centers, ks=3):
    L = ks // 2
    offsets = [[L+i,L+j,L+k] for i in range(-L, L+1) for j in range(-L, L+1) for k in range(-L, L+1)]
    pad_vol = F.pad(volume, tuple([L]*6))
    return torch.stack([pad_vol[...,
            centers[:, 0] + off[0],
            centers[:, 1] + off[1],
            centers[:, 2] + off[2]] for off in offsets
    ]).permute(2,1,0).reshape(centers.size(0), volume.size(0), ks, ks, ks).contiguous()


# Benchmark the above variants (takes ~5min with the profiler on)
if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    NUM_BRICKS = int(2**16)
    DEV = torch.device('cuda')
    DTYP = torch.half

    ONE = torch.ones(1, device=DEV)
    vol = torch.rand(1, 200,200,200, device=DEV, dtype=DTYP)
    centers = vol.squeeze(0).nonzero()[torch.multinomial(ONE.expand(vol.numel()), NUM_BRICKS)].to(DEV)

    print(f'Indexing {centers.size(0)}  blocks  from  a  {tuple(vol.shape)} volume.')
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                record_shapes=False, profile_memory=False) as prof:
        with record_function('gather_receiptive_fields1_9x9'):
            rec_fields1 = gather_receiptive_fields(vol, centers, ks=9)
        with record_function('gather_receiptive_fields2_9x9'):
            rec_fields2 = gather_receiptive_fields2(vol, centers, ks=9)

    print(prof.key_averages().table(sort_by="cuda_time_total"))
    print('All close?: ', torch.allclose(rec_fields1, rec_fields2))
