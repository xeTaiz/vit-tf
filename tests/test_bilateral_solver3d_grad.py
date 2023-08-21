import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

from icecream import ic

from torchvtk.utils import make_4d
from domesutils import *
from bilateral_solver3d import apply_bilateral_solver3d

if __name__ == "__main__":
    vol = np.load()



    sims = {n.replace('.pt', ''): torch.load(f'similarity_images/{n}', map_location='cpu') for n in os.listdir('similarity_images') if n not in ['background.pt', 'bladder.pt']}
    vol_u8 = (255.0 * norm_minmax(torch.load('data/CT-ORG/volume_010.pt')['vol'].float())).to(torch.uint8)
    mask = torch.load('data/CT-ORG/volume_010.pt')['mask']
    IDX = 400
    mask_slices = {n: mask[:,:,IDX] for n,v in sims.items()}
    vol_slices = {n: vol_u8[:,:,IDX] for n,v in sims.items()}
    sim_slices = {n: v[:,:,IDX].float() for n,v in sims.items()}
    print({k:v.shape for k,v in sim_slices.items()})
    # Run bilateral solver
    grid_params = {
        'sigma_luma': 4,  # Brightness bandwidth
        'sigma_chroma': 4,  # Color bandwidth
        'sigma_spatial': 8  # Spatial bandwidth
    }
    bil_slices, thresh_sims = {}, {}
    resolution = (512, 512)
    vol_slice = F.interpolate(make_4d(vol_slices['bone'][None].expand(3, -1, -1)), resolution, mode='nearest').squeeze(0)
    for n,v in sim_slices.items():
        print(f'{n}:')
        sim_slice_float = F.interpolate(make_4d(sim_slices[n]), resolution, mode='nearest').squeeze(0)
        thresh = 0.75 * sim_slice_float.max().item()# sim_slice_float.quantile(0.85).item()
        ic(thresh)
        # sim_slice = (sim_slice > thresh).to(torch.uint8)
        sim_slice_float[sim_slice_float < thresh] = torch.clamp((sim_slice_float[sim_slice_float < thresh] + (1.0 - thresh))** 2 - (1.0 - thresh), 0, 1)
        # sim_slice_float[sim_slice_float < thresh] = 0
        sim_slice = (255.0 * sim_slice_float).to(torch.uint8)
        ic(sim_slice_float)
        ic(vol_slice)
        bin_sim, cont_sim = apply_bilateral_solver(sim_slice, vol_slice, c=(sim_slice_float+1)/2, grid_params=grid_params)
        ic(cont_sim)
        ic(bin_sim)
        thresh_sims[n] = sim_slice.squeeze()
        # cont_sim[cont_sim < 255*0.5] = 0.0
        bil_slices[n] = (bin_sim, norm_minmax(cont_sim))



    # Plot stuff
    fig, ax = plt.subplots(4,6, figsize=(30,20), tight_layout=True)
    for x in ax.reshape(-1): x.set_axis_off()
    for i, n in enumerate(sim_slices.keys()):
        orig_slice = vol_slices[n]
        avg_sim = sim_slices[n]
        thresh_sim = thresh_sims[n]
        bil_sim_bin, bil_sim_cont = bil_slices[n]
        mask = mask_slices[n]
        ax[i, 0].imshow(orig_slice)
        ax[i, 0].set_title("Raw Data")
        ax[i, 1].imshow(avg_sim)
        ax[i, 1].set_title(f"Average Similarity: {n}")
        ax[i, 2].imshow(thresh_sim)
        ax[i, 2].set_title(f"Thresholded Similarity: {n}")
        ax[i, 3].imshow(bil_sim_bin)
        ax[i, 3].set_title(f'Bilaterally Solved (Binary): {n}')
        ax[i,4].imshow(bil_sim_cont)
        ax[i,4].set_title(f'Bilaterally Solved (Continuous): {n}') 
        ax[i,5].imshow(mask)
        ax[i,5].set_title("Ground Truth Segmentation")

    fig.savefig('bilateral_comparison.png')
