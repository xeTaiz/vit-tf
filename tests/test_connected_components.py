import torch
import torch.nn.functional as F
import numpy as np
from icecream import ic
from cc_torch import connected_components_labeling

import domesutils
from argparse import ArgumentParser
from pathlib import Path

from skimage.color import label2rgb
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = ArgumentParser("Connected Components")
    parser.add_argument('--data', type=str, help='Path to data')
    args = parser.parse_args()

    dir = Path(args.data)
    similarities = np.load(dir / 'similarities.npy', allow_pickle=True)[()]

    sim_largest_island = {}
    for name, sim in similarities.items():
        print(f'Finding islands for {name}: {sim.shape}, {sim.dtype}')
        ic(sim)
        sim_thresh = torch.as_tensor(255 * (sim > 69)).to(torch.uint8).cuda()
        ic(sim_thresh)
        labels = connected_components_labeling(sim_thresh)
        ic(labels)
        print(f'Found {labels.max()} islands, unique labels: {len(labels.unique())}')
        largest_island = torch.zeros_like(sim_thresh)
        uniq, sizes = labels.unique(sorted=True, return_counts=True)
        ic(uniq)
        print(uniq[:10])
        ic(sizes)
        print(sizes[:10])
        largest_island_idx = sizes[1:].argmax() + 1
        largest_size = sizes[largest_island_idx]
        largest_island = (labels == uniq[largest_island_idx]).cpu()
        #for i in labels.unique():
        #    if i == 0: continue
        #    mask = labels == i
        #    size = mask.sum()
        #    if size > largest_size:
        #        largest_size = size
        #        largest_island = mask.cpu()
        rgb = label2rgb(largest_island.cpu().numpy()[:,:,161], bg_label=0, image=sim.cpu().numpy()[:,:,161])
        ic(rgb)
        print(f'Largest island size: {largest_size}')
        ic(largest_island)
        ic(sim)
        sim_largest_island[name] = torch.zeros_like(sim).cpu()
        sim_largest_island[name][largest_island] = sim[largest_island].cpu()
        ic(sim_largest_island[name])
        fig, ax = plt.subplots(1,3, dpi=200, tight_layout=True)
        ax[0].imshow(sim.cpu().numpy()[:,:,161])
        ax[1].imshow(largest_island.numpy()[:,:,161])
        ax[2].imshow(rgb)
        fig.savefig(dir / f'compare_islands_{name}.png')

    np.save(dir / 'similarities_largest_island.npy', sim_largest_island)

