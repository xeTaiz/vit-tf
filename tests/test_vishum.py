import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from infer_multi import sample_features3d

if __name__ == '__main__':
    rgba = torch.from_numpy(np.load('/home/dome/Data/Volumes/2006 - Visible Human/visible-human-rgba.npy', allow_pickle=True))
    feats = torch.load('/home/dome/Data/Volumes/2006 - Visible Human/visible-human-rgba-FeatsALL.pt')['k']
    feats = F.normalize(feats.float(), dim=0)
    extent = list(rgba.shape[:3])

    print(rgba.shape, rgba.dtype)
    print(feats.shape, feats.dtype)
    coord = [340, 250, 200]
    rel_coords = torch.tensor([(c + 0.5) / e * 2.0 - 1.0 for c, e in zip(coord, extent)])
    slic = rgba[:,:, coord[2],:3]
    feat = feats[..., coord[0]//8, coord[1]//8, coord[2]//8]
    print('Coords:', coord, rel_coords.tolist())
    qf = sample_features3d(feats, rel_coords, mode='nearest')
    print('feat, qf, allclose()', feat.squeeze()[:3], qf.squeeze()[:3], torch.allclose(feat, qf))
    feat_slic = feats[..., :, :, coord[2]//8]
    print('feat', feat.shape, feat.dtype)
    print('qf', qf.shape, qf.dtype)

    save_image(slic, 'vishum_slice.png')

    # Compute sim
    sim = torch.einsum('fwh,f->wh', feat_slic.float(), qf.squeeze()) ** 4.0
    print('sim', sim.shape, sim.dtype, sim.min(), sim.max())

    fig, ax = plt.subplots(2,1)
    ax[0].imshow(slic)
    ax[0].plot(coord[1], coord[0], marker='x', color="white")
    ax[1].imshow(sim)
    fig.savefig('vishum_test.png')
