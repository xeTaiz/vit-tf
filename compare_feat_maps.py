import torch
import torch.nn.functional as F

from torchvtk.utils import make_5d

import matplotlib.pyplot as plt

if __name__ == "__main__":
    along_x = torch.load('data/CT-ORG/vits8_feats_along_x_010.pt')['k'].cuda()
    along_y = torch.load('data/CT-ORG/vits8_feats_along_y_010.pt')['k'].cuda()
    along_z = torch.load('data/CT-ORG/vits8_feats_along_z_010.pt')['k'].cuda()

    target_shape = tuple(torch.tensor([
        [*along_x.shape[1:]], [*along_y.shape[1:]], [*along_z.shape[1:]]
    ]).min(dim=0).values.tolist())
    print(target_shape)

    along_x = F.normalize(F.interpolate(make_5d(along_x), target_shape, mode='trilinear').squeeze(), dim=0)
    along_y = F.normalize(F.interpolate(make_5d(along_y), target_shape, mode='trilinear').squeeze(), dim=0)
    along_z = F.normalize(F.interpolate(make_5d(along_z), target_shape, mode='trilinear').squeeze(), dim=0)

    along_all = torch.stack([along_x, along_y, along_z]).mean(0)
    torch.save({'q': along_all.cpu(), 'k': along_all.cpu(), 'v': along_all.cpu()}, 
        'data/CT-ORG/vits8_feats_along_all_010.pt')

    sim_xy = torch.einsum('fwhd,fwhd->whd', (along_x, along_y))
    sim_xz = torch.einsum('fwhd,fwhd->whd', (along_x, along_z))
    sim_yz = torch.einsum('fwhd,fwhd->whd', (along_y, along_z))

    histxy = sim_xy.float().histc(bins=100)
    histxz = sim_xz.float().histc(bins=100)
    histyz = sim_yz.float().histc(bins=100)

    fig, ax = plt.subplots(3,1, dpi=200)
    ax[0].bar(torch.linspace(0, 1, 100), histxy.cpu(), align='center')
    ax[1].bar(torch.linspace(0, 1, 100), histxz.cpu(), align='center')
    ax[2].bar(torch.linspace(0, 1, 100), histyz.cpu(), align='center')
    fig.savefig('sim_histograms.png')
