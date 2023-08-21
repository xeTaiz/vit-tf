import torch

def crop_pad(sim, thresh=0.1, pad=0):
    ''' Crop `sim` to the region where `sim > thresh` and pad by `pad` pixels on each side. If `sim` is a list the first element is used to determine the crop region.
        Args:
            sim(list or torch.Tensor): similarity map (W, H, D) or List of such tensors
            thresh(float): threshold for cropping
            pad(int): padding size (Can be tuple like for `torch.nn.functional.pad`)

        Returns:
            torch.Tensor: cropped and padded similarity map
    '''
    if isinstance(sim, list):
        others = sim
        sim = others[0]
    else:
        others = [sim]
    nz = torch.nonzero(sim > thresh)
    print(sim > thresh)
    print(nz)
    mi = torch.clamp(nz.min(dim=0).values - pad, 0, None)
    ma = nz.max(dim=0).values + pad + 1
    if len(others) > 1:
        return [s[...,mi[0]:ma[0], mi[1]:ma[1]] for s in others], (mi, ma)
    else:
        return sim[mi[0]:ma[0], mi[1]:ma[1]], (mi, ma)

def write_crop_into(uncropped, crop, mima):
    mi, ma = mima
    if not isinstance(uncropped, list):
        uncropped = [uncropped]

    for u,c in zip(uncropped, crop):
        u[..., mi[0]:ma[0], mi[1]:ma[1]] = c

    return uncropped[0] if len(uncropped) == 1 else uncropped


if __name__ == '__main__':
    mg = torch.meshgrid(torch.linspace(-1,1,9), torch.linspace(-1,1,9))
    gauss = torch.exp(-((mg[0]**2 + mg[1]**2)/0.5))
    print(gauss)
    thresh_gauss = torch.where(gauss > 0.3, gauss, 0)
    print(thresh_gauss)
    # cropped_gauss, mima= crop_pad(thresh_gauss, thresh=0.3, pad=0)
    # print(cropped_gauss.shape)
    # cropped_gauss, mima = crop_pad(thresh_gauss, thresh=0.5, pad=0)
    # print(cropped_gauss.shape)
    cropped_gauss, mima = crop_pad([thresh_gauss, thresh_gauss], thresh=0.6, pad=0)
    print(mima)
    g1, g2 = cropped_gauss
    print('g1', g1)
    ones = torch.ones_like(thresh_gauss)
    res = write_crop_into(ones, cropped_gauss, mima)
    print(res)
