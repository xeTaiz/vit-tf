import torch
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg

import icecream as ic
from domesutils import *
from infer import make_5d

__all__ = ['apply_bilateral_solver3d']


RGB_TO_YUV = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]])
YUV_TO_RGB = np.array([[1.0, 0.0, 1.402], [1.0, -0.34414, -0.71414],
                       [1.0, 1.772, 0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)
MAX_VAL = 255.0


def rgb2yuv(im):
    return np.tensordot(im, RGB_TO_YUV, ([3], [1])) + YUV_OFFSET


def yuv2rgb(im):
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([3], [1]))

def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    locs = np.searchsorted(valid, candidates)
    # Handle edge case where the candidate is larger than all valid values
    locs = np.clip(locs, 0, len(valid) - 1)
    # Identify which values are actually present
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx]
    return valid_idx, locs


class BilateralGrid(object):

    def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
        im_yuv = rgb2yuv(im)
        # Compute 5-dimensional XYLUV bilateral-space coordinates
        Iz, Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1], :im.shape[2]]
        x_coords = (Ix / sigma_spatial).astype(int)[...,None]
        y_coords = (Iy / sigma_spatial).astype(int)[...,None]
        z_coords = (Iz / sigma_spatial).astype(int)[...,None]
        luma_coords = (im_yuv[..., [0]] / sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        coords = np.concatenate((x_coords, y_coords, z_coords, luma_coords, chroma_coords), axis=-1)
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        # Hacky "hash vector" for coordinates,
        # Requires all scaled coordinates be < MAX_VAL
        self.hash_vec = (MAX_VAL**np.arange(self.dim))
        # Construct S and B matrix
        self._compute_factorization(coords_flat)

    def _compute_factorization(self, coords_flat):
        # Hash each coordinate in grid to a unique value
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = \
            np.unique(hashed_coords, return_index=True, return_inverse=True)
        # Identify unique set of vertices
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        # Construct sparse splat matrix that maps from pixels to vertices
        self.S = csr_matrix(
            (np.ones(self.npixels), (idx, np.arange(self.npixels))))
        # Construct sparse blur matrices.
        # Note that these represent [1 0 1] blurs, excluding the central element
        self.blurs = []
        for d in range(self.dim):
            blur = 0.0
            for offset in (-1, 1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix(
                    (np.ones((len(valid_coord), )), (valid_coord, idx)),
                    shape=(self.nvertices, self.nvertices))
            self.blurs.append(blur)

    def _hash_coords(self, coord):
        """Hacky function to turn a coordinate into a unique value"""
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x):
        return self.S.dot(x)

    def slice(self, y):
        return self.S.T.dot(y)

    def blur(self, x):
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert x.shape[0] == self.nvertices
        out = 2 * self.dim * x
        for blur in self.blurs:
            out = out + blur.dot(x)
        return out

    def filter(self, x):
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) /  \
               self.slice(self.blur(self.splat(np.ones_like(x))))


def bistochastize(grid, maxiter=10):
    """Compute diagonal matrices to bistochastize a bilateral grid"""
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for i in range(maxiter):
        n = np.sqrt(n * m / grid.blur(n))
    # Correct m to satisfy the assumption of bistochastization regardless
    # of how many iterations have been run.
    m = n * grid.blur(n)
    Dm = diags(m, 0)
    Dn = diags(n, 0)
    return Dn, Dm


class BilateralSolver(object):

    def __init__(self, grid, params):
        self.grid = grid
        self.params = params
        self.Dn, self.Dm = bistochastize(grid)

    def solve(self, x, w):
        # Check that w is a vector or a nx1 matrix
        if w.ndim == 2:
            assert (w.shape[1] == 1)
        elif w.ndim == 1:
            w = w.reshape(w.shape[0], 1)
        A_smooth = (self.Dm - self.Dn.dot(self.grid.blur(self.Dn)))
        w_splat = self.grid.splat(w)
        A_data = diags(w_splat[:, 0], 0)
        A = self.params["lam"] * A_smooth + A_data
        xw = x * w
        b = self.grid.splat(xw)
        # Use simple Jacobi preconditioner
        A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
        M = diags(1 / A_diag, 0)
        # Flat initialization
        y0 = self.grid.splat(xw) / w_splat
        yhat = np.empty_like(y0)
        for d in range(x.shape[-1]):
            yhat[..., d], info = cg(A,
                                    b[..., d],
                                    x0=y0[..., d],
                                    M=M,
                                    maxiter=self.params["cg_maxiter"],
                                    tol=self.params["cg_tol"])
        xhat = self.grid.slice(yhat)
        return xhat

grid_params_default = {
    'sigma_luma' : 4, # Brightness bandwidth
    'sigma_chroma': 4, # Color bandwidth
    'sigma_spatial': 24 # Spatial bandwidth
}

bs_params_default = {
    'lam': 256, # The strength of the smoothness parameter
    'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
    'cg_tol': 1e-5, # The tolerance on the convergence in PCG
    'cg_maxiter': 25 # The number of PCG iterations
}

def filter_gauss_separated(input):
    win = torch.tensor([0.25, 0.5, 0.25])[None, None, None, None].to(input.dtype)
    out = F.conv3d(input, win, groups=input.size(1), padding=(0,0,1))
    out = F.conv3d(out, win.transpose(3, 4), groups=input.size(1), padding=(0,1,0))
    out = F.conv3d(out, win.transpose(2, 4), groups=input.size(1), padding=(1,0,0))
    return out

def filter_sobel_separated(input):
    win = torch.tensor([-0.5, 0, 0.5])[None, None, None, None].to(input.dtype)
    out = F.conv3d(input, win, groups=input.size(1), padding=(0,0,1))**2
    out += F.conv3d(input, win.transpose(3, 4), groups=input.size(1), padding=(0,1,0))**2
    out += F.conv3d(input, win.transpose(2, 4), groups=input.size(1), padding=(1,0,0))**2
    return out.sqrt()

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
    mi = torch.clamp(nz.min(dim=0).values[-3:] - pad,     0, None)
    ma = torch.clamp(nz.max(dim=0).values[-3:] + pad + 1, None, torch.tensor(sim.shape[-3:]))
    if len(others) > 1:
        return [s[...,mi[0]:ma[0], mi[1]:ma[1], mi[2]:ma[2]] for s in others], (mi, ma)
    else:
        return sim[...,mi[0]:ma[0], mi[1]:ma[1], mi[2]:ma[2]], (mi, ma)

def write_crop_into(uncropped, crop, mima):
    mi, ma = mima
    uncropped[..., mi[0]:ma[0], mi[1]:ma[1], mi[2]:ma[2]] = crop
    return uncropped

def apply_bilateral_solver3d(t: torch.Tensor, r: torch.Tensor, c: torch.Tensor = None, grid_params={}, bs_params={}):
    ''' Applies bilateral solver on target `t` using confidence `c` and reference `r`.

    Args:
        t (torch.Tensor): Target to filter (1, W, H, D) as float with value range [0,1]
        r (torch.Tensor): Reference image  (3, W, H, D) as uint8 with value range [0,255]
        c (torch.Tensor): Confidence for target (Defaults to target image `t`) (1, W, H, D) as float with value range [0, 1]
        grid_params (dict, optional): Grid parameters for bilateral solver. May include `sigma_luma`, `sigma_chroma` and `sigma_spatial`.
        bs_params (dict, optional): Bilateral solver parameters. May inlcude `lam`, `A_diag_min`, `cg_tol` and `cg_maxiter`.

    Returns:
        torch.Tensor: Bilaterally solved target (1, W, H, D) as torch.float32
    '''
    gp = {**grid_params_default, **grid_params}
    bs = {**bs_params_default, **bs_params}
    shap = t.shape[-3:]
    t = t.cpu().permute(1,2,3,0).numpy().squeeze(-1).reshape(-1, 1).astype(np.double)
    if c is None:
        # c = np.ones(shap).reshape(-1,1) * 0.999
        # print('np.ones confidence', c.shape, c.dtype, c.min(), c.max())

        # print('reference in', r.shape, r.min(), r.max())
        c = filter_sobel_separated(make_5d(r[[0]]).float() / 255.0)
        c = c.squeeze(0) # switch with below line to enable blurring
        # c = filter_gauss_separated(c).squeeze(0)
        # print('confidence', c.shape, c.dtype, c.min(), c.max())
        c = (c.max() - c).numpy().astype(np.double).reshape(-1, 1)
        # print('confidence', c.shape, c.dtype, c.min(), c.max())
    else:
        c = c.cpu().permute(1,2,3,0).numpy().astype(np.double).reshape(-1,1)
    r = r.cpu().permute(1,2,3,0).numpy()
    grid = BilateralGrid(r, **gp)
    solver = BilateralSolver(grid, bs)
    out = solver.solve(t, c).reshape(*shap)
    return torch.nan_to_num(torch.from_numpy(out).to(torch.float32).squeeze())



if __name__ == '__main__':
    t = np.random.rand(512,512) # To filter
    c = np.random.rand(512,512) # Confidence
    r = np.random.rand(512,512,3) * 255.0# Reference
    im_shape = r.shape[:2]
    grid = BilateralGrid(r, **grid_params_default)
    tc_filt = grid.filter(t * c)
    c_filt = grid.filter(c)
    output_filter = (tc_filt / c_filt).reshape(im_shape)
    output = BilateralSolver(grid, bs_params_default).solve(t, c).reshape(im_shape)
