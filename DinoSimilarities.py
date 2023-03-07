# Name: DinoSimilarities

import inviwopy as ivw
import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
NTF_REPO = '/home/dome/Dev/ntf'
sys.path.append(NTF_REPO)
from infer import sample_features3d, resample_topk, make_4d, make_5d, norm_minmax
# from bilateral_solver3d import apply_bilateral_solver3d
import os
import subprocess
from contextlib import contextmanager

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg

######### Bilateral Solver

RGB_TO_YUV = np.array([[0.299, 0.587, 0.114], 
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]])
YUV_TO_RGB = np.array([[1.0, 0.0, 1.402], 
                       [1.0, -0.34414, -0.71414],
                       [1.0, 1.772, 0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, 1, -1)
MAX_VAL = 255.0


def rgb2yuv(im):
    return np.tensordot(im, RGB_TO_YUV, ([3], [1])) + YUV_OFFSET

def yuv2rgb(im): 
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([3], [1]))

def log(name, t):
    # if isinstance(t, (np.ndarray, np.matrix)):
    if isinstance(t, np.ndarray):
        contig = f'C contiguous: {t.flags["C_CONTIGUOUS"]}     F contiguous: {t.flags["F_CONTIGUOUS"]}'
    elif torch.is_tensor(t):
        contig = f'C contiguous: {t.is_contiguous()}'
    else:
        contig = ''
    print(f'{name}: {tuple(t.shape)} ({t.dtype}) in range ({t.min():.2f}, {t.max():.2f}) {contig}')

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
        # Compute 6-dimensional XYZLUV bilateral-space coordinates
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
        # log('A_smooth', A_smooth)
        w_splat = self.grid.splat(w)
        A_data = diags(w_splat[:, 0], 0)
        # log('A_data', A_data)
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
    tmp = t
    gp = {**grid_params_default, **grid_params}
    bs = {**bs_params_default, **bs_params}
    shap = t.shape[-3:]
    t = t.cpu().permute(1,2,3,0).numpy().squeeze(-1).reshape(-1, 1).astype(np.double)
    r = r.cpu().permute(1,2,3,0).numpy()
    if c is None:
        c = t
        c = np.ones(shap).reshape(-1,1) * 0.999
    else:
        c = c.cpu().permute(1,2,3,0).numpy().astype(np.double).reshape(-1,1)

    grid = BilateralGrid(r, **gp)
    solver = BilateralSolver(grid, bs)
    out = solver.solve(t, c).reshape(*shap)
    return torch.from_numpy(out).to(torch.float32).squeeze()


######### Inviwo Processor


def get_processor(id):
    net = ivw.getApp().network
    procs = [p for p in net.processors if p.identifier == id]
    if len(procs) == 0: raise Exception(f'Processor not found in network: {id}')
    elif len(procs) > 1: 
        print('Something weird happened. Multiple processors have that id:')
        print(procs)
        print('Using first one')
        return procs[0]
    else: return procs[0]

def get_network(): return ivw.getApp().network

@contextmanager
def lockedNetwork(net=None):
    if net is None: net = get_network()
    net.lock()
    try:
        yield
    finally:
        net.unlock()

def get_annotations(proc_name, typ=torch.float32, dev=torch.device('cpu')):
    dino_proc = get_processor(proc_name)
    annotations = { 
        ntf.identifier: np.array([a.value.array for a in ntf.properties['annotations'].properties])
        for ntf in dino_proc.tfs.properties 
    }
    return {k: torch.from_numpy(v.astype(np.int64)).to(typ).to(dev) for k,v in annotations.items()}#{k: torch.from_numpy(v.astype(np.int64)) for k,v in annotations.items() if v.shape[0] > 0}

def get_similarity_params(proc_name):
    dino_proc = get_processor(proc_name)
    return { 
        ntf.identifier: {
            'exponent': ntf.properties['simexponent'].value,
            'threshold': ntf.properties['simthresh'].value,
            'normalize': ntf.properties['normBeforeBilat'].value
        } for ntf in dino_proc.tfs.properties 
    }

def is_path_creatable(pathname: str) -> bool:
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)

class DinoSimilarities(ivw.Processor):
    def __init__(self, id, name):
        ivw.Processor.__init__(self, id, name)
        # Ports
        self.inport = ivw.data.VolumeInport("inport")
        self.addInport(self.inport, owner=False)
        # Properties    TODO: set cleanupTemporaryVolume default to True
        self.dinoProcessorIdentifier = ivw.properties.StringProperty("dinoProcId", "DINO Volume Renderer Identifier", "DINOVolumeRenderer")
        self.useCuda = ivw.properties.BoolProperty("useCuda", "Use CUDA", False)
        self.cleanupTemporaryVolume = ivw.properties.BoolProperty("cleanupTempVol", "Clean up volume that's temporarily created on disk to pass to infer.py", False)
        self.similarityVolumeScalingFactor = ivw.properties.FloatProperty("simScaleFact", "Similarity Volume Downscale Factor", 4.0, 1.0, 8.0)
        self.runWithScaling = ivw.properties.ButtonProperty("runWithScaling", "Run with scaling", )
        self.updatePorts = ivw.properties.ButtonProperty("updateEverything", "Update Callbacks, Ports & Connections", self.updateCallbacksPortsAndConnections)
        self.sliceAlong = ivw.properties.OptionPropertyString("sliceAlong", "DINO Slice along Axis", [
            ivw.properties.StringOption("alongALL", 'Slice along ALL', 'all'),
            ivw.properties.StringOption("alongX", 'Slice along X', 'x'),
            ivw.properties.StringOption("alongY", 'Slice along Y', 'y'),
            ivw.properties.StringOption("alongZ", 'Slice along Z', 'z')
        ])
        self.sigmaSpatial = ivw.properties.IntProperty("blSigmaSpatial", "BL: Sigma Spatial", 3, 1, 32)
        self.sigmaChroma = ivw.properties.IntProperty("blSigmaChroma", "BL: SigmaChroma", 3, 1, 16)
        self.sigmaLuma = ivw.properties.IntProperty("blSigmaLuma", "BL: SigmaLumal", 3, 1, 16)
        self.addProperty(self.dinoProcessorIdentifier)
        self.addProperty(self.useCuda)
        self.addProperty(self.cleanupTemporaryVolume)
        self.addProperty(self.similarityVolumeScalingFactor)
        self.addProperty(self.updatePorts)
        self.addProperty(self.sliceAlong)
        self.addProperty(self.sigmaSpatial)
        self.addProperty(self.sigmaChroma)
        self.addProperty(self.sigmaLuma)
        # Callbacks
        self.inport.onChange(self.getVolumeDataPath)
        self.inport.onChange(self.loadVolume)
        self.useCuda.onChange(self.updateFeatvolDevice)
        self.dinoProcessorIdentifier.onChange(self.updateCallbacksPortsAndConnections)
        # Init other variables
        self.outs = {}
        self.registeredCallbacks = {}
        self.feat_vol = None
        self.vol = None
        self.dims = None
        self.cache_path = None

    @staticmethod
    def processorInfo():
        return ivw.ProcessorInfo(
            classIdentifier="org.inviwo.DinoSimilarities",
            displayName="DinoSimilarities", 
            category="Python",
            codeState=ivw.CodeState.Stable,
            tags=ivw.Tags.PY
        )

    def getProcessorInfo(self):
        return DinoSimilarities.processorInfo()

    def getVolumeDataPath(self):
        print('getVolumeDataPath()')
        if self.inport.isConnected():
            data_path = Path(self.inport.getConnectedOutport().processor.properties['filename'].value)
            print(f'New volume data connected: {data_path}')
            clean_name = data_path.stem.replace(" ", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
            self.cache_path = data_path.parent/f'{clean_name}_DINOfeats_{self.sliceAlong.selectedValue}.npy'
            self.initializeResources()
        else:
            self.cache_path = None

    def updateCallbacksPortsAndConnections(self):
        self.registerCallbacks()
        self.addAndConnectOutports()

    def registerCallbacks(self):
        print('registerCallbacks()')
        proc = get_processor(self.dinoProcessorIdentifier.value)
        # Register this function as callback on annotations list property
        if proc.annotationButtons.path not in self.registeredCallbacks.keys():
            cb = proc.annotationButtons.onChange(self.updateCallbacksPortsAndConnections)
            self.registeredCallbacks[proc.annotationButtons.path] = cb 
        # Register Callbacks for "Add to Class" buttons
        for btnProp in proc.annotationButtons.properties:
            if btnProp.path not in self.registeredCallbacks:
                cb = btnProp.onChange(self.invalidateOutput)
                self.registeredCallbacks[btnProp.path] = cb
        # Register Callbacks for transfer functions
        for ntfProp in proc.tfs.properties:
            for tfProp in ntfProp.properties:
                if tfProp.identifier in ['simexponent', 'simthresh', 'normBeforeBilat'] \
                and tfProp.path not in self.registeredCallbacks:
                    cb = tfProp.onChange(self.invalidateOutput)
                    self.registeredCallbacks[tfProp.path] = cb

    def invalidateOutput(self): 
        print('Invalidating Output!')
        self.invalidate(ivw.properties.InvalidationLevel.InvalidOutput)

    def addAndConnectOutports(self):
        self.addVolumeOutports()
        self.connectVolumeOutports()
        self.invalidateOutput()

    def addVolumeOutports(self):
        print('addVolumeOutports()')
        new_names = set(get_similarity_params(self.dinoProcessorIdentifier.value).keys())
        cur_names = set(self.outs.keys())
        for k in (cur_names - new_names):
            self.removeOutport(self.outs[k])
            del self.outs[k]
        for k in sorted(new_names - cur_names):
            self.outs[k] = ivw.data.VolumeOutport(k)
            self.addOutport(self.outs[k])

    def connectVolumeOutports(self):
        print('connectVolumeOutports()')
        proc = get_processor(self.dinoProcessorIdentifier.value)
        simInport = proc.getInport('similarity')
        net = get_network()
        with lockedNetwork(net):
            for k,v in self.outs.items():
                print(f'Connection between {simInport.identifier} and {v.identifier}: {simInport.isConnectedTo(v)}')
                if not simInport.isConnectedTo(v):
                    # simInport.connectTo(v) # does not fully connect the ports (visual link is missing, some things go wrong)
                    net.addConnection(v, simInport)
                    print(f'Connecting {simInport.identifier} to {v.identifier}.')

    def updateFeatvolDevice(self):
        print('updateFeatvolDevice()')
        dev = torch.device("cuda" if torch.cuda.is_available and self.useCuda.value else "cpu")
        typ = torch.float16 if dev == torch.device("cuda") else torch.float32
        self.feat_vol = self.feat_vol.to(dev).to(typ)

    def loadCache(self, cache_path, attention_features='k'):
        print('loadCache()')
        if cache_path.suffix in ['.pt', '.pth']:
            data = torch.load(cache_path)
            if type(data) == dict:
                feat_vol = data[attention_features]
            else:
                feat_vol = data
        elif cache_path.suffix == '.npy':
            data = np.load(cache_path, allow_pickle=True)
            if data.dtype == "O":
                feat_vol = torch.from_numpy(data[()][attention_features])
            else:
                feat_vol = torch.from_numpy(data)
        else:
            raise Exception(f'Unsupported file extension: {cache_path.suffix}')
        assert feat_vol.ndim == 4
        dev = torch.device("cuda" if torch.cuda.is_available and self.useCuda.value else "cpu")
        typ = torch.float16 if dev == torch.device("cuda") else torch.float32
        self.feat_vol = F.normalize(feat_vol.to(typ).to(dev), dim=0)
        log('Loaded self.feat_vol', self.feat_vol)

    def loadVolume(self, use_featvol_dims=True):
        print('loadVolume()')
        vol = np.ascontiguousarray(self.inport.getData().data).astype(np.float32)
        self.dims = tuple(vol.shape[:3])
        log('self.feat_vol', self.feat_vol)
        # if use_featvol_dims and self.feat_vol is not None:
        #     sim_shape = tuple(self.feat_vol.shape[-3:])
        # else:
        sim_shape = tuple(self.feat_vol.shape[-3:])
        other_vol = F.interpolate(make_5d(norm_minmax(torch.from_numpy(vol))), sim_shape, mode='nearest').squeeze(0)
        log('other_vol', other_vol)
        sim_shape = tuple((torch.tensor([*self.dims]) // self.similarityVolumeScalingFactor.value).long().tolist())
        vol_resized = F.interpolate(make_5d(norm_minmax(torch.from_numpy(vol))), sim_shape, mode='nearest').squeeze(0)
        log('vol_resized', vol_resized)
        self.vol = (255.0 * vol_resized).to(torch.uint8)
        log('Updated self.vol', self.vol)

    def computeSimilarities(self):
        print('computeSimilarities()')
        with torch.no_grad():
            dev, typ = self.feat_vol.device, self.feat_vol.dtype
            simparams = get_similarity_params(self.dinoProcessorIdentifier.value)
            annotations = get_annotations(self.dinoProcessorIdentifier.value, dev=dev, typ=typ)
            def split_into_classes(t):
                sims = {}
                idx = 0
                for k,v in annotations.items():
                    sims[k] = t[idx:idx+v.size(0)]
                    idx += v.size(0)
                return sims
            if len(annotations) == 0: return
            vol_extent = torch.tensor([[[*self.dims]]], device=dev, dtype=typ)
            abs_coords = torch.cat(list(annotations.values()))
            rel_coords = (abs_coords.float() + 0.5) / vol_extent * 2.0 - 1.0

            qf = sample_features3d(self.feat_vol, rel_coords, mode='nearest')
            sims = torch.einsum('fwhd,caf->cawhd', (self.feat_vol, qf))
            sims = resample_topk(self.feat_vol, sims, K=8).squeeze(0)

            bls_params = {
                'sigma_spatial': self.sigmaSpatial.value,
                'sigma_chroma': self.sigmaChroma.value,
                'sigma_luma': self.sigmaLuma.value
            }
            sim_split = {}
            for k,v in split_into_classes(sims).items():
                print(k, v.shape) # ntf3 torch.Size([11, 64, 64, 62])
                sim = torch.where(v >= simparams[k]['threshold'], v, torch.zeros(1, dtype=typ, device=dev)) # Throw away low similarities
                sim = torch.max(sim ** simparams[k]['exponent'], dim=0).values.clamp(0,1) # Exponentiate and accumulate sims per annotation
                # Apply Bilateral Solver
                if simparams[k]['normalize']:
                    sim /= sim.max()
                if sim.shape[-3:] != self.vol.shape[-3:]:
                    print('Resizing similarity to', tuple(self.vol.shape[-3:]))
                    sim = F.interpolate(make_5d(sim), self.vol.shape[-3:], mode='nearest').squeeze(0)
                blsim = apply_bilateral_solver3d(sim, self.vol.expand(3,-1,-1,-1), grid_params=bls_params)
                sim_split[k] = (255.0 / blsim.max() * blsim).cpu().to(torch.uint8)
            return sim_split

    def initializeResources(self):
        print('initializeResources()')
        # Takes care of computing, caching and loading self.feat_vol
        if self.cache_path is None: return # No cache path means no feat_vol to load
        if self.cache_path.exists(): # Load feat_vol if it exists
            print(f'Loading {self.cache_path}')
            self.loadCache(self.cache_path) # Loads self.feat_vol
            self.addAndConnectOutports() # Add volume outports for similarities if necessary and connect to DINOVolumeRenderer
            self.registerCallbacks() # Registers callbacks on DINOVolumeRenderer's NTF properties
        elif self.inport.hasData(): # Compute and cache feat_vol
            self.loadVolume()  # Loads self.vol and sets self.dims
            if is_path_creatable(str(self.cache_path)): # only if we can save to cache_path
                print(f'Computing features and saving cache to {self.cache_path}')
                # Save incoming volume temporarily as .npy
                vol_np = self.inport.getData().data
                tmpvol_path = str(self.cache_path.parent/'tmpvol.npy')
                np.save(tmpvol_path, np.ascontiguousarray(vol_np))
                # Run infer.py script to produce feat_vol cache
                cmd = f'{sys.executable} {NTF_REPO}/infer.py --data-path "{tmpvol_path}" --cache-path "{self.cache_path}" --slice-along {self.sliceAlong.selectedValue}'
                print(f'Running command: {cmd}')
                comp = subprocess.run(cmd, encoding='UTF-8', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if comp.returncode == 0: # infer.py was successful, load cache
                    self.loadCache(self.cache_path) # Load self.feat_vol
                    self.addAndConnectOutports()    # Update volume outports and connect automatically
                else: # infer.py failed, log error
                    print(f'Something went wrong with computing the features (Return Code {comp.returncode}):')
                    print(comp.stdout)
                if self.cleanupTemporaryVolume.value: # Remove temporary .npy file
                    os.remove(tmpvol_path)
            else:
                print(f'Use valid Cache Path. ("{self.cache_path}" is invalid or cannot be written to)')

    def process(self):
        print('process()')
        if self.feat_vol is None:
            self.initializeResources() # If there's no feat_vol, try to load it
        else:
            similarities = self.computeSimilarities() # Compute similarity maps
            # print({k:v.shape for k,v in similarities.items()})
            if similarities is None or len(similarities) == 0:
                print('Could not compute similarities. Did you annotate anything?')
            else: # Output similarity volumes through volume outports
                in_vol = self.inport.getData() # ivw.Volume
                for k,v in similarities.items():
                    volume = ivw.data.Volume(np.asfortranarray(v.numpy()))
                    volume.modelMatrix = in_vol.modelMatrix
                    volume.worldMatrix = in_vol.worldMatrix
                    volume.dataMap.dataRange = ivw.glm.dvec2(0.0, 255.0)
                    volume.dataMap.valueRange= ivw.glm.dvec2(0.0, 255.0)
                    self.outs[k].setData(volume)
