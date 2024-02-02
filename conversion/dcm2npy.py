import pydicom as dcm
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import os
import nibabel as nb

if __name__ == '__main__':
    # open
    path = Path('/home/dome/Data/Volumes/VC_WOLF_LYMPHOM/1000A05D/1000A05E/')
    dirs = os.listdir(path)
    for dir in dirs:
        files = os.listdir(path/dir)
        arrays = []
        for fn in sorted(files):
            ds = dcm.dcmread(path/dir/fn)
            if hasattr(ds, 'pixel_array'):
                arrays.append(ds.pixel_array)
            else:
                print(fn, 'has no pixel_array')
        vol = np.stack(arrays, axis=-1)
        print('Found volume:', vol.shape, vol.dtype, vol.min(), vol.max())
        np.save(f'/home/dome/Data/Volumes/VC_WOLF_LYMPHOM/1000A05D.npy', vol)
        nib_im = nb.Nifti1Image(vol, np.eye(4))
        nb.save(nib_im, '/home/dome/Data/Volumes/VC_WOLF_LYMPHOM/1000A05D.nii.gz')
