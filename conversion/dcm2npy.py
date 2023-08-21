import pydicom as dcm
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import os

if __name__ == '__main__':
    # open
    path = Path('/home/dome/Downloads/Vattenfantom/DICOM/22110307')
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
        np.save(f'/run/media/dome/SSD/Data/Volumes/VattenfantomNpy/vattenfantom_{dir}.npy', vol)
