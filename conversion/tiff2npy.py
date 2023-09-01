import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import tifffile

if __name__ == '__main__':
    parser = ArgumentParser("Tiff to Numpy")
    parser.add_argument('--data', type=str, help='Path to data')
    args = parser.parse_args()

    p= Path(args.data)
    for dir in p.iterdir():
        if not dir.is_dir():
            continue
        slices = []
        try:
            print(f'Converting {dir} to {dir}.npy')
            for f in sorted(dir.rglob('*.tif')):
                im = tifffile.imread(dir/f)
                slices.append(im)
            if len(slices) == 0:
                print(f'No tiffs found in {dir}')
                continue
        except Exception as e:
            print(f'Error converting {dir}: {e}')
            continue
        array = np.stack(slices, axis=-1)
        print(array.shape)
        print('Writing to disk...')
        np.save(f'{dir}/../{dir.name}.npy', array)
