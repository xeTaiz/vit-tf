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
        slices = []
        try:
            print(f'Converting {dir} to {dir}.npy')
            for f in sorted(dir.rglob('*.tif')):
                with tifffile.TiffFile(dir/f) as tif:
                    im = tif.pages[0].asarray()
                    properties = {}
                    for tag in tif.pages[0].tags.values():
                        name, value = tag.name, tag.value
                        properties[name] = value
                    slices.append(im)
            if len(slices) == 0:
                print(f'No tiffs found in {dir}')
                continue
        except Exception as e:
            # print(f'Error converting {dir}: {e}')
            continue
        array = np.stack(slices, axis=-1)
        print(array.shape)
        print('Writing to disk...')
        print(properties)
        np.save(f'{dir}/../{dir.name}.npy', array)
