import numpy as np
from collections import defaultdict

### Brush Export ###


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position
    """
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data
    """
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def decode_rle(rle, print_params: bool = False):
    """ from LS RLE to numpy uint8 3d image [width, height, channel]
    
    Args:
        print_params (bool, optional): If true, a RLE parameters print statement is suppressed
    """
    input = InputStream(bytes2bit(rle))
    num = input.read(32)
    word_size = input.read(5) + 1
    rle_sizes = [input.read(4) + 1 for _ in range(4)]
    
    if print_params:
        print('RLE params:', num, 'values', word_size, 'word_size', rle_sizes, 'rle_sizes')
        
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = input.read(1)
        j = i + 1 + input.read(rle_sizes[input.read(2)])
        if x:
            val = input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = input.read(word_size)
                out[i] = val
                i += 1
    return out


def decode_from_annotation(from_name, results):
    """ from LS annotation to {"tag_name + label_name": [numpy uint8 image (width x height)]}
    """
    layers = {}
    counters = defaultdict(int)
    for result in results:
        key = 'brushlabels' if result['type'].lower() == 'brushlabels' else \
            ('labels' if result['type'].lower() == 'labels' else None)
        if key is None or 'rle' not in result['value']:
            continue

        rle = result['value']['rle']
        width = result['original_width']
        height = result['original_height']
        labels = result['value'][key] if key in result['value'] else ['no_label']
        name = from_name + '-' + '-'.join(labels)

        # result count
        i = str(counters[name])
        counters[name] += 1
        name += '-' + i

        image = decode_rle(rle)
        layers[name] = np.reshape(image, [height, width, 4])[:, :, 3]
    return layers
