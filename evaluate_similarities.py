import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, jaccard_score, confusion_matrix, accuracy_score
import json

from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

from icecream import ic, argumentToString
@argumentToString.register(np.ndarray)
def _(t):
    contig = f'{"C" if t.flags["C_CONTIGUOUS"] else ""} {"F" if t.flags["F_CONTIGUOUS"] else ""}'.strip()
    if len(contig) == 0: contig = "NOT"
    return f'{tuple(t.shape)} of type {t.dtype} (NumPy) in value range [{t.min().item():.3f}, {t.max().item():.3f}] ({contig} contiguous)'

@argumentToString.register(torch.Tensor)
def _(t):
    if t.is_contiguous():
        contig = '(C contiguous)'
    else:
        contig = '(NOT contiguous!!)'
    return f'{tuple(t.shape)} of type {t.dtype} ({t.device.type}) in value range [{t.min().item():.3f}, {t.max().item():.3f}] {contig}'
ic.configureOutput(prefix='')

label2idx = {
    'background': 0,
    'liver': 1,
    'bladder': 2,
    'lung': 3,
    'kidney': 4,
    'bone': 5
}
idx2label = ['liver', 'bladder', 'lung', 'kidney', 'bone']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=Path, help='Path to features, annotations, volume etc.')
    parser.add_argument('--label', type=Path, default='userstudy/labels-10.npy', help='Path to label volume')
    parser.add_argument('--labels', type=str, nargs='+', default=['lung', 'liver', 'kidney'], help='Labels found in predictions (in order)')
    args = parser.parse_args()

    # Load data
    dir = Path(args.data)
    label_fn = Path(args.label)
    label_names = args.labels
    assert (dir/'predictions.npy').exists()
    assert (label_fn).exists()
    assert (dir/'metadata.json').exists()

    with (dir/'metadata.json').open('r', encoding='UTF-8') as f:
        metadata = json.load(f)
    labels_orig = torch.as_tensor(np.load(label_fn, allow_pickle=True)[()])
    preds = {k: torch.as_tensor(v) for k,v in np.load(dir / 'predictions.npy', allow_pickle=True)[()].items()}
    ic(labels_orig)

    results = {}
    keys = sorted(preds.keys())
    for ln, k in zip(label_names, keys):
        p = preds[k]
        meta = metadata[k]
        labels = F.interpolate((labels_orig == label2idx[ln]).to(torch.uint8)[None, None], p.shape[-3:], mode='nearest').reshape(-1)
        pred  = p.reshape(-1)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, pred, average=None)
        cm = confusion_matrix(labels, pred)
        acc = accuracy_score(labels, pred)
        iou = jaccard_score(labels, pred, average=None)
        metrics = {
            'accuracy': acc,
            'precision': prec.tolist(),
            'recall': rec.tolist(),
            'f1': f1.tolist(),
            'iou': iou.tolist(),
            'confusion_matrix': cm.tolist(),
            'annotation_time': meta['time'],
            'num_annotations': meta['num_annotations']
        }
        results[ln] = metrics

    pprint(results)
    with open(dir / 'metrics.json', 'w') as f:
        json.dump(results, f)
        # for i in range(0, labels.max()+1):
    #    nam = label_names[i]
    #    pred_bin = pred == i
    #    labl_bin = labels.reshape(-1) == i
    #    bin_iou = jaccard_score(labl_bin, pred_bin)
    #    print(f'Binary {nam} IOU:', bin_iou)
