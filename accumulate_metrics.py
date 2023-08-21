import numpy as np
import json
from pathlib import Path
from argparse import ArgumentParser
from pprint import pprint
from collections import defaultdict
from functools import partial

if __name__ == '__main__':
    parser = ArgumentParser("Accumulate metrics")
    parser.add_argument('--data', type=str, required=True, help='Directory containing all data')
    parser.add_argument('--glob-pattern', type=str, default='metrics*.json', help='Glob pattern to match')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    args = parser.parse_args()

    p = Path(args.data)
    assert p.exists() and p.is_dir()
    assert args.glob_pattern.endswith('.json')
    if args.output is None:
        args.output = args.glob_pattern.replace('*', '')
        if 'metrics' not in args.output:
            args.output = f'metrics_{args.output}'
    files = list(p.rglob(args.glob_pattern))

    print(f'Found {len(files)} files matching {args.glob_pattern}')
    if len(files) == 0:
        print('No files found, exiting.')
        exit(1)

    metric_files = {}
    metrics = {}
    for f in files:
        with open(p/f) as fp:
            metric_files[f.name] = json.load(fp)

    metric_names = list(metric_files[files[0].name].keys())
    metric_names.remove('confusion_matrix')
    for file in metric_files.values():
        for m in metric_names:
            if isinstance(file[m], dict):
                if m not in metrics:
                    metrics[m] = defaultdict(list)
                for c in file[m].keys():
                    metrics[m][c].append(file[m][c])
            elif isinstance(file[m], list):
                if m not in metrics:
                    metrics[m] = []
                metrics[m].append(file[m])
            elif isinstance(file[m], float):
                if m not in metrics:
                    metrics[m] = []
                metrics[m].append(file[m])
            else:
                print(f'Unknown metric type: {type(file[m])}')

    for k in metrics.keys():
        metric = metrics[k]
        if isinstance(metrics[k], dict):
            metrics[k] = {cn: np.mean(metric[cn], axis=0).item() for cn in metric.keys()}
        elif isinstance(metric, list):
            metrics[k] = np.mean(metric, axis=0).item()

    pprint(metrics)
    metrics['files'] = [str(p/f) for f in files]

    with open(args.output, 'w') as fp:
        json.dump(metrics, fp)
