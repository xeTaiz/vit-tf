import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from pprint import pprint

default_metrics = ['accuracy', 'precision', 'recall', 'iou', 'f1', 'num_annotations', 'annotation_time']
pretty_name = {
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'iou': 'IoU',
    'f1': 'F1',
    'num_annotations': '# Annotations',
    'annotation_time': 'Time (s)'
}
default_classes = ['lung', 'liver', 'kidney']
sus_questions = [
    'I think that I would like to use this system frequently',
    'I found the system unnecessarily complex',
    'I thought the system was easy to use',
    'I think that I would need the support of a technical person to be able to use this system',
    'I found the various functions in this system were well integrated',
    'I thought there was too much inconsistency in this system',
    'I would imagine that most people would learn to use this system very quickly',
    'I found the system very cumbersome to use',
    'I felt very confident using the system',
    'I needed to learn a lot of things before I could get going with this system'
]
#sus_results = [87.5, 91.66666667, 94.44444444, 91.66666667, 87.5, 97.22222222, 91.66666667, 91.66666667, 86.11111111, 91.66666667]
#sus_stddevs = [14.43375673, 8.703882798, 8.206099399, 8.703882798, 18.96967277, 6.487491201, 8.703882798, 8.703882798, 13.9141185, 8.703882798]

sus_results = [87.5, 87.5, 91.66666667, 87.5, 87.5, 95.83333333, 87.5, 87.5, 83.33333333, 87.5]
sus_stddevs = [13.0558242, 13.0558242, 12.3091491, 13.0558242, 16.85499656, 9.731236802, 13.0558242, 13.0558242, 12.3091491, 13.0558242]


def use_first(a):
    ''' Checks if `a` contains multiple numbers, if yes returns the first, otherwise `a` '''
    if   isinstance(a, (int, float)):  return a
    elif isinstance(a, (list, tuple)): return a[0]
    else: raise Exception('use_first expects either a number or an iterable')

if __name__ == '__main__':
    parser = ArgumentParser('Summarize Userstudy results')
    parser.add_argument('--glob', type=str, default='userstudy/nobls/*/metrics.json', help='glob string to find result jsons')
    parser.add_argument('--output', type=Path, default='userstudy-results')
    parser.add_argument('--metrics', type=str, nargs='+', default=default_metrics, help='Which metrics to summarize from the jsons')
    parser.add_argument('--classes', type=str, nargs='+', default=default_classes, help='Which classes to summarize from the jsons')

    args = parser.parse_args()

    metric_names = args.metrics
    class_names = args.classes
    metric_files = list(Path('.').glob(args.glob))
    print('Processing files:')
    pprint(metric_files)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse JSONs
    metrics = []
    for metricf in metric_files:
        with metricf.open('r', encoding='UTF-8') as f:
            metrics.append(json.load(f))

    for c in class_names:
        print(c, 'annotation_time', [int(use_first(m[c]['annotation_time'])) for m in metrics])

    # Aggregate metrics
    agg_metrics = {
        c: {  # Class
            k: {  # Metric
                'mean': np.mean([use_first(m[c][k]) for m in metrics]),
                'std':  np.std([ use_first(m[c][k]) for m in metrics])
            } for k in metric_names
        } for c in class_names
    }
    print('\n\nAverage metrics:')
    pprint(agg_metrics)

    # Plot Metrics
    plt.rcParams.update({'font.size': 11})
    left_metrics  = [m for m in metric_names if m not in ('num_annotations', 'annotation_time')]
    right_metrics = [m for m in metric_names if m     in ('num_annotations', 'annotation_time')]
    n_metrics = len(metric_names)
    gridspeck = {'width_ratios': [len(left_metrics) / n_metrics, 1 / (n_metrics), 1 / (n_metrics)]}
    fig, (ax, ax2, ax3) = plt.subplots(1,3, tight_layout=True, dpi=300, figsize=(8.5,4), gridspec_kw=gridspeck)
    width = 0.25
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0.9, 1)
    ax.set_xticks(np.arange(len(left_metrics)) + width)
    ax.set_xticklabels([pretty_name[m] for m in left_metrics])

    # ax2.yaxis.tick_right()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xticks([width])
    ax2.set_xticklabels([pretty_name['num_annotations']])

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xticks([width])
    ax3.set_yticks([30, 60, 90, 120, 150, 180, 210])
    ax3.set_xticklabels([pretty_name['annotation_time']])

    offset = 0
    for c in class_names:
        v_mean = [agg_metrics[c][m]['mean'] for m in left_metrics]
        # v_std  = [agg_metrics[c][m]['std']  for m in left_metrics]
        x = np.arange(len(left_metrics)) + offset
        ax.bar(x, v_mean, width, zorder=0, label=c.capitalize())
        # ax.errorbar(x, v_mean, yerr=v_std, fmt='.', color='black', ecolor='#00000069')
        for xpos, val in zip(x, v_mean):
            ax.text(xpos, val-0.005, f'{val:.3f}', color='white', horizontalalignment='center', zorder=10)
        # ax.axhline(y=1.0)
        offset += width
    ax.legend(loc='lower left')

    offset = 0
    for c in class_names:
        v_mean = agg_metrics[c]['num_annotations']['mean']
        v_std  = agg_metrics[c]['num_annotations']['std']
        x = offset
        ax2.bar(x, v_mean, width, zorder=0, label=c.capitalize())
        ax2.errorbar(x, v_mean, yerr=v_std, fmt='_-', color='black', ecolor='#00000069', elinewidth=2)
        ax2.text(x, v_mean-3, f'{v_mean:.1f}\n±{v_std:.1f}', color='white', horizontalalignment='center', zorder=10)
        offset += width

    offset = 0
    for c in class_names:
        v_mean = agg_metrics[c]['annotation_time']['mean']
        v_std  = agg_metrics[c]['annotation_time']['std']
        x = offset
        ax3.bar(x, v_mean, width, zorder=0, label=c.capitalize())
        ax3.errorbar(x, v_mean, yerr=v_std, fmt='_-', color='black', ecolor='#00000069', elinewidth=2)
        ax3.text(x, v_mean-25, f'{v_mean:.1f}\n±{v_std:.1f}', color='white', horizontalalignment='center', zorder=10)

        offset += width

    fig.savefig(out_dir/'plot.png')
    fig.savefig(out_dir/'plot.pdf')

    # Plot SUS results
    fig, ax = plt.subplots(1,1, tight_layout=True, dpi=300, figsize=(8,4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(range(len(sus_questions)))
    ax.set_xticklabels(range(1, len(sus_questions)+1))
    ax.set_ylim(50, 100)
    xs = np.arange(len(sus_questions))
    ax.bar(xs, sus_results, zorder=0)
    for x, y, std in zip(xs, sus_results, sus_stddevs):
        ax.text(x, y-5, f'{y:.1f}', color='white', horizontalalignment='center', zorder=10)
        ax.errorbar(x, y , yerr=std, fmt='_-', color='black', ecolor='#00000069', elinewidth=2)
    ax.axhline(y=68, color='#00000069', linestyle='--', zorder=20)

    fig.savefig(out_dir/'sus.png')
    fig.savefig(out_dir/'sus.pdf')
