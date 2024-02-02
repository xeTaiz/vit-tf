import numpy as np
import matplotlib.pyplot as plt
import json

from argparse import ArgumentParser
from pathlib import Path

def extract_num(s):
    idx1 = s.find('metrics')
    idx2 = s.find('uniform')
    s = s[idx1+7:idx2-2]
    return int(s)

if __name__ == '__main__':
    parser = ArgumentParser("Plot performance")
    parser.add_argument('--data', type=str, required=True, help='Directory containing all data')
    args = parser.parse_args()

    p = Path(args.data)
    assert p.exists() and p.is_dir()
    ntf_file = p / 'ntf_metrics0.0annotated.json'
    rf_files = list(p.glob('rf_metrics*uniform.json'))
    svm_files= list(p.glob('svm_metrics*uniform.json'))
    assert ntf_file.exists() and len(rf_files) > 0 and len(svm_files) > 0
    rf, svm = {}, {}
    for rff in rf_files:
        with open(rff) as fp:
            rf[extract_num(str(fp))] = json.load(fp)
    for svmf in svm_files:
        with open(svmf) as fp:
            svm[extract_num(str(fp))] = json.load(fp)
    with open(ntf_file) as fp:
        ntf = json.load(fp)

    metric_names = ['IoU']  # ['Precision', 'Recall', 'F1', 'IoU']
    print('Metrics:', metric_names)
    for metric in metric_names:
        xticks = [5.2] + sorted(list(rf.keys()))[1:]
        tnr = {'fontfamily': 'serif', 'fontsize': 10}
        fig, (ax2, ax) = plt.subplots(2,1, dpi=300, tight_layout=True, figsize=(6,4), sharex=True)
        ax.set_facecolor('#ebebeb')
        ax2.set_facecolor('#ebebeb')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#3f3f3f')
        ax.spines['left'].set_color('#3f3f3f')
        ax.set_xlabel('Number of Annotations per Class', **tnr)
        ax.set_ylabel(' ', **tnr, loc='top')
        ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticklabels(['0.10', '0.20', '0.30', '0.40', '0.50'], fontfamily='serif', fontsize=10)
        ax.set_ylim(0.0, 0.55)
        ax.xaxis.tick_bottom()
        ax2.set_ylim(0.88, 1.0)
        # ax2.xaxis.tick_top()
        ax2.tick_params(labeltop=False, labelbottom=False, top=False, bottom=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#3f3f3f')
        ax2.set_yticks([0.9, 0.95, 1.0])
        ax2.set_yticklabels(['0.90', '0.95', '1.00'], fontfamily='serif', fontsize=10)
        fig.text(0.03, 0.55, 'Intersection over Union', ha='center', va='center', rotation='vertical', **tnr)
        print(list(rf.keys()))
        ax2.scatter(0, 0.981, label='Ours', color='purple', marker='*', s=64)
        ax2.scatter(0, 0.957, label='SAM-Med3D (turbo)', color='green', marker='x', s=52)
        ax2.scatter(0, 0.906, label='SAM-Med3D (organ)', color='orange', marker='x', s=52)
        ax2.axhline(y=0.981, xmin=0.05, color='purple', linestyle='dotted', alpha=0.7)
        ax2.axhline(y=0.957, xmin=0.05, color='green', linestyle='dotted', alpha=0.7)
        ax2.axhline(y=0.906, xmin=0.05, color='orange', linestyle='dotted', alpha=0.7)
        ax.plot(xticks, [np.mean(list(rf[k][metric.lower()].values())) for k in sorted(rf.keys())], label='Random Forests', color='blue', marker='o')
        ax.plot(xticks, [np.mean(list(svm[k][metric.lower()].values())) for k in sorted(svm.keys())], label='Support Vector Machines', color='red', marker='o')
        ax2.plot(xticks, [np.mean(list(rf[k][metric.lower()].values())) for k in sorted(rf.keys())], label='Random Forests', color='blue', marker='o')
        ax2.plot(xticks, [np.mean(list(svm[k][metric.lower()].values())) for k in sorted(svm.keys())], label='Support Vector Machines', color='red', marker='o')
        ax.set_xticks(xticks, **tnr)
        ax.set_xticklabels(['5'] + xticks[1:], fontfamily='serif', fontsize=8)
        d = 0.015
        kwargs = dict(transform=ax2.transAxes, clip_on=False, linewidth=1, color='#3f3f3f')
        ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.legend(prop={'family': 'serif', 'size': 10}, loc='right')
        fig.savefig(f'{metric}.png')
        fig.savefig(f'{metric}.pdf')
