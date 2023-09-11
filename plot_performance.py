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

    metric_names = ['Precision', 'Recall', 'F1', 'IoU']
    print('Metrics:', metric_names)
    for metric in metric_names:
        xticks = [5.2] + sorted(list(rf.keys()))[1:]
        tnr = {'fontfamily': 'serif', 'fontsize': 10}
        fig, ax = plt.subplots(1,1, dpi=300, tight_layout=True, figsize=(6,4))
        ax.set_facecolor('#ebebeb')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#3f3f3f')
        ax.spines['left'].set_color('#3f3f3f')
        ax.set_xlabel('Number of Annotations per Class', **tnr)
        ax.set_ylabel(metric, **tnr)
        ax.set_xticks(xticks, **tnr)
        ax.set_xticklabels(['5'] + xticks[1:], fontfamily='serif', fontsize=8)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontfamily='serif', fontsize=10)
        ax.set_ylim(0,1)
        print(list(rf.keys()))
        ax.plot(xticks, [np.mean(list(rf[k][metric.lower()].values())) for k in sorted(rf.keys())], label='Random Forests', color='blue', marker='o')
        ax.plot(xticks, [np.mean(list(svm[k][metric.lower()].values())) for k in sorted(svm.keys())], label='Support Vector Machines', color='red', marker='o')
        ax.axhline(y=np.mean(list(ntf[metric.lower()].values())), color='black', linestyle='--')
        ax.scatter(0, np.mean(list(ntf[metric.lower()].values())), label='Ours with 5.2 Annotations', color='black', marker='o')
        ax.legend(prop={'family': 'serif', 'size': 10})
        fig.savefig(f'{metric}.png')
