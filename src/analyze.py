import argparse
import itertools
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from utils import ATTACKS, NUM_CHECKPOINTS

matplotlib.use('agg')
# Use Type 42 fonts (TrueType) instead of the default Type 3.
matplotlib.rcParams['pdf.fonttype'] = 42
# The font cache is at ~/.cache/matplotlib. Fonts can be viewed from the files in that directory.
# The directory should be deleted after new fonts are installed, so the cache gets rebuilt.
plt.rcParams['font.family'] = 'TeX Gyre Heros'


METRICS = ('euclidean', 'cosine')


def load_data(workspace):
    """Load data and normalize, returning the normalized data."""
    data = {}
    normalization_df = pd.read_csv(os.path.join(workspace, 'pairwise_distances.csv'))
    normalization_lookup = {
        'euclidean': dict(normalization_df[['checkpoint', 'euclidean_mean']].to_records(index=False)),
        'cosine': dict(normalization_df[['checkpoint', 'cosine_mean']].to_records(index=False))
    }
    for metric in METRICS:
        for attack in ATTACKS:
            root_adv_dir = os.path.join(workspace, 'attack', attack)
            adv_success = np.logical_not(np.loadtxt(
                os.path.join(root_adv_dir, 'correct.csv'), dtype=bool, delimiter=','))
            for checkpoint in range(NUM_CHECKPOINTS):
                distances_path = os.path.join(workspace, 'distances', attack, f'{checkpoint}.npz')
                with open(distances_path, 'rb') as f:
                    npz = np.load(f)
                    # Limit to images that were successfully attacked.
                    distances = npz[metric][adv_success]
                    normalized_distances = distances / normalization_lookup[metric][checkpoint]
                    data[(metric, attack, checkpoint)] = normalized_distances
    return data


def plot(data, outdir):
    tick_label_size = 12
    axis_label_size = 13
    column_row_size = 14
    fig, axs = plt.subplots(nrows=len(METRICS), ncols=len(ATTACKS), figsize=(12, 5.0))
    for metric_idx, metric in enumerate(METRICS):
        for attack_idx, attack in enumerate(ATTACKS):
            ax = axs[metric_idx, attack_idx]
            model_data = [data[(metric, attack, checkpoint)] for checkpoint in range(NUM_CHECKPOINTS)]
            parts = ax.violinplot(
                model_data,
                points=100,
                showmeans=False,
                showextrema=False,
                showmedians=False
            )
            for pc in parts['bodies']:
                pc.set_facecolor('#A7C7E7')
                pc.set_edgecolor('black')
                pc.set_linewidth(0.3)  # default 1.0
                pc.set_alpha(1)
            means = [x.mean() for x in model_data]
            ax.scatter(range(1, NUM_CHECKPOINTS + 1), means, s=7, c='black', marker='D', edgecolors='none')
            ax.set_xticks(range(1, NUM_CHECKPOINTS + 1))
            ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
            ax.yaxis.grid(which='major', color='#DCDCDC', linestyle='solid')
            ax.set_axisbelow(True)
            ax.set_ylim((0, 1.75))
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.25))
            # Keep x labels only for bottom row
            if metric_idx != len(METRICS) - 1:
                ax.set_xticklabels([])
            # Keep y labels and ticks for only left row
            if attack_idx != 0:
                ax.set_yticklabels([])
    # Add x-axis label to bottom row
    for idx in range(len(ATTACKS)):
        axs[len(METRICS) - 1, idx].set_xlabel('Checkpoint', fontsize=axis_label_size)
    # Add y-axis label to left column
    for idx in range(len(METRICS)):
        axs[idx, 0].set_ylabel('Distance', fontsize=axis_label_size)
    # Add column header to top row
    for idx in range(len(ATTACKS)):
        axs[0, idx].set_title(f'Attack:\n{ATTACKS[idx].upper()}', pad=10, fontsize=column_row_size)
    # Add row labels to left column
    for idx in range(len(METRICS)):
        pad = 40
        ax = axs[idx, 0]
        ax.annotate(
            f'Distance\nMetric:\nNormalized\n{METRICS[idx].capitalize()}',
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords='offset points',
            ha='center',
            va='center',
            fontsize=column_row_size
        )
    fig.tight_layout()
    fig_path = os.path.join(outdir, 'plot.pdf')
    fig.savefig(fig_path, transparent=True)
    plt.close(fig)
    if shutil.which('pdfcrop'):
        with tempfile.TemporaryDirectory() as tmp:
            cropped_path = os.path.join(tmp, 'cropped.pdf')
            subprocess.run(['pdfcrop', fig_path, cropped_path])
            shutil.copyfile(cropped_path, fig_path)


def tabulate(data, outdir):
    records = []
    product = itertools.product(METRICS, ATTACKS, range(NUM_CHECKPOINTS))
    for (metric, attack, checkpoint) in product:
        series = data[(metric, attack, checkpoint)]
        mean = series.mean()
        median = np.median(series)
        std = series.std(ddof=1)
        record = (metric, attack, checkpoint, mean, median, std)
        records.append(record)
    columns = ('metric', 'attack', 'checkpoint', 'mean', 'median', 'std')
    df = pd.DataFrame.from_records(records, columns=columns)
    df = pd.pivot(
        df, index=['attack', 'checkpoint'], columns='metric', values=['mean', 'median', 'std'])
    df = df.reset_index()
    columns = [f'{c[1]}_{c[0]}' if c[1] else c[0] for c in df.columns]
    df.columns = columns
    df.to_csv(os.path.join(outdir, 'table.csv'), index=False)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace')
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    analysis_dir = os.path.join(args.workspace, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    data = load_data(args.workspace)
    plot(data, analysis_dir)
    tabulate(data, analysis_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
