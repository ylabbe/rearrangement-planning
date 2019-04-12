import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_fields(run_ids,
                fields,
                semilogy=True,
                subplot=None,
                xlabel='epochs',
                ylabel=None,
                title='loss',
                legend=True,
                log_dir=None):
    sns.set()
    assert log_dir is not None, 'log_dir is not set'

    log_dir = Path(log_dir)

    if subplot is not None:
        subp = list(str(subplot))
        subp = [subp[0], subp[1], "".join(subp[2:])]
        ax = plt.subplot(*map(int, subp))
    else:
        plt.figure(figsize=(16, 8))
        ax = plt.subplot(111)

    def plot(frame, field, style, color):
        y = np.stack(frame[field]).reshape(-1,1).flatten()
        ax.plot(frame['epoch'], y, style, c=color)

    dfs = [pd.read_json(log_dir / run_id / 'log.txt', lines=True) for run_id in run_ids]

    for field, style in zip(fields, ('-', '-.', '--')):
        for frame, color in zip(dfs, sns.color_palette()):
            if field in frame:
                plot(frame, field, style, color)

    if legend:
        plt.legend(run_ids)
    if semilogy:
        plt.yscale('log')
    plt.xlabel(xlabel)
    if ylabel is None and title is not None:
        ylabel = title
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
