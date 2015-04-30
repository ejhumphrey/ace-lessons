import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import mir_eval
import numpy as np
# import mpld3
import seaborn
seaborn.set()
# mpld3.enable_notebook()

import lexicon as lex

vocab = lex.Strict(157)


def trackwise_scatter(x, y, ax=None, figsize=(8, 8)):
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    assert (x.index == y.index).all()
    ax.vlines(0.5, 0, 1, linestyles=u'dashed', alpha=0.25)
    ax.hlines(0.5, 0, 1, linestyles=u'dashed', alpha=0.25)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.25)
    scat_handle = ax.scatter(x, y)
    ax.xaxis.labelpad = 10
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.labelpad = 10
    # tooltip = mpld3.plugins.PointLabelTooltip(
    #     scat_handle, [str(_) for _ in x.index])
    # mpld3.plugins.connect(fig, tooltip)
    return ax


def chord_idx_to_colors(idx, max_idx=156, no_chord_idx=156, x_chord_idx=-1):
    """Transform a chord class index to an (R, G, B) color.

    Parameters
    ----------
    idx : array_like
        Chord class index.
    max_idx : int, default=156
        Maximum index, for color scaling purposes.
    no_chord_idx : int, default=156
        Index of the no-chord class.
    x_chord_idx : int, default=-1
        Index of the X-chord class (ignored).

    Returns
    -------
    colors : np.ndarray, shape=(1, len(idx), 3)
        Matrix of color values.
    """
    hue = (idx % 12) / 12.0
    value = 0.9 - 0.7*(((idx).astype(int) / 12) / (max_idx / 12.0))
    hsv = np.array([hue, (hue*0) + 0.6, value]).T

    hsv[idx == no_chord_idx, :] = np.array([0, 0.8, 0.0])
    hsv[idx == x_chord_idx, :] = np.array([0.0, 0.0, 0.8])
    return hsv_to_rgb(hsv.reshape(1, -1, 3))


def labels_to_colors(labels):
    """Transform a collection of labels to a color matrix.

    Parameters
    ----------
    labels : array_like
        Collection of chord labels to map to a color space
    """
    label_idx = np.array(vocab.label_to_index(labels))
    label_idx[np.equal(label_idx, None)] = -1
    return chord_idx_to_colors(label_idx.astype(int))


def plot_color_legend(ax, labels, max_labels=20, min_ratio=0.005):
    labels = np.array(labels).flatten()
    unique_labels = np.unique(labels)
    counts = np.array([np.array([_ == y for _ in labels]).sum()
                       for y in unique_labels])
    sidx = np.argsort(counts)[::-1]
    labels = unique_labels[sidx[:max_labels]]
    colors = labels_to_colormap(labels).squeeze()
    for n, (l, c) in enumerate(zip(labels, colors)):
        print counts[sidx[n]], (counts[sidx[n]] / float(np.sum(counts)))
        if (counts[sidx[n]] / float(np.sum(counts))) < min_ratio:
            n -= 1
            break
        ax.bar(left=n, width=1, height=1, fc=c)
        ax.text(n + 0.5, -0.4, l, horizontalalignment='center')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim((-0.5, n + 1.5))
    ax.set_ylim((-0.75, 1))


def draw_annotations(key):
    # Select the JAMS for this key
    ref_jam = refs[key]
    est_jam = ests[0][key]

    # Align all annotations
    dt_labels, est_labels = align_ref_to_est(ref_jam.chord[0], est_jam.chord[0])
    tdc_labels = align_ref_to_est(ref_jam.chord[1], est_jam.chord[0])[0]

    # Draw the annotations
    fig = plt.figure(figsize=(11, 3))
    labels = [dt_labels, tdc_labels, est_labels]
    subject = ['DT', 'TdC', 'XL-0.25']
#     plt.title(ref_jam.)
    plt.xticks([])
    plt.yticks([])
    for idx, (l, n) in enumerate(zip(labels, subject)):
        ax = fig.add_subplot(311 + idx)
        ax.imshow(labels_to_colormap(l), interpolation='nearest', aspect='auto')
        ax.set_ylabel(n)
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig("{0}/rc_{1}_annotations.pdf".format(outdir, key),
                transparent=True)

    # Draw the legend
    fig = plt.figure(figsize=(11, 1.25))
    ax = fig.gca()
    plot_color_legend(ax, labels, 20)
    plt.tight_layout()
    plt.savefig("{0}/rc_{1}_legend.pdf".format(outdir, key),
                transparent=True)
