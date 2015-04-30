import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import mpld3
import numpy as np
import seaborn

import util
import lexicon as lex

seaborn.set()

VOCAB = dict(strict=lex.Strict(157), soft=lex.Soft(157))


def trackwise_scatter(x, y, fig=None, ax=None, figsize=(8, 8),
                      voffset=50, location='bottom right'):
    if not fig and ax:
        raise ValueError("Cannot specify `ax` without `fig`.")

    if not fig:
        fig = plt.figure(figsize=figsize)

    if not ax:
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
    tooltip = mpld3.plugins.PointLabelTooltip(
        scat_handle, [str(_) for _ in x.index],
        voffset=voffset, location=location)
    mpld3.plugins.connect(fig, tooltip)
    return fig, ax


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


def labels_to_colors(labels, vocab='strict'):
    """Transform a collection of labels to a color matrix.

    Parameters
    ----------
    labels : array_like
        Collection of chord labels to map to a color space.

    Returns
    -------
    colors : np.ndarray, shape=(1, len(idx), 3)
        Matrix of color values.
    """
    label_idx = np.array(VOCAB[vocab].label_to_index(labels))
    label_idx[np.equal(label_idx, None)] = -1
    return chord_idx_to_colors(label_idx.astype(int))


def draw_color_legend(ax, labels, max_labels=20, min_ratio=0.005,
                      vocab='strict'):
    """Transform a collection of labels to a color matrix.

    Parameters
    ----------
    ax : axes
        Axes on which to draw the color legend.
    labels :

    Returns
    -------
    colors : np.ndarray, shape=(1, len(idx), 3)
        Matrix of color values.
    """
    labels = np.concatenate([np.array(_) for _ in labels]).flatten()
    unique_labels = np.unique(labels)
    counts = np.array([np.array([_ == y for _ in labels]).sum()
                       for y in unique_labels])
    sidx = np.argsort(counts)[::-1]
    labels = unique_labels[sidx[:max_labels]]
    colors = labels_to_colors(labels, vocab=vocab).squeeze()
    for n, (l, c) in enumerate(zip(labels, colors)):
        # print counts[sidx[n]], (counts[sidx[n]] / float(np.sum(counts)))
        if (counts[sidx[n]] / float(np.sum(counts))) < min_ratio:
            n -= 1
            break
        ax.bar(left=n, width=1, height=1, fc=c)
        ax.text(n + 0.5, -0.4, l, horizontalalignment='center')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim((-0.5, n + 1.5))
    ax.set_ylim((-0.75, 1))


def plot_annotations(annotations, names, plt_size=(11, 3),
                     leg_size=(11, 1.25), vocab='strict'):
    """
    Parameters
    ----------
    annotations : list of Annotations, len=n
        JAMS RangeAnnotations; will align all to the first.
    names : list of str
        Labels to assign to each annotation, len=n

    Returns
    -------
    fig, ax : plt.figure, plt.axes
    """

    # Align all annotations
    if len(annotations) > 1:
        labels = list(util.align_annotations(*annotations[:2]))
        labels += [util.align_annotations(annotations[0], a)[1]
                   for a in annotations[2:]]
    else:
        labels = util.align_annotations(annotations[0], annotations[0])[0]

    # return labels
    # Draw the annotations
    figs = [plt.figure(figsize=plt_size)]
    plt.xticks([])
    plt.yticks([])
    base_idx = 11 + 100*len(labels)
    for idx, (l, n) in enumerate(zip(labels, names)):
        ax = figs[0].add_subplot(base_idx + idx)
        ax.imshow(labels_to_colors(l, vocab=vocab),
                  interpolation='nearest', aspect='auto')
        ax.set_ylabel(n)
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel("Time")
    plt.tight_layout()

    # Draw the legend
    figs += [plt.figure(figsize=leg_size)]
    ax = figs[-1].gca()
    draw_color_legend(ax, labels, 20, vocab=vocab)
    plt.tight_layout()
    return figs

    # plt.savefig("{0}/rc_{1}_annotations.pdf".format(outdir, key),
    #             transparent=True)
