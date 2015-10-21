"""Evaluation module for chord estimation."""
import argparse
import fnmatch
import numpy as np
import json
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

import mir_eval

import labels as L
import lexicon as lex
import util

STRICT = lex.Strict(157)


def align_labeled_intervals(ref_intervals, ref_labels, est_intervals,
                            est_labels, ref_fill_value=L.NO_CHORD,
                            est_fill_value=L.NO_CHORD):
    """Align two sets of labeled intervals.

    Parameters
    ----------
    ref_intervals : np.ndarray, shape=(n, 2)
        Reference start and end times.
    ref_labels : list, shape=(n,)
        Reference labels.
    est_intervals : np.ndarray, shape=(n, 2)
        Estimated start and end times.
    est_labels : list, shape=(n,)
        Estimated labels.

    Returns
    -------
    durations : np.ndarray, shape=(m, 2)
        Time durations (weights) of each aligned interval.
    ref_labels : list, shape=(m,)
        Reference labels.
    est_labels : list, shape=(m,)
        Estimated labels.
    """
    t_min = ref_intervals.min()
    t_max = ref_intervals.max()
    ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
        ref_intervals, ref_labels, t_min, t_max,
        ref_fill_value, ref_fill_value)

    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, t_min, t_max,
        est_fill_value, est_fill_value)

    # Merge the time-intervals
    intervals, ref_labels, est_labels = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)
    return durations, ref_labels, est_labels


def align_chord_annotations(ref_annot, est_annot, transpose=False):
    """Align two JAMS chord range annotations.

    Parameters
    ----------
    ref_annot : JAMS.Annotation
        Annotation to use as a chord reference.
    est_annot : JAMS.Annotation
        Annotation to use as a chord estimation.
    transpose : bool, default=False
        Transpose all chord pairs to the equivalent relationship in C.

    Returns
    -------
    durations : np.ndarray, shape=(m, 2)
        Time durations (weights) of each aligned interval.
    ref_labels : list, shape=(m,)
        Reference labels.
    est_labels : list, shape=(m,)
        Estimated labels.
    """
    ref_intervals, ref_labels = ref_annot.data.to_interval_values()
    est_intervals, est_labels = est_annot.data.to_interval_values()
    durations, ref_labels, est_labels = align_labeled_intervals(
        ref_intervals=ref_intervals, ref_labels=ref_labels,
        est_intervals=est_intervals, est_labels=est_labels)

    if transpose:
        ref_labels, est_labels = L.relative_transpose(ref_labels, est_labels)

    return durations, ref_labels, est_labels


def v157_strict(reference_labels, estimated_labels):
    '''Compare chords along lexicon157 rules. Chords with qualities
    outside the following are ignored:
        [maj, min, maj7, min7, 7, maj6, min6,
        dim, aug, sus4, sus2, dim7, hdim7, N]

    :usage:
        >>> (ref_intervals,
             ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
        >>> (est_intervals,
             est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
        >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
                est_intervals, est_labels, ref_intervals.min(),
                ref_intervals.max(), mir_eval.chord.NO_CHORD,
                mir_eval.chord.NO_CHORD)
        >>> (intervals,
             ref_labels,
             est_labels) = mir_eval.util.merge_labeled_intervals(
                 ref_intervals, ref_labels, est_intervals, est_labels)
        >>> durations = mir_eval.util.intervals_to_durations(intervals)
        >>> comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)
        >>> score = mir_eval.chord.weighted_accuracy(comparisons, durations)

    :parameters:
        - reference_labels : list, len=n
            Reference chord labels to score against.
        - estimated_labels : list, len=n
            Estimated chord labels to score against.

    :returns:
        - comparison_scores : np.ndarray, shape=(n,), dtype=float
            Comparison scores, in [0.0, 1.0], or -1 if the comparison is out of
            gamut.
    '''
    mir_eval.chord.validate(reference_labels, estimated_labels)

    ref_idx = STRICT.label_to_index(reference_labels)
    est_idx = STRICT.label_to_index(estimated_labels)
    is_invalid = np.equal(ref_idx, None)

    comparison_scores = np.equal(ref_idx, est_idx).astype(float)

    # Drop if invalid
    comparison_scores[is_invalid] = -1.0
    return comparison_scores


COMPARISONS = dict(
    thirds=mir_eval.chord.thirds,
    triads=mir_eval.chord.triads,
    tetrads=mir_eval.chord.tetrads,
    root=mir_eval.chord.root,
    mirex=mir_eval.chord.mirex,
    majmin=mir_eval.chord.majmin,
    sevenths=mir_eval.chord.sevenths,
    v157_strict=v157_strict)

METRICS = COMPARISONS.keys()
METRICS.sort()


def pairwise_score_labels(ref_labels, est_labels, weights, compare_func):
    """Tabulate the score and weight for a pair of annotation labels.

    Parameters
    ----------
    ref_annot : pyjams.RangeAnnotation
        Chord annotation to use as a reference.
    est_annot : pyjams.RangeAnnotation
        Chord annotation to use as a estimation.
    compare_func : method
        Function to use for comparing a pair of chord labels.

    Returns
    -------
    score : float
        Average score, in [0, 1].
    weight : float
        Relative weight of the comparison, >= 0.
    """
    scores = compare_func(ref_labels, est_labels)
    valid_idx = scores >= 0
    total_weight = weights[valid_idx].sum()
    correct_weight = np.dot(scores[valid_idx], weights[valid_idx])
    norm = total_weight if total_weight > 0 else 1.0
    return correct_weight / norm, total_weight


def pairwise_reduce_labels(ref_labels, est_labels, weights, compare_func,
                           label_counts=None):
    """Accumulate estimated timed of a collection label pairs.

    Parameters
    ----------
    ref_annot : pyjams.RangeAnnotation
        Chord annotation to use as a reference.
    est_annot : pyjams.RangeAnnotation
        Chord annotation to use as a estimation.
    compare_func : method
        Function to use for comparing a pair of chord labels.

    Returns
    -------
    label_counts : dict
        Map of reference labels to estimated label counts and support.
    """
    scores = compare_func(ref_labels, est_labels)

    if label_counts is None:
        label_counts = dict()

    for ref, est, s, w in zip(ref_labels, est_labels, scores, weights):
        if s < 0:
            continue
        if ref not in label_counts:
            label_counts[ref] = dict()
        if est not in label_counts[ref]:
            label_counts[ref][est] = dict(count=0.0, support=0.0)
        label_counts[ref][est]['count'] += s*w
        label_counts[ref][est]['support'] += w

    return label_counts


def pair_annotations(ref_jams, est_jams, ref_pattern='*', est_pattern='*'):
    """Align annotations given a collection of jams and regex patterns.

    Note: Uses glob-style filepath matching. See fnmatch.fnmatch for more info.

    Parameters
    ----------
    ref_jams : list
        A set of reference jams.
    est_jams : list
        A set of estimated jams.
    ref_pattern : str, default='*'
        Pattern to use for filtering reference annotation keys.
    est_pattern : str, default='*'
        Pattern to use for filtering estimated annotation keys.

    Returns
    -------
    ref_annots, est_annots : lists, len=n
        Equal length lists of corresponding annotations.
    """
    ref_annots, est_annots = [], []
    for ref, est in zip(ref_jams, est_jams):
        # All reference annotations vs all estimated annotations.
        for ref_annot in ref.chord:
            # Match against the given reference key pattern.
            if not fnmatch.fnmatch(ref_annot.sandbox.key, ref_pattern):
                continue
            for est_annot in est.chord:
                # Match against the given estimation key pattern.
                if not fnmatch.fnmatch(est_annot.sandbox.key, est_pattern):
                    continue
                ref_annots.append(ref_annot)
                est_annots.append(est_annot)

    return ref_annots, est_annots


def score_annotations(ref_annots, est_annots, keys, metrics, verbose=False):
    """Tabulate overall scores for two sets of annotations.

    Parameters
    ----------
    ref_annots : list, len=n
        Filepaths to a set of reference annotations.
    est_annots : list, len=n
        Filepaths to a set of estimated annotations.
    metrics : list, len=k
        Metric names to compute overall scores.

    Returns
    -------
    scores : pd.DataFrame
        Metric scores, (track_key, metric).
    support : pd.DataFrame
        Duration supports, (track_key, metric).
    """
    scores, support = np.zeros([2, len(ref_annots), len(metrics)])
    for n, (ref_annot, est_annot) in enumerate(zip(ref_annots, est_annots)):
        (weights, ref_labels,
            est_labels) = align_chord_annotations(ref_annot, est_annot)
        for k, metric in enumerate(metrics):
            scores[n, k], support[n, k] = pairwise_score_labels(
                ref_labels, est_labels, weights, COMPARISONS[metric])
        if verbose:
            print(pd.DataFrame(scores[n:n+1], columns=METRICS,
                               index=[keys[n]]))

    scores = pd.DataFrame(scores, index=keys, columns=METRICS)
    support = pd.DataFrame(support, index=keys, columns=METRICS)
    return scores, support


def score_annotations_parallel(ref_annots, est_annots, keys, metrics,
                               num_cpus=8, verbose=False):
    """Tabulate overall scores for two sets of annotations.

    Parameters
    ----------
    ref_annots : list, len=n
        Filepaths to a set of reference annotations.
    est_annots : list, len=n
        Filepaths to a set of estimated annotations.
    metrics : list, len=k
        Metric names to compute overall scores.

    Returns
    -------
    scores : np.ndarray, shape=(n, k)
        Resulting annotation-wise scores.
    weights : np.ndarray, shape=(n, k)
        Relative weight of each score.
    """
    def score_one(n, k, ref_annot, est_annot, metric):
        (weights, ref_labels,
            est_labels) = align_chord_annotations(ref_annot, est_annot)
        return (n, k, pairwise_score_labels(
            ref_labels, est_labels, weights, COMPARISONS[metric]))

    def gen(ref_annots, est_annots, metrics):
        for n, (ref, est) in enumerate(zip(ref_annots, est_annots)):
            for k, metric in enumerate(metrics):
                yield (n, k, ref, est, metric)

    scores, support = np.zeros([2, len(ref_annots), len(metrics)])
    pool = Parallel(n_jobs=num_cpus)
    fx = delayed(score_one)
    results = pool(fx(*args) for args in gen(ref_annots, est_annots, metrics))
    for n, k, res in results:
        scores[n, k], support[n, k] = res

    scores = pd.DataFrame(scores, index=keys, columns=METRICS)
    support = pd.DataFrame(support, index=keys, columns=METRICS)
    return scores, support


def reduce_annotations(ref_annots, est_annots, metrics):
    """Collapse annotations to a sparse matrix of label estimation supports.

    Parameters
    ----------
    ref_annots : list, len=n
        Filepaths to a set of reference annotations.
    est_annots : list, len=n
        Filepaths to a set of estimated annotations.
    metrics : list, len=k
        Metric names to compute overall scores.

    Returns
    -------
    all_label_counts : list of dicts
        Sparse matrix mapping {metric, ref, est, support} values.
    """
    label_counts = dict([(m, dict()) for m in metrics])
    for ref_annot, est_annot in zip(ref_annots, est_annots):
        weights, ref_labels, est_labels = align_chord_annotations(
            ref_annot, est_annot, transpose=True)
        for metric in metrics:
            pairwise_reduce_labels(ref_labels, est_labels, weights,
                                   COMPARISONS[metric], label_counts[metric])

    return label_counts


def macro_average(label_counts, sort=True, min_support=0):
    """Tally the support of each reference label in the map.

    Parameters
    ----------
    label_counts : dict
        Map of reference labels to estimations, containing a `support` count.
    sort : bool, default=True
        Sort the results in descending order.
    min_support : scalar
        Minimum support value for returned results.

    Returns
    -------
    labels : list, len=n
        Unique reference labels in the label_counts set.
    scores : np.ndarray, len=n
        Resulting label-wise scores.
    support : np.ndarray, len=n
        Support values corresponding to labels and scores.
    """
    N = len(label_counts)
    labels = [''] * N
    scores, supports = np.zeros([2, N], dtype=float)
    for idx, (ref_label, estimations) in enumerate(label_counts.items()):
        labels[idx] = ref_label
        supports[idx] = sum([_['support'] for _ in estimations.values()])
        scores[idx] = sum([_['count'] for _ in estimations.values()])
        scores[idx] /= supports[idx] if supports[idx] > 0 else 1.0

    labels = np.asarray(labels)
    if sort:
        sidx = np.argsort(supports)[::-1]
        labels, scores, supports = labels[sidx], scores[sidx], supports[sidx]

    # Boolean mask of results with adequate support.
    midx = supports >= min_support
    return labels[midx].tolist(), scores[midx], supports[midx]


def tally_scores(ref_annots, est_annots, min_support, metrics=None):
    """Produce cumulative statistics over a paired set of annotations.

    Parameters
    ----------
    ref_annots : list, len=n
        Filepaths to a set of reference annotations.
    est_annots : list, len=n
        Filepaths to a set of estimated annotations.
    min_support : scalar
        Minimum support value for macro-quality measure.
    metrics : list, len=k, default=all
        Metric names to compute overall scores.

    Returns
    -------
    results : dict
        Score dictionary of {statistic, metric, value} results.
    """
    if metrics is None:
        metrics = COMPARISONS.keys()

    scores, supports = score_annotations(ref_annots, est_annots, metrics)
    scores_macro = scores.mean(axis=0)
    scores_micro = (supports * scores).sum(axis=0) / supports.sum(axis=0)

    results = dict(macro=dict(), micro=dict(), macro_quality=dict())
    for m, smac, smic in zip(metrics, scores_macro, scores_micro):
        results['macro'][m] = smac
        results['micro'][m] = smic

    label_counts = reduce_annotations(ref_annots, est_annots, metrics)
    for m in metrics:
        quality_scores = macro_average(
            label_counts[m], sort=True, min_support=min_support)[1]
        results['macro_quality'][m] = quality_scores.mean()
    return results


def score_jamsets(reference_jamset, estimated_jamset, results_file,
                  ref_annotation_idx=0, est_annotation_idx=0, num_cpus=1,
                  verbose=False):

    ref_jams = util.load_jamset(reference_jamset)
    est_jams = util.load_jamset(estimated_jamset)
    keys = [key for key in est_jams if key in ref_jams]

    ref_annots = [ref_jams[k].chord[ref_annotation_idx] for k in keys]
    est_annots = [est_jams[k].chord[est_annotation_idx] for k in keys]

    kwargs = dict(keys=keys, metrics=METRICS, verbose=verbose)
    if num_cpus > 1:
        score_fx = score_annotations_parallel
        kwargs.update(num_cpus=num_cpus)
    else:
        score_fx = score_annotations

    scores, supports = score_fx(ref_annots, est_annots, **kwargs)
    with open(results_file, 'w') as fp:
        data = dict(scores=scores.to_dict(), supports=supports.to_dict())
        json.dump(data, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("reference_jamset",
                        metavar="reference_jamset", type=str,
                        help=".")
    parser.add_argument("estimated_jamset",
                        metavar="estimated_jamset", type=str,
                        help=".")
    parser.add_argument("results_file",
                        metavar="results_file", type=str,
                        help=".")
    parser.add_argument("--ref_annotation_idx",
                        metavar="ref_annotation_idx", type=int,
                        default=0,
                        help="Path to a JSON file of CQT parameters.")
    parser.add_argument("--est_annotation_idx",
                        metavar="est_annotation_idx", type=int,
                        default=0,
                        help="Path to a JSON file of CQT parameters.")
    parser.add_argument("--num_cpus",
                        metavar="num_cpus", type=int,
                        default=1,
                        help="Path to a JSON file of CQT parameters.")
    parser.add_argument("--verbose",
                        dest='verbose', action='store_true', default=False,
                        help="")
    args = parser.parse_args()

    score_jamsets(
        args.reference_jamset, args.estimated_jamset, args.results_file,
        args.ref_annotation_idx, args.est_annotation_idx, args.num_cpus,
        args.verbose)
