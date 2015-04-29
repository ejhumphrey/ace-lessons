import json
import numpy as np
import scipy.stats

from itertools import groupby

import pyjams


def hwr(x):
    return x * (x > 0.0)


def mode(*args, **kwargs):
    return scipy.stats.mode(*args, **kwargs)[0]


def mode2(x_in, axis):
    value_to_idx = dict()
    idx_to_value = dict()
    for x in x_in:
        obj = buffer(x)
        if obj not in value_to_idx:
            idx = len(value_to_idx)
            value_to_idx[obj] = idx
            idx_to_value[idx] = x
    counts = np.bincount([value_to_idx[buffer(x)] for x in x_in])
    return idx_to_value[counts.argmax()]


def inarray(ar1, ar2):
    """Test whether each element of an array is present in a second array.

    Returns a boolean array the same shape as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    Parameters
    ----------
    ar1 : array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of `ar1`.

    Returns
    -------
    out : ndarray, bool
        The values of `ar1` that are in `ar2`.
    """
    ar1 = np.asarray(ar1)
    out = np.zeros(ar1.shape, dtype=bool)
    unique1 = np.unique(ar1)
    unique2 = np.unique(ar2)
    if len(unique1) > len(unique2):
        for val in unique2:
            out |= np.equal(ar1, val)
    else:
        for val in unique1:
            out |= np.equal(ar1, val) * (val in unique2)
    return out


def boundary_pool(x_in, index_edges, axis=0, pool_func='mean'):
    """Pool the values of an array, bounded by a set of edges.

    Parameters
    ----------
    x_in : np.ndarray, shape=(n_points, ...)
        Array to pool.
    index_edges : array_like, shape=(n_edges,)
        Boundary indices for pooling the array.
    pool_func : str
        Name of pooling function to use; one of {`mean`, `median`, `max`}.

    Returns
    -------
    z_out : np.ndarray, shape=(n_edges-1, ...)
        Pooled output array.
    """
    fxs = dict(mean=np.mean, max=np.max, median=np.median, mode=mode2)
    assert pool_func in fxs, \
        "Function '%s' unsupported. Expected one of {%s}" % (pool_func,
                                                             fxs.keys())
    pool = fxs[pool_func]
    num_points = len(index_edges) - 1
    axes_order = range(x_in.ndim)
    axes_order.insert(0, axes_order.pop(axis))
    axes_reorder = np.array(axes_order).argsort()
    x_in = x_in.transpose(axes_order)

    z_out = np.empty([num_points] + list(x_in.shape[1:]), dtype=x_in.dtype)
    for idx, delta in enumerate(np.diff(index_edges)):
        if delta > 0:
            z = pool(x_in[index_edges[idx]:index_edges[idx + 1]], axis=0)
        elif delta == 0:
            z = x_in[index_edges[idx]]
        else:
            raise ValueError("`index_edges` must be monotonically increasing.")
        z_out[idx, ...] = z
    return z_out.transpose(axes_reorder)


def run_length_encode(seq):
    """Run-length encode a sequence of items.

    Parameters
    ----------
    seq : array_like
        Sequence to compress.

    Returns
    -------
    comp_seq : list
        Compressed sequence containing (item, count) tuples.
    """
    return [(obj, len(list(group))) for obj, group in groupby(seq)]


def run_length_decode(comp_seq):
    """Run-length decode a sequence of (item, count) tuples.

    Parameters
    ----------
    comp_seq : array_like
        Sequence of (item, count) pairs to decompress.

    Returns
    -------
    seq : list
        Expanded sequence.
    """
    seq = list()
    for obj, count in seq:
        seq.extend([obj]*count)
    return seq


def boundaries_to_durations(boundaries):
    """Return the durations in a monotonically-increasing set of boundaries.

    Parameters
    ----------
    boundaries : array_like, shape=(N,)
        Monotonically-increasing scalar boundaries.

    Returns
    -------
    durations : array_like, shape=(N-1,)
        Non-negative durations.
    """
    if boundaries != np.sort(boundaries).tolist():
        raise ValueError("Input `boundaries` is not monotonically increasing.")
    return np.abs(np.diff(boundaries))


def find_closest_idx(x, y):
    """Find the closest indexes in `x` to the values in `y`."""
    return np.array([np.abs(x - v).argmin() for v in y])


def equals_value(arr, value):
    """Elementwise equals between a 1-d array of objects and a single value.

    Parameters
    ----------
    arr : array_like, shape=(n,)
        Set of values / objects to test for equivalence.
    value : obj
        Object to test against each element of the input iterable.

    Returns
    -------
    out : {ndarray, bool}
        Output array of bools.
    """
    return np.array([_ == value for _ in arr], dtype=bool)


def intervals_to_durations(intervals):
    """Translate a set of intervals to an array of boundaries."""
    return np.abs(np.diff(np.asarray(intervals), axis=1)).flatten()


def compress_samples_to_intervals(labels, time_points):
    """Compress a set of time-aligned labels via run-length encoding.

    Parameters
    ----------
    labels : array_like
        Set of labels of a given type.
    time_points : array_like
        Points in time corresponding to the given labels.

    Returns
    -------
    intervals : np.ndarray, shape=(N, 2)
        Start and end times, in seconds.
    labels : list, len=N
        String labels corresponding to the returned intervals.
    """
    assert len(labels) == len(time_points)
    intervals, new_labels = [], []
    idx = 0
    for label, count in run_length_encode(labels):
        start = time_points[idx]
        end = time_points[min([idx + count, len(labels) - 1])]
        idx += count
        intervals += [(start, end)]
        new_labels += [label]
    return np.array(intervals), new_labels


def load_jamset(filepath):
    """Load a collection of keyed JAMS (a JAMSet) into memory.

    Parameters
    ----------
    filepath : str
        Path to a JAMSet on disk.

    Returns
    -------
    jamset : dict of JAMS
        Collection of JAMS objects under unique keys.
    """
    jamset = dict()
    with open(filepath) as fp:
        for k, v in json.load(fp).iteritems():
            jamset[k] = pyjams.JAMS(**v)

    return jamset


def save_jamset(jamset, filepath):
    """Save a collection of keyed JAMS (a JAMSet) to disk.

    Parameters
    ----------
    jamset : dict of JAMS
        Collection of JAMS objects under unique keys.
    """
    output_data = dict()
    with pyjams.JSONSupport():
        for k, jam in jamset.iteritems():
            output_data[k] = jam.__json__

    with open(filepath, 'w') as fp:
        json.dump(output_data, fp)
