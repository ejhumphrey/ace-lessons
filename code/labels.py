import json
import mir_eval
import os
import numpy as np

import util


ROOTS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

QUALITIES = {
    25: ['maj', 'min', ''],
    61: ['maj', 'min', 'maj7', 'min7', '7', ''],
    157: ['maj', 'min', 'maj7', 'min7', '7',
          'maj6', 'min6', 'dim', 'aug', 'sus4',
          'sus2', 'dim7', 'hdim7', ''],
}

_QINDEX = dict([(v, dict([(tuple(mir_eval.chord.QUALITIES[q]), i)
                          for i, q in enumerate(QUALITIES[v])]))
                for v in QUALITIES])

# mir_eval aliases
NO_CHORD = mir_eval.chord.NO_CHORD
SKIP_CHORD = mir_eval.chord.X_CHORD
encode = mir_eval.chord.encode
encode_many = mir_eval.chord.encode_many
split = mir_eval.chord.split
join = mir_eval.chord.join
pitch_class_to_semitone = mir_eval.chord.pitch_class_to_semitone


def semitone_matrix(vocab_dim):
    """Return a matrix of semitone vectors for a given vocabulary dimension."""
    if vocab_dim not in QUALITIES:
        raise ValueError(
            "Invalid vocabulary dimension: {0}. "
            "Must be one of {1}.".format(vocab_dim, QUALITIES.keys()))
    return np.array([mir_eval.chord.QUALITIES[q]
                     for q in QUALITIES[vocab_dim]])


def semitone_to_pitch_class(semitone):
    """Convert a semitone to its pitch class.

    Parameters
    ----------
    semitone : int
        Semitone value of the pitch class.

    Returns
    -------
    pitch_class : str
        Spelling of a given pitch class, e.g. 'C#', 'Gbb'
    """
    return ROOTS[semitone % 12]


def semitones_index(semitones, vocab_dim=157):
    """Return the index of the semitone bitvector, or None if undefined."""
    return _QINDEX[vocab_dim].get(tuple(semitones), None)


def chord_label_to_quality_index(label, vocab_dim=157):
    """Map a chord label to its quality index, or None if undefined."""
    singleton = False
    if isinstance(label, str):
        label = [label]
        singleton = True
    root, semitones, bass = mir_eval.chord.encode_many(label)
    quality_idx = [semitones_index(s, vocab_dim) for s in semitones]
    return quality_idx[0] if singleton else quality_idx


def get_quality_index(semitones, vocab_dim):
    return _QINDEX[vocab_dim].get(tuple(semitones), None)


def chord_label_to_chroma(label, bins_per_pitch=1):
    flatten = False
    if isinstance(label, str):
        label = [label]
        flatten = True

    root, semitones, bass = mir_eval.chord.encode_many(label)
    chroma = np.array([mir_eval.chord.rotate_bitmap_to_root(s, r)
                       for s, r in zip(semitones, root)], dtype=int)

    chroma_out = np.zeros([len(chroma), 12*bins_per_pitch])
    chroma_out[:, ::bins_per_pitch] = chroma
    return chroma_out[0] if flatten else chroma_out


def rotate(class_vector, root):
    """Rotate a class vector to C (root invariance)"""
    return np.array([class_vector[(n + root) % 12 + 12*(n/12)]
                     for n in range(len(class_vector) - 1)]+[class_vector[-1]])


def subtract_mod(reference, index, base):
    """Return the distance relative to reference, modulo `base`.

    Note: If 'reference' or `index` is None, this will return `index`.

    Parameters
    ----------
    reference : int
        Reference value.
    index : int
        Value to subtract.
    """
    if None in [reference, index]:
        return None
    ref_idx = reference % base
    idx = index % base
    octave = int(index) / base
    idx_out = base * octave + (idx - ref_idx) % base
    return idx_out


def add_mod(reference, value, base):
    """Return the distance relative to reference, modulo `base`.

    Note: If 'reference' or `value` is None, this will return `index`.

    Parameters
    ----------
    reference : int
        Reference value.
    value : int
        Value to add.
    """
    if None in [reference, value]:
        return None
    ref_idx = reference % base
    idx = value % base
    octave = int(reference) / base
    idx_out = base * octave + (idx + ref_idx) % base
    return idx_out


def _generate_tonnetz_matrix(radii):
    """Return a Tonnetz transform matrix.

    Parameters
    ----------
    radii: array_like, shape=(3,)
        Desired radii for each harmonic subspace (fifths, maj-thirds,
        min-thirds).

    Returns
    -------
    phi: np.ndarray, shape=(12,6)
        Bases for transforming a chroma matrix into tonnetz coordinates.
    """
    assert len(radii) == 3
    basis = []
    for l in range(12):
        basis.append([
            radii[0]*np.sin(l*7*np.pi/6), radii[0]*np.cos(l*7*np.pi/6),
            radii[1]*np.sin(l*3*np.pi/2), radii[1]*np.cos(l*3*np.pi/2),
            radii[2]*np.sin(l*2*np.pi/3), radii[2]*np.cos(l*2*np.pi/3)])
    return np.array(basis)


def chroma_to_tonnetz(chroma, radii=(1.0, 1.0, 0.5)):
    """Return a Tonnetz coordinates for a given chord label.

    Parameters
    ----------
    chroma: str
        Chord label to transform.
    radii: array_like, shape=(3,), default=(1.0, 1.0, 0.5)
        Desired radii for each harmonic subspace (fifths, maj-thirds,
        min-thirds). Default based on E. Chew's spiral model.

    Returns
    -------
    tonnnetz: np.ndarray, shape=(6,)
        Coordinates in tonnetz space for the given chord label.
    """
    phi = _generate_tonnetz_matrix(radii)
    tonnetz = np.dot(chroma, phi)
    scalar = 1 if np.sum(chroma) == 0 else np.sum(chroma)
    return tonnetz / scalar


def chord_label_to_tonnetz(chord_label, radii=(1.0, 1.0, 0.5)):
    chroma = chord_label_to_chroma(chord_label)
    return chroma_to_tonnetz(chroma, radii)


def _load_json_labeled_intervals(label_file):
    """Load labeled intervals from a JSON file.

    Returns
    -------
    intervals : np.ndarray, shape=(N, 2)
        Intervals in time, should be monotonically increasing.
    labels : list, len=N
        String labels corresponding to the given time intervals.
    """
    data = json.load(open(label_file, 'r'))
    chord_labels = [str(l) for l in data['labels']]
    return np.asarray(data['intervals']), chord_labels


LOADERS = {
    "lab": mir_eval.io.load_labeled_intervals,
    "txt": mir_eval.io.load_labeled_intervals,
    "json": _load_json_labeled_intervals
    }


def compress_labeled_intervals(intervals, labels):
    """Collapse repeated labels and the corresponding intervals.

    Parameters
    ----------
    intervals : np.ndarray, shape=(N, 2)
        Intervals in time, should be monotonically increasing.
    labels : list, len=N
        Labels corresponding to the given time intervals.
    """
    intervals = np.asarray(intervals)
    new_labels, new_intervals = list(), list()
    idx = 0
    for label, step in util.run_length_encode(labels):
        new_labels.append(label)
        new_intervals.append([intervals[idx, 0], intervals[idx + step - 1, 1]])
        idx += step
    return np.asarray(new_intervals), new_labels


def load_labeled_intervals(label_file, compress=True):
    ext = os.path.splitext(label_file)[-1].strip(".")
    assert ext in LOADERS, "Unsupported extension: %s" % ext
    intervals, labels = LOADERS[ext](label_file)
    if compress:
        intervals, labels = compress_labeled_intervals(intervals, labels)
    return intervals, labels


def relative_transpose(reference, relative):
    """Rotate a pair of chord names to the equivalent relationships in C.

    Parameters
    ----------
    reference : str or list
        Reference chord names; will return {'C:*', 'N', 'X'}.
    relative : str or list
        Relative chord names.

    Returns
    -------
    new_references : str
        Equivalent reference chords in C.
    new_relatives : str
        Equivalent relationship to the references.
    """
    singleton = False
    if not np.shape(reference) and not np.shape(relative):
        reference = [str(reference)]
        relative = [str(relative)]
        singleton = True
    elif np.shape(reference) != np.shape(relative):
        raise ValueError("Inputs must have same shape")

    ref_roots = encode_many(reference)[0]
    rel_roots = encode_many(relative)[0]
    new_roots = (rel_roots - ref_roots) % 12

    new_refs, new_rels = list(), list()
    for ref, rel, root in zip(reference, relative, new_roots):
        if ref not in [NO_CHORD, SKIP_CHORD]:
            ref = join('C', *list(split(ref)[1:]))
        if rel not in [NO_CHORD, SKIP_CHORD]:
            rel = join(ROOTS[root], *list(split(rel)[1:]))
        new_refs.append(ref)
        new_rels.append(rel)

    return (new_refs[0], new_rels[0]) if singleton else (new_refs, new_rels)
