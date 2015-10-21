from __future__ import print_function
import argparse
import json
import jams
import sys
import time

from joblib import Parallel, delayed


def modernize_chord_annotation(annot):
    # fields = ['time', 'duration', 'value', 'confidence']
    data = []
    for item in annot['data']:
        t0 = item['start']['value']
        t1 = item['end']['value']
        label = item['label']['value']
        record = dict(time=t0,
                      duration=t1 - t0,
                      value=str(label),
                      confidence=1.0)
        data.append(record)
    return jams.Annotation(
        namespace='chord', data=data, sandbox=annot.get('sandbox', dict()),
        annotation_metadata=annot.get('annotation_metadata', dict()))


def modernize_chord_jam(key, jam):
    annots = [modernize_chord_annotation(adata)
              for adata in jam['chord']]
    file_metadata = jam['file_metadata'].copy()
    del file_metadata['jams_version']
    jam = jams.JAMS(annotations=annots, file_metadata=file_metadata,
                    sandbox=jam.get('sandbox', {}))
    return key, jams.JAMS.loads(jam.dumps())


def modernize_jamset(jset, parallel=False):
    num_cpus = 1 if not parallel else -1
    pool = Parallel(n_jobs=num_cpus)
    fx = delayed(modernize_chord_jam)
    return dict(pool(fx(k, j) for k, j in jset.iteritems()))


def _deserialize(key, value):
    return key, jams.JAMS.__json_init__(**value)


def load_jamset_v2(filepath, parallel=False):
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

    with open(filepath) as fp:
        raw_jams = json.load(fp)

    num_cpus = 1 if not parallel else -1
    pool = Parallel(n_jobs=num_cpus)
    fx = delayed(_deserialize)
    return dict(pool(fx(k, v) for k, v in raw_jams.iteritems()))


def _serialize(key, jam):
    return key, jam.__json__


def save_jamset_v2(jamset, filepath, parallel=False):
    """Save a collection of keyed JAMS (a JAMSet) to disk.

    Parameters
    ----------
    jamset : dict of JAMS
        Collection of JAMS objects under unique keys.
    """
    num_cpus = 1 if not parallel else -1
    pool = Parallel(n_jobs=num_cpus)
    fx = delayed(_serialize)

    output_data = dict(pool(fx(k, j) for k, j in jamset.iteritems()))
    with open(filepath, 'w') as fp:
        json.dump(output_data, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("old_jamset",
                        metavar="old_jamset", type=str,
                        help="Path to an old jamset version.")
    parser.add_argument("new_jamset",
                        metavar="new_jamset", type=str,
                        help="Path to the output (modern) jamset.")
    parser.add_argument("--parallel",
                        action='store_true',
                        help="Path to a JSON file of CQT parameters.")
    args = parser.parse_args()

    print("[{0}] Starting...".format(time.asctime()))
    jamset = modernize_jamset(json.load(open(args.old_jamset)),
                              args.parallel)
    save_jamset_v2(jamset, args.new_jamset, args.parallel)
    print("[{0}] Finished!".format(time.asctime()))
