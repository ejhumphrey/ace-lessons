from __future__ import print_function
import json
import jams
import time


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
        namespace='chord', data=data, sandbox=annot['sandbox'],
        annotation_metadata=annot['annotation_metadata'])


def modernize_chord_jam(jam):
    annots = [modernize_chord_annotation(adata)
              for adata in jam['chord']]
    file_metadata = jam['file_metadata'].copy()
    del file_metadata['jams_version']
    jam = jams.JAMS(annotations=annots, file_metadata=file_metadata,
                    sandbox=jam['sandbox'])
    return jams.JAMS.loads(jam.dumps())


def modernize_jamset(jset, verbose=False):
    newset = dict()
    for count, (key, jdata) in enumerate(jset.iteritems()):
        newset[key] = modernize_chord_jam(jdata)
        if verbose:
            print("[{0}] {1} / {2} \tFinished: {3}"
                  "".format(time.asctime(), count, len(jset), key))
    return newset


def load_jamset_v2(filepath):
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

    # TODO: Use joblib to speed this up.
    return dict((k, jams.JAMS.loads(v)) for k, v in raw_jams.iteritems())


def save_jamset_v2(jamset, filepath):
    """Save a collection of keyed JAMS (a JAMSet) to disk.

    Parameters
    ----------
    jamset : dict of JAMS
        Collection of JAMS objects under unique keys.
    """
    # TODO: Use joblib to speed this up.
    output_data = dict((k, j.dumps()) for k, j in jamset.iteritems())

    with open(filepath, 'w') as fp:
        json.dump(output_data, fp)
