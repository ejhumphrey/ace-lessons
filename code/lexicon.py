import numpy as np
import mir_eval
import labels as L


class Lexicon(object):

    def __init__(self, vocab_dim):
        self.vocab_dim = vocab_dim
        self.num_classes = vocab_dim
        self._label_map = dict()
        self._index_map = dict()

    def label_to_index(self, label):
        """Index chord label.

        Parameters
        ----------
        label : str, or array_like
            Chord label(s) to map.
        """
        singleton = False
        if not np.shape(label):
            label = [str(label)]
            singleton = True

        for l in np.unique(label):
            if l not in self._index_map:
                self.__store_label__(l)

        chord_idx = np.array([self._index_map[l] for l in label])
        return chord_idx[0] if singleton else chord_idx

    def index_to_label(self, index):
        singleton = False
        if not np.shape(index):
            index = [index]
            singleton = True

        for i in np.unique(index):
            if i not in self._label_map:
                self.__store_index__(i)

        chord_labels = [self._label_map[idx] for idx in index]
        return chord_labels[0] if singleton else chord_labels


class Strict(Lexicon):

    def __init__(self, vocab_dim=157):
        Lexicon.__init__(self, vocab_dim=vocab_dim)
        self.valid_qualities = L.QUALITIES[vocab_dim]

    def __store_label__(self, label):
        try:
            row = L.split(label)
        except mir_eval.chord.InvalidChordException:
            row = ['X', '', set(), '']
        skip = [row[0] == 'X',
                not row[1] in self.valid_qualities,
                len(row[2]) > 0,
                not row[3] in ['', '1']]
        if any(skip):
            idx = None
        elif row[0] == 'N':
            idx = self.vocab_dim - 1
        else:
            idx = mir_eval.chord.pitch_class_to_semitone(row[0])
            idx += self.valid_qualities.index(row[1]) * 12
        self._index_map[label] = idx

    def __store_index__(self, index):
        if index < 0 or index >= self.vocab_dim:
            raise ValueError("index out of bounds: %d" % index)
        if index == self.vocab_dim - 1:
            label = "N"
        else:
            label = "%s:%s" % (L.ROOTS[index % 12],
                               L.QUALITIES[self.vocab_dim][int(index) / 12])
        self._label_map[index] = label


class Soft(Lexicon):

    def __init__(self, vocab_dim=157):
        Lexicon.__init__(self, vocab_dim=vocab_dim)
        self.valid_qualities = L.QUALITIES[vocab_dim]

    def __store_label__(self, label):
        try:
            row = L.split(label)
        except mir_eval.chord.InvalidChordException:
            row = ['X', '', set(), '']
        skip = [row[0] == 'X',
                row[1] not in self.valid_qualities]
        if any(skip):
            idx = None
        elif row[0] == 'N':
            idx = self.vocab_dim - 1
        else:
            idx = mir_eval.chord.pitch_class_to_semitone(row[0])
            idx += self.valid_qualities.index(row[1]) * 12

        # weak_flag = [len(row[2]) > 0, row[3] not in ['', '1']]
        self._index_map[label] = idx  # * (-1 if True in weak_flag else 1)

    def __store_index__(self, index):
        if np.abs(index) >= self.vocab_dim:
            raise ValueError("index out of bounds: %d" % index)
        if index == self.vocab_dim - 1:
            label = "N"
        else:
            label = "%s:%s" % (L.ROOTS[index % 12],
                               L.QUALITIES[self.vocab_dim][int(index) / 12])
        self._label_map[index] = label
