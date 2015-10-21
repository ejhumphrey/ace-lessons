from __future__ import print_function
import argparse
import glob
import json
import os
import time

import ace_tools.convert_to_jamsv2 as J2


def process_one(idx, input_file, output_file, parallel):
    print("[{0}] {1}\tStarting: {2}".format(time.asctime(), idx, output_file))
    jamset = J2.modernize_jamset(json.load(open(input_file)), parallel)
    J2.save_jamset_v2(jamset, output_file, parallel)
    print("[{0}] Finished!".format(time.asctime()))


def process_all(base_dir, output_ext='jv2', max_depth=4, parallel=False):
    fpatterns = [os.path.join(base_dir, *(["*"]*n + ["*.jamset"]))
                 for n in range(max_depth)]
    fpaths = []
    for fmt in fpatterns:
        fpaths.extend(glob.glob(fmt))

    for idx, input_file in enumerate(fpaths):
        output_file = '{0}.{1}'.format(os.path.splitext(input_file)[0],
                                       output_ext.strip('.'))
        process_one(idx, input_file, output_file, parallel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("base_dir",
                        metavar="old_jamset", type=str,
                        help="")
    parser.add_argument("--output_ext",
                        metavar="output_ext", type=str, default='jv2',
                        help="")
    parser.add_argument("--max_depth",
                        metavar="max_depth", type=int, default=4,
                        help="")
    parser.add_argument("--parallel",
                        action='store_true',
                        help="Path to a JSON file of CQT parameters.")
    args = parser.parse_args()
    process_all(args.base_dir, args.output_ext, args.max_depth, args.parallel)
