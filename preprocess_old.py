from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob
import argparse
import os


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class Preprocess(object):
    def __init__(self, data_dir, save_dir, mode='train', delimiter='\t'):
        """get input function using dataset api"""
        self._data_dir = data_dir
        self._mode = mode
        self._filenames = glob.glob(os.path.join(data_dir, "*.txt"))
        self._delimiter = delimiter
        self._save_dir = save_dir
        self._user_id_map, self._ad_id_map = \
            self._build_maps(self._filenames, delimiter)
        self._save_id_maps()
        self._vector_dim = len(self._ad_id_map) # input_dim
        self.data = self._preprocess()
       
    def _save_id_maps(self):
        with open(os.path.join(self._save_dir, "user_id.dict"), "w") as output:
            for key, val in sorted(self._user_id_map.items(), key=lambda k:k[1]):
                output.write("{}\t{}\n".format(key, val))

        with open(os.path.join(self._save_dir, "ad_id.dict"), "w") as output:
            for key, val in sorted(self._ad_id_map.items(), key=lambda k:k[1]):
                output.write("{}\t{}\n".format(key, val))

    def _preprocess(self):
        data = dict()
        
        for source_file in self._filenames:
            with open(source_file) as src:
                for line in src:
                    parts = line.strip().split(self._delimiter)
                    if len(parts)<3:
                        raise ValueError(
                            'Encountered badly formatted line in {}'.
                            format(source_file))
                    key = self._user_id_map[parts[0]]
                    value = self._ad_id_map[parts[1]]
                    rating = int(parts[2])
                    if key not in data:
                        data[key] = []
                    data[key].append((value, rating)) 

        filename = os.path.join(self._save_dir, self._mode + ".libfm")
        with open(filename, "w") as wf:
          for u_id, val_list in data.items():
            i = [v[0] for v in val_list]
            vals = [float(v[1]) for v in val_list]
            sp = ' '.join(map(lambda x: "{}:{}".format(x[-1], x[1]), zip(i, vals)))
            wf.write("{} {}\n".format(len(vals), sp))

    @staticmethod
    def _build_maps(filenames, delimiter="\t"):
        user_id_map = dict()
        ad_id_map = dict()
        
        u_id = 0
        ad_id = 0
        for source_file in filenames:
            with open(source_file) as src:
                for line in src:
                    parts = line.strip().split(delimiter)
                    if len(parts) < 3:
                        raise ValueError(
                            'Encountered badly formatted line in {}'
                            .format(source_file))
                    
                    u_id_orig = str(parts[0])
                    if u_id_orig not in user_id_map:
                        user_id_map[u_id_orig] = u_id
                        u_id += 1
                
                    ad_id_orig = str(parts[1])
                    if ad_id_orig not in ad_id_map:
                        ad_id_map[ad_id_orig] = ad_id
                        ad_id += 1
        return user_id_map, ad_id_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_dir',
        type=str,
        required=True)
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True)

    args = parser.parse_args()
    Preprocess(args.data_dir, args.save_dir)

if __name__ == "__main__":
    main()
