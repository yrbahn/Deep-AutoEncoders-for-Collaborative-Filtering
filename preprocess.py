from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob
import argparse
import os
import math
import utils
import base64

_UNKNOWN_AD_ID = 0


class Preprocess(object):
    def __init__(self, data_dir, save_dir, ad2id_file,
                 user_rating_file, mode='train', delimiter='\t'):
        """get input function using dataset api"""
        self._data_dir = data_dir
        self._mode = mode
        self._filenames = glob.glob(os.path.join(data_dir, "*.txt"))
        self._delimiter = delimiter
        self._save_dir = save_dir
        self._user_rating_file = user_rating_file

        if not ad2id_file:
            print("building a ad2id dict...")
            self._ad2id_map = \
                Preprocess.build_ad2id_maps(self._filenames, delimiter=delimiter)
            print("saving ad2id dict")
            self._save_ad2id_maps()
        else:
            self._ad2id_map = \
                self._load_ad2id_map(ad2id_file, delimiter)

        self._vector_dim = len(self._ad2id_map) # input_dim
        print("vector_dim: %d" % self._vector_dim)
        self.data = self._preprocess()
       
    def _save_ad2id_maps(self):
        with open(os.path.join(self._save_dir, "ad2id.dict"), "w") as output:
            for key, val in sorted(self._ad2id_map.items(), key=lambda k:k[0]):
                output.write("{}\t{}\n".format(key, val))

    def _load_ad2id_map(self, ad2id_file, delimiter):
        print("load ad2id %s" % ad2id_file)
        ad2id_dict = {}
        with open(ad2id_file, "r") as f:
            for line in f:
                s_line = line.split('\t')
                ad2id_dict[s_line[0]] = int(s_line[1])
        return ad2id_dict

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
                    key = parts[0]
                    if parts[1] in self._ad2id_map:
                        value = self._ad2id_map[parts[1]]
                    else:
                        value =  _UNKNOWN_AD_ID
                    rating = float(parts[2])
                    rating = math.log(1.+rating) 

                    if key not in data:
                        data[key] = []
                    data[key].append((value, rating)) 

        if self._user_rating_file :
            user_rating_f = open(self._user_rating_file, "w")
        else:
            user_rating_f = None

        filename = os.path.join(self._save_dir, self._mode +'.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)
        for u_id, val_list in data.items():
            i = [v[0] for v in val_list]
            vals = [float(v[1]) for v in val_list]
            feature_dict = {
                'indices' : utils.int64_feature(i),
                'values' : utils.floats_feature(vals)}
            example = tf.train.Example(
              features=tf.train.Features(feature=feature_dict))
            example_string = example.SerializeToString()
            writer.write(example_string)
            if user_rating_f:
                index_and_rating = map(lambda x: "{}:{}".format(x[0], x[1]), zip(i, vals))
                index_and_rating = " ".join(index_and_rating)
                user_rating_f.write(
                    "{}\t{}\t{}\n".format(
                        u_id, 
                        base64.b64encode(example_string),
                        index_and_rating))

        writer.close()
        if user_rating_f:
            user_rating_f.close()

    @staticmethod
    def build_ad2id_maps(filenames, min_freq=0, delimiter="\t"):
        ad_freq_map = dict()
        ad2id_map = {'UNKNOWN_AD' : _UNKNOWN_AD_ID }
        ad_id = 1
        for source_file in filenames:
            with open(source_file) as src:
                for line in src:
                    parts = line.strip().split(delimiter)
                    if len(parts) < 3:
                        raise ValueError(
                            'Encountered badly formatted line in {}'
                            .format(source_file))
                    
                    ad_id_orig = str(parts[1])
                    if ad_id_orig not in ad2id_map:
                        ad_freq_map[ad_id_orig] = 1
                    else:
                        ad_freq_map[ad_id_orig] += 1

        for ad, freq in sorted(ad_freq_map.items(), key=lambda k: k[0]):
            if freq > min_freq:
               print(ad_id)
               ad2id_map[ad] = ad_id
               ad_id += 1 
        return ad2id_map


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
    parser.add_argument(
        '--mode',
        type=str,
        default="train")
    parser.add_argument(
        '--ad2id_file',
        type=str,
        default=None)
    parser.add_argument(
        '--user_rating_file',
	type=str,
	default=None)
    args = parser.parse_args()

    Preprocess(args.data_dir,
               args.save_dir,
               args.ad2id_file,
               args.user_rating_file,
               mode=args.mode)

if __name__ == "__main__":
    main()
