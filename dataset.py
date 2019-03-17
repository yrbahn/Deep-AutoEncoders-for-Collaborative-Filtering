from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob
import argparse
import os
import utils
from model import AutoEncoder

class AutoEncoderDataset(object):
    def __init__(self, data_dir, batch_size,
                 num_epochs, input_dim, 
                 delimiter='\t', mode="train"):
        
        """get input function using dataset api"""
        self._vector_dim = input_dim
        self._sparse_features = AutoEncoder.get_feature_spec(input_dim)
        self._tfrecord_files = glob.glob(os.path.join(data_dir, mode+".tfrecords")) 
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._mode = mode

    def input_fn(self):
        def _parse(example):
            sparse_feature = tf.parse_single_example(
                example, features=self._sparse_features)
            #print(sparse_feature)
            return sparse_feature

        self._dataset = tf.data.TFRecordDataset(self._tfrecord_files)
        #self._dataset = tf.data.TextLineDataset(["/workspace/autoencoder_rec/data_in/ad_id.dict"])
        dataset = self._dataset.map(_parse)
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.repeat(self._num_epochs)
	iterator = dataset.make_one_shot_iterator()

        features = iterator.get_next()
        return features

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True)
    parser.add_argument(
        '--ad2id_file',
        type=str,
        required=True)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2)
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1)

    args = parser.parse_args()

    ad2id_dict = utils.load_ad2id_dict(args.ad2id_file)
 
    dataset = AutoEncoderDataset(
        args.data_dir, args.batch_size, 
        args.num_epochs, len(ad2id_dict))

    out = dataset.input_fn()
    # test code
    with tf.Session() as sess:
        print(sess.run(out))

if __name__ == "__main__":
    main()
