from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six


_OPTIMIZER_CLS_NAMES = {
    'Adagrad': tf.train.AdagradOptimizer,
    'Adam': tf.train.AdamOptimizer,
    'Ftrl': tf.train.FtrlOptimizer,
    'RMSProp': tf.train.RMSPropOptimizer,
    'SGD': tf.train.GradientDescentOptimizer,
}


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_optimizer_instance(opt, learning_rate=None):
    if isinstance(opt, six.string_types):
        if opt in six.iterkeys(_OPTIMIZER_CLS_NAMES):
            if not learning_rate:
                raise ValueError('learning_rate must be specified when opt is string.')
            return _OPTIMIZER_CLS_NAMES[opt](learning_rate=learning_rate)
        raise ValueError(
            'Unsupported optimizer name: {}. Supported names are: {}'.format(
                opt, tuple(sorted(six.iterkeys(_OPTIMIZER_CLS_NAMES)))))
    if not isinstance(opt, tf.train.Optimizer):
        raise ValueError(
            'The given object is not an Optimizer instance. Given: {}'.format(opt))
    return opt
   

def load_ad2id_dict(id_dict_file):
    map_dict = {}
    with open(id_dict_file, "r") as f:
        for line in f:
            splited_line = line.split()
            if len(splited_line) != 2:
                raise ValueError("id_dict format is incorrect!")
            u, i = splited_line
            map_dict[u] = i
    return map_dict


def tf_print(tensor, message='', sparse_tensor=False):
    """
    print a tensor for debugging
    """
    def _print_tensor(tensor):
        print(message, tensor)
        return tensor

    if not sparse_tensor:
        log_op = tf.py_func(_print_tensor, [tensor], [tensor.dtype])[0]
        with tf.control_dependencies([log_op]):
            res = tf.identity(tensor)

        return res
    else:
        tensor_values = tensor.values
        log_op = tf.py_func(_print_tensor, [tensor_values], [tensor_values.dtype])[0]
        with tf.control_dependencies([log_op]):
            res = tf.identity(tensor_values)

        return res
