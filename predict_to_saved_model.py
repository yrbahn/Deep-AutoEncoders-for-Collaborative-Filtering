import tensorflow as tf
import argparse
import utils
import base64

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.core.framework import graph_pb2
from tensorflow.python.saved_model import loader
from tensorflow.python.tools import saved_model_utils

from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils


def get_examples(query_file, ad_id_cnt):
    result = []
    for s in open(query_file, 'r'):
        arr = s.split('\t')
	example_string = base64.b64decode(arr[1])
        
        result.append(example_string)
    return result


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir",
                        default="../model_dir/",
                        type=str,
                        help="Frozen model file to import")
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--ad_id_cnt", type=int, required=True)
    parser.add_argument("--tags", type=str, default="serve")
    parser.add_argument("--signature_def", type=str, default='serving_default')

    args = parser.parse_args()

    examples = get_examples(args.query_file, args.ad_id_cnt)
    example_lens = len(examples)
    print(example_lens)

    #DEFAULT_TAGS = 'serve'
    tags = args.tags
    signature_def_key = args.signature_def

    meta_graph_def = saved_model_utils.get_meta_graph_def(
        args.export_dir,
        tags)

    outputs_tensor_info = signature_def_utils.get_signature_def_by_key(
        meta_graph_def,
        signature_def_key).outputs

    output_tensor_keys_sorted = sorted(outputs_tensor_info.keys())
    print(output_tensor_keys_sorted)
    output_tensor_names_sorted = [
        outputs_tensor_info[tensor_key].name
        for tensor_key in output_tensor_keys_sorted
    ]

    print(output_tensor_names_sorted)

    with tf.Graph().as_default() as graph:
        session = tf.Session(graph=graph)
        loader.load(session, tags.split(","), args.export_dir)
        
        pred = session.run(output_tensor_names_sorted, 
                           feed_dict={'input_example_tensor:0': examples})
        print(len(pred[0]))
        print(pred)
        #for op in graph.get_operations():
        #    print(op)
 
