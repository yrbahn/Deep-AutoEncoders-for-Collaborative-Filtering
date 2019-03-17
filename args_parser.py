import argparse

def create_parser():
    """
    parse command arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data_dir',
        type=str,
        required=True)
    parser.add_argument(
        '--eval_data_dir',
        type=str,
        required=True)
    parser.add_argument(
        '--ad2id_file',
        type=str,
        required=True)
    parser.add_argument(
        '--autoencoder_layers',
	nargs='*',
        required=True,
	type=int)
    parser.add_argument(
        '--learning_rate',
        default=0.05,
        type=float)
    parser.add_argument(
        '--save_checkpoints_steps',
        default=None,
        type=int)
    parser.add_argument(
        '--optimizer',
        default='Adam',
        type=str)
    parser.add_argument(
        '--steps',
        help='Number of train steps to perform',
        type=int,
        default=None)
    parser.add_argument(
        '--eval_steps',
        help='Number of eval steps to perform',
        type=int,
        default=None)
    parser.add_argument(
        '--train_set_size',
        help='Number of samples on the train dataset',
        default=None,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int)
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='num of epoch',
        default=1)
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None)
    parser.add_argument(
        '--min_eval_frequency',
        type=int,
        default=3000)
    parser.add_argument(
        '--dropout',
        type=float,
        default=None)
    parser.add_argument(
        '--log_device_placement',
        action='store_true',
        help="""\
        If present when running in a distributed environment will run on sync mode.\
        """,
        default=False)
    parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
    parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """) 
    return parser


def get_args():
    return create_parser().parse_args()
