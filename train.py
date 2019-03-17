import tensorflow as tf
import args_parser
import model
import dataset
import utils

tf.logging.set_verbosity(tf.logging.INFO)


def train_and_eval():
    args = args_parser.get_args()

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=args.log_device_placement,
        intra_op_parallelism_threads=args.num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    run_config = tf.contrib.learn.RunConfig(
        session_config=sess_config,
        save_checkpoints_steps=args.save_checkpoints_steps,
        model_dir=args.output_dir)

    ad2id_dict = utils.load_ad2id_dict(args.ad2id_file)
    input_dim = len(ad2id_dict)

    autoencoder_layers = [input_dim] + args.autoencoder_layers
    print("autoencoder layers:", autoencoder_layers)

    autoencoder = model.AutoEncoder(
        autoencoder_layers,
        optimizer=args.optimizer,
        dropout=args.dropout,
        learning_rate=args.learning_rate)

    autoencoder_estimator = tf.estimator.Estimator(
        model_fn=autoencoder.model_fn,
        model_dir=args.output_dir,
        config=run_config,
        params={
        })

    train_dataset = dataset.AutoEncoderDataset(
      args.train_data_dir, args.batch_size, args.num_epochs, input_dim)
    
    eval_dataset = dataset.AutoEncoderDataset(
      args.eval_data_dir, args.batch_size, 1, input_dim, mode="eval")
 
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_dataset.input_fn, 
	max_steps=args.steps)

    export_input_fn = autoencoder.serving_input_receiver_fn()

    #exporter = tf.estimator.FinalExporter(
    #    "saved_model", export_input_fn)
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_dataset.input_fn,
        steps=args.eval_steps)

    tf.estimator.train_and_evaluate(
        autoencoder_estimator, train_spec, eval_spec)
    
    # export
    export_dir = autoencoder_estimator.export_savedmodel(
        args.output_dir,
        export_input_fn)
    
    print("exported %s" % export_dir)

if __name__ == "__main__":
    train_and_eval()

