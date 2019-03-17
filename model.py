import tensorflow as tf
import utils


class AutoEncoder(object):
    def __init__(self, layers, 
                 optimizer='Adam',
                 dropout=None,
                 activation_fn=tf.nn.selu, 
                 is_constrained=True,
		 loss_reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                 top_k=1000,
                 learning_rate=0.05):
        self._hidden_layers   = layers
	self._optimizer       = optimizer
        self._is_constrained  = is_constrained
        self._activation_fn   = activation_fn
        self._learning_rate   = learning_rate
        self._input_dim       = layers[0]
        self._dropout         = dropout
	self._loss_reduction  = loss_reduction
	self._top_k           = top_k

    @staticmethod
    def get_feature_spec(size):
	return { 'sparse': tf.SparseFeature(index_key='indices',
                                            value_key='values',
                                            dtype=tf.float32,
                                            size=size) }

    def _init_encoder_weights(self): 
        # encoder weights and baises init
        self._encoder_weights = []
        self._encoder_biases = [] 
        for i, dims in enumerate(zip(self._hidden_layers, self._hidden_layers[1:])):
            self._encoder_weights.append(
                tf.get_variable(name="encoderWeight%s_%sx%s" % (i, dims[0], dims[1]),
                                    shape=[dims[0], dims[1]],
                                    initializer=tf.contrib.layers.xavier_initializer()))
            self._encoder_biases.append(
                tf.get_variable(name="encoderBias%s_%s" % (i, dims[1]),
                                shape=[dims[1]],
                                initializer=tf.zeros_initializer()))
            
    def _init_decoder_weights(self):
        # decoder weights and baises init
        self._decoder_weights = []

        rev_hidden_layers = list(reversed(self._hidden_layers))
        if self._is_constrained :
            self._decoder_weights = list(reversed(self._encoder_weights))
        else:
            for i, dims in enumerate(zip(rev_hidden_layers, rev_hidden_layers[1:])):
                self._decoder_weights.append(
                    tf.get_variable(name="decoderWeight%s_%sx%s" % (i, dims[0], dims[1]),
                                    shape=[dims[0], dims[1]],
                                    initializer=tf.contrib.layers.xavier_initializer()))
        self._decoder_biases = [] 
        for i in range(len(rev_hidden_layers) -1):        
            self._decoder_biases.append(
                tf.get_variable(name="decoderBias%s_%s" % (i, rev_hidden_layers[i+1]),
                                shape=[rev_hidden_layers[i+1]],
                                initializer=tf.zeros_initializer()))
 
    # encoder
    def _encoder(self, x, mode):
        with tf.variable_scope("encoder"):
            self._init_encoder_weights()
            for i, weight in enumerate(self._encoder_weights):
                x = self._activation_fn(
                    tf.add(
                        tf.matmul(x, weight),
                        self._encoder_biases[i]))
            if self._dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
  	        x = tf.layers.dropout(x, rate=self._dropout, training=True)
                #x = utils.tf_print(x, "x=")
            return x

    # decoder 
    def _decoder(self, x, mode):
        with tf.variable_scope("decoder"):
            self._init_decoder_weights()
            if self._is_constrained :
                for i, weight in enumerate(self._decoder_weights):
                    x = self._activation_fn(
                        tf.add(
                            tf.matmul(x, tf.transpose(weight, perm=[1,0])),
                            self._decoder_biases[i]))
                    #x = utils.tf_print(x, "_x=")
                return x
            else:
                for i, weight in enumerate(self._decoder_weights):
                    x = self._activation_fn(
                        tf.add(
                            tf.matmul(x, weight),
                            self._decoder_biases[i]))
                return x

    def _get_loss(self, labels, y_pred,  scope_name="loss"):
        with tf.variable_scope(scope_name):
            masks = tf.not_equal(y_pred, 0)
            masks = tf.to_float(masks)
            loss = tf.losses.mean_squared_error(
                labels,
                y_pred,
                weights=masks,
                reduction=self._loss_reduction)
            return loss

    def serving_input_receiver_fn(self):
        def serving_input_receiver_fn():

            feature_spec = AutoEncoder.get_feature_spec(self._input_dim)
            
            serialized_tf_example = tf.placeholder(dtype=tf.string,
                                                   shape=[None],
                                                   name='input_example_tensor')

            receiver_tensors = {'example': serialized_tf_example}

            features = tf.parse_example(serialized_tf_example, feature_spec)

            return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
        return serving_input_receiver_fn

    def model_fn(self, features, labels, mode, params):
	sparse_inputs = features["sparse"]
        #sparse_inputs = utils.tf_print(sparse_inputs, "sparse:", True)
 
        inputs = tf.sparse_tensor_to_dense(
            sparse_inputs, default_value=0.0)

        #inputs = utils.tf_print(inputs, "inputs:")
        inputs = tf.reshape(inputs, shape=[-1, self._input_dim])

        y_pred = self._decoder(self._encoder(inputs, mode), mode)

        with tf.name_scope(None, 'predictions', (y_pred,)):
            values, indices = tf.nn.top_k(y_pred, self._top_k, name="topk_y_pred")
            predictions = {
                'indices' : indices,
                'values' : values
            }
         
            prediction_output = tf.estimator.export.PredictOutput({
                'indices' : indices,
                'values' : values})

        #y_pred = utils.tf_print(y_pred)
        if mode == tf.estimator.ModeKeys.PREDICT:
            #values, indices = tf.nn.top_k(y_pred, self._top_k, name="topk_y_pred")
            #values = tf.reshape(values, [-1, 1, self._top_k])
            #indices = tf.reshape(tf.to_float(indices), [-1, 1, self._top_k])
            #topk_ypred = tf.concat([indices, values], 1)
            #top_y_pred = zip(indices, values)
            #topk_y_pred = [indices, values]
        
            #predictions = {
            #    'indices' : indices,
            #    'values' : values
            #}

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
		export_outputs={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:\
                    prediction_output}                )

        eval_metric_ops = {
            "rmse": tf.metrics.mean_squared_error(
                tf.cast(inputs, tf.float32), y_pred)
        }

        loss = self._get_loss(inputs, y_pred)
           
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, 
                loss=loss,
                eval_metric_ops=eval_metric_ops)
 
        with tf.variable_scope("train_layer"):

            optimizer = utils.get_optimizer_instance(self._optimizer,
                learning_rate=self._learning_rate)

            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            return  tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)
