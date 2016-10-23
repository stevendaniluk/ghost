# A collection of useful functions for tensorflow

import tensorflow as tf
import tflearn

# Attach summaries to a Tensor
def variable_summaries(var, name):
	# -Mean
	#	-Standard Deviation
	#	-Max
	#	-Min
	#	-Histogram of values
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(var)
    tf.scalar_summary("mean/" + name, mean)
    with tf.name_scope("stddev"):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary("stddev/" + name, stddev)
    tf.scalar_summary("max/" + name, tf.reduce_max(var))
    tf.scalar_summary("min/" + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

# Create a weight variable with appropriate initialization
def weight_variable(shape, name = None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  if name is not None:
  	variable_summaries(initial, name + "/weights")
  return tf.Variable(initial)

# Create a bias variable with appropriate initialization
def bias_variable(shape, name = None):
  initial = tf.constant(0.0, shape=shape)
  if name is not None:
  	variable_summaries(initial, name + "/biases")
  return tf.Variable(initial)

# Create a convolution layer
def conv2d(x, W, b, stride):
  conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
  return tf.nn.bias_add(conv, b)

# Create a transpose convolution layer
def conv2d_transpose(x, W, b, shape, stride):
	conv = tf.nn.conv2d_transpose(x, W, shape, strides=[1, stride, stride, 1], padding="SAME")
	return tf.nn.bias_add(conv, b)

# Create a dilated convolution layer
def conv2d_dilated(x, W, b, rate):
	conv = tf.nn.atrous_conv2d(x, W, rate, padding="SAME")
	return tf.nn.bias_add(conv, b)

# Max-pooling operation
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Sub-sampling operation
def subsample(x, factor):
  return tf.nn.max_pool(x, ksize=[1, factor, factor, 1], strides=[1, 1, 1, 1], padding="SAME")

# Normalize a batch
def batch_norm(x, n_out, training, scope='batch_norm'):
	# Code taken from: http://stackoverflow.com/a/34634291/2267819

  """
  Batch normalization on convolutional maps.
  Args:
    x:           Tensor, 4D BHWD input maps
    n_out:       Integer, depth of input maps
    training:    boolean tf.Varialbe, true indicates training phase
    scope:       string, variable scope
  Return:
    normed:      batch-normalized maps
  """
  with tf.variable_scope(scope):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                 name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                  name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
  return normed

# Bottleneck unit
def bottleneck(x, k, rate, training, name = None):
	"""
	Args:
    x:           Tensor - 4D BHWD input
    k:           Integer - [W, H] of kernel
    rate:        Integer - Rate aprameter for dilated convolution
    training:    Bool - Training or inference
  Return:
    normed:      batch-normalized maps
  """

	in_shape = x.get_shape()

	with tf.name_scope("branch_a"):
		# Subsample input by a factor of 2
	  a = subsample(x, 2)

	with tf.name_scope("branch_b"):
		# Convolution to reduce dimensionality 
		with tf.name_scope('b1'):
			if name is not None:
				local_name = name + "/b1"
			else:
				local_name = name

			with tf.name_scope('weights'):
			  W = weight_variable([1, 1, in_shape[3].value, 1], local_name)
			with tf.name_scope('biases'):
			  b = bias_variable([1], local_name)
			conv = conv2d(x, W, b, 1)
			b1 = tf.nn.tanh(conv)

		# Convolution with kernel
		with tf.name_scope('b2'):
			if name is not None:
				local_name = name + "/b2"
			else:
				local_name = name

			with tf.name_scope('weights'):
			  W = weight_variable([k[0], k[1], 1, 1], local_name)
			with tf.name_scope('biases'):
			  b = bias_variable([1], local_name)
			if rate == 1:
				conv = conv2d(b1, W, b, 1)
			else:
				conv = conv2d_dilated(b1, W, b, rate)
			b2 = tf.nn.tanh(conv)

		# Convolution to expand dimensionality
		with tf.name_scope('b3'):
			if name is not None:
				local_name = name + "/b3"
			else:
				local_name = name

			with tf.name_scope('weights'):
			  W = weight_variable([1, 1, 1, in_shape[3].value], local_name)
			with tf.name_scope('biases'):
			  b = bias_variable([in_shape[3].value], local_name)
			b3 = conv2d(b2, W, b, 1)

		# Normalization
		with tf.name_scope('b4'):
			b4 = batch_norm(b3, in_shape[3].value, training)

	with tf.name_scope('prelu_out'):
	  out = tf.nn.tanh(tf.add(a, b4))

	return out
