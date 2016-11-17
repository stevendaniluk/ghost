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
def weight_variable(shape, name=None):
	with tf.name_scope('weights'):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  if name is not None:
	  	variable_summaries(initial, name + "/weights")
	  return tf.Variable(initial)

# Create a bias variable with appropriate initialization
def bias_variable(shape, name = None):
	with tf.name_scope('biases'):
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

# Batch normalization
def batch_norm(inputT, is_training=True, scope=None):
	# Scope is mandatory
	# Variables are updated one layer at a time (not the most efficient, but more convenient)
  return tf.cond(is_training, 
  	             lambda: tf.contrib.layers.batch_norm(inputT, is_training=True, updates_collections=None, center=False, scope=scope, reuse = False), 
  	             lambda: tf.contrib.layers.batch_norm(inputT, is_training=False, updates_collections=None, center=False, scope=scope, reuse = True))

# Bottleneck unit
def bottleneck(x, k, rate=1, training=True, batch_norm=False, name=None, summaries=False):
	"""
	Bottleneck unit consists of the following layers:
	  -1x1 convolution to reduce dimensionality (halfed)
	  -kxk convolution at reduced dimensionality
	  -1x1 convolution to increase dimensionality
	  -Skip connection from the input before activation

	Args:
    x:           Tensor - 4D BHWD input
    k:           Integer - [W, H] of kernel
    rate:        Integer - Rate aprameter for dilated convolution
    training:    Bool - Training or inference
    summaries:   Bool - Record summaries
    name:        String - Layer name for scoping
  Return:
    out:         Output tensor, same dimensions as x
  """

	in_shape = x.get_shape()
	ch_in = in_shape[3].value
	ch_reduced = ch_in/2

	# Convolution to reduce dimensionality 
	with tf.name_scope('b1'):
		if summaries:
			local_name = name + "/b1"
		else:
			local_name = None

		with tf.name_scope('weights'):
		  W = weight_variable([1, 1, ch_in, ch_reduced], local_name)
		with tf.name_scope('biases'):
		  b = bias_variable([ch_reduced], local_name)
		conv = conv2d(x, W, b, 1)
		b1 = tf.nn.tanh(conv)

	# Convolution with kernel
	with tf.name_scope('b2'):
		if summaries:
			local_name = name + "/b2"
		else:
			local_name = None

		with tf.name_scope('weights'):
		  W = weight_variable([k[0], k[1], ch_reduced, ch_reduced], local_name)
		with tf.name_scope('biases'):
		  b = bias_variable([ch_reduced], local_name)

		if rate == 1:
			conv = conv2d(b1, W, b, 1)
		else:
			conv = conv2d_dilated(b1, W, b, rate)
		b2 = tf.nn.tanh(conv)

	# Convolution to expand dimensionality
	with tf.name_scope('b3'):
		if summaries:
			local_name = name + "/b3"
		else:
			local_name = None

		with tf.name_scope('weights'):
		  W = weight_variable([1, 1, ch_reduced, ch_in], local_name)
		with tf.name_scope('biases'):
		  b = bias_variable([ch_in], local_name)
		b3 = conv2d(b2, W, b, 1)

	# Normalization
	if batch_norm:
		with tf.variable_scope(name + "/norm") as scope:
			b4 = batch_norm(b3, is_training=training, scope=scope)
	else:
		b4 = b3

	with tf.name_scope('out'):
	  out = tf.add(x, b4)

	return out
