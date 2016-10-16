# Road classifier model
#
# ***Not the proper architecture, just an initial version to verify everything works
#
# Architecture:
#		-Layer 1: Convolution - 5x5 kernel, same-padding, 3 input channels, 3 output channels
#		-Layer 2: Convolution with dropout - 5x5 kernel, same-padding, 3 input channels, 1 output channels
#		-Layer 3: Fully connected layer - Single output channel
#
# Summaries saved for the following variables:
#		-Mean, std. dev, max, min, and histogram of all weights and biases
#		-Dropout probability
#		-Images out of convolution layers 1 and 2

import tensorflow as tf
import scipy
import parameters as params

# Create a weight variable with appropriate initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Create a bias variable with appropriate initialization
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Create a convolution layer
def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

# Max-pooling operation
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Attach summaries to a Tensor
# 	-Mean
#		-Standard Deviation
#		-Max
#		-Min
#		-Histogram of values
def variable_summaries(var, name):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

num_pixels = params.res["height"]*params.res["width"]

# Make input and output variables
x = tf.placeholder(tf.float32, shape=[None, params.res["height"], params.res["width"], 3])
y_ = tf.placeholder(tf.bool, shape=[None, params.res["height"], params.res["width"]])

x_image = x

# Convolutional layer 1
layer_name = "conv_1"
with tf.name_scope(layer_name):
	# Variable to hold the state of the weights for the layer (save summaries)
	with tf.name_scope('weights'):
	  W_conv1 = weight_variable([5, 5, 3, 3])
	  variable_summaries(W_conv1, layer_name + "/weights")
	# Variable to hold the state of the biases for the layer (save summaries)
	with tf.name_scope('biases'):
	  b_conv1 = bias_variable([3])
	  variable_summaries(b_conv1, layer_name + "/biases")

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 1) + b_conv1)
	tf.image_summary(layer_name, h_conv1, 3)

# Convolutional layer 2
layer_name = "conv_2"
with tf.name_scope(layer_name):
	# Variable to hold the state of the weights for the layer (save summaries)
	with tf.name_scope('weights'):
	  W_conv2 = weight_variable([5, 5, 3, 1])
	  variable_summaries(W_conv2, layer_name + "/weights")
	# Variable to hold the state of the biases for the layer (save summaries)
	with tf.name_scope('biases'):
	  b_conv2 = bias_variable([1])
	  variable_summaries(b_conv2, layer_name + "/biases")
	
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 1) + b_conv2)
	tf.image_summary(layer_name, h_conv2, 3)
	h_conv2_flat = tf.reshape(h_conv2, [-1, num_pixels])

# Perform dropout
with tf.name_scope('dropout'):
	keep_prob = tf.placeholder(tf.float32)
	tf.scalar_summary('dropout_keep_probability', keep_prob)
	h_conv2_drop = tf.nn.dropout(h_conv2_flat, keep_prob)

# Fully connected layer 1
layer_name = "full_1"
with tf.name_scope(layer_name):
	# Variable to hold the state of the weights for the layer (save summaries)
	with tf.name_scope('weights'):
	  W_fc1 = weight_variable([num_pixels*1, num_pixels])
	  variable_summaries(W_fc1, layer_name + "/weights")
	# Variable to hold the state of the biases for the layer (save summaries)
	with tf.name_scope('biases'):
	  b_fc1 = bias_variable([num_pixels])
	  variable_summaries(b_fc1, layer_name + "biases")
	
	h_fc1 = tf.nn.relu(tf.matmul(h_conv2_drop, W_fc1) + b_fc1)

# Output (sigmoid not applied yet)
y = tf.reshape(h_fc1, [-1, params.res["height"], params.res["width"]])
