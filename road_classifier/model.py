# Road classifier model
#
# ***Not the proper architecture, just an initial version to verify everything works
#
# Architecture:
#		-Layer 1: Convolution with dropout - 5x5 kernel, 3 output channels
#		-Layer 2: Convolution with dropout - 5x5 kernel, 1 output channel
#		-Layer 3: Fully connected layer - Single output channel

import tensorflow as tf
import TensorflowUtils as utils
import scipy
import parameters as params

num_pixels = params.res["height"]*params.res["width"]

training = tf.placeholder(tf.bool, name='training')

# Dropout variable
with tf.name_scope('dropout'):
	keep_prob = tf.placeholder(tf.float32)

# Make input and output variables
x = tf.placeholder(tf.float32, shape=[None, params.res["height"], params.res["width"], 3])
y_ = tf.placeholder(tf.bool, shape=[None, params.res["height"], params.res["width"]])

# Normalize the image
x_image = tf.sub(x, 0.5)

# Convolution
layer_name = "conv1_1"
with tf.name_scope(layer_name):
	with tf.name_scope('weights'):
	  W = utils.weight_variable([5, 5, 3, 3], layer_name)
	with tf.name_scope('biases'):
	  b = utils.bias_variable([3], layer_name)

	conv = utils.conv2d(x, W, b, 1)
	tanh = tf.nn.relu(conv)
	conv1_1 = tf.nn.dropout(tanh, keep_prob)

# Convolution
layer_name = "conv1_2"
with tf.name_scope(layer_name):
	with tf.name_scope('weights'):
	  W = utils.weight_variable([5, 5, 3, 1], layer_name)
	with tf.name_scope('biases'):
	  b = utils.bias_variable([1], layer_name)

	conv = utils.conv2d(conv1_1, W, b, 1)
	tanh = tf.nn.tanh(conv)
	conv1_2 = tf.nn.dropout(tanh, keep_prob)

	conv1_2_flat = tf.reshape(conv1_2, [-1, num_pixels])

# Fully connected layer 1
layer_name = "fcn1_1"
with tf.name_scope(layer_name): 
	with tf.name_scope('weights'):
	  W = utils.weight_variable([num_pixels, num_pixels], layer_name)
	with tf.name_scope('biases'):
	  b = utils.bias_variable([num_pixels], layer_name)
	
	fcn = tf.nn.bias_add(tf.matmul(conv1_2_flat, W), b)
	tanh = tf.nn.tanh(fcn)
	fcn1_1 = tf.reshape(tanh, [-1, params.res["height"], params.res["width"]])

# Form logits and a prediction
y = fcn1_1
prediction = tf.cast(tf.round(tf.sigmoid(y)), tf.bool)

# Image summarries
tf.image_summary("input", x, max_images=1)
tf.image_summary("label", tf.expand_dims(tf.cast(y_, tf.float32), dim=3), max_images=1)
tf.image_summary("prediction", tf.expand_dims(tf.cast(prediction, tf.float32), dim=3), max_images=1)
