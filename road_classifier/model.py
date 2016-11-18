# Road classifier model
#
# Based on ENet: "ENet: A Deep Neural Network Architecture for
# Real-Time Semantic Segmentation"
#
# Architecture:
#   SECTION 1:
#		-Convolution - 5x5 kernel, 8 output channels
#
#   SECTION 2:
#   -Asymmetric Convolution - 5x1 and 1x5 kernel
#   -Dilated Convolution - 5x5 kernel, rate 2
#   -Asymmetric Convolution - 5x1 and 1x5 kernel
#   -Dilated Convolution - 5x5 kernel, rate 4
#   -Asymmetric Convolution - 5x1 and 1x5 kernel
#   -Dilated Convolution - 5x5 kernel, rate 8
#
#   SECTION 3:
#   -Convolution - 3x3 kernel, 1 output channel

import tensorflow as tf
import TensorflowUtils as utils
import scipy
import parameters as params

ch = [8, 8]  # Number of channels for each section (except output section)

training = tf.placeholder(tf.bool, name='training')

# Dropout variable
with tf.name_scope('dropout'):
	keep_prob = tf.placeholder(tf.float32)

# Dropout variable
with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)

# Make input and output variables
x = tf.placeholder(tf.float32, shape=[None, params.res["height"], params.res["width"], 1])
y_ = tf.placeholder(tf.bool, shape=[None, params.res["height"], params.res["width"]])
prev_y = tf.placeholder(tf.float32, shape=[None, params.res["height"], params.res["width"]])

prev_y_in = tf.sub(tf.expand_dims(prev_y, 3), 0.5)
x_in = tf.concat(3, [x, prev_y_in])
ch_in = 2

##############################
# Section 1

# Convolution
layer_name = "s1_conv1_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 5, ch_in, ch[0]])
  b = utils.bias_variable([ch[0]])
  conv = utils.conv2d(x_in, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s1_conv1_1 = tf.nn.dropout(tanh, keep_prob)


##############################
# Section 2

# Asymmetric convolution (1x5)
layer_name = "s2_conv1_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([1, 5, ch[0], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d(s1_conv1_1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s2_conv1_1 = tf.nn.dropout(tanh, keep_prob)

# Asymmetric convolution (5x1)
layer_name = "s2_conv1_2"
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 1, ch[1], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d(s2_conv1_1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s2_conv1_2 = tf.nn.dropout(tanh, keep_prob)

# Dilated convolution (rate 2)
layer_name = "s2_conv2_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([1, 5, ch[1], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d_dilated(s2_conv1_2, W, b, 2)

  tanh = tf.nn.tanh(conv)
  s2_conv2_1 = tf.nn.dropout(tanh, keep_prob)

# Asymmetric convolution (1x5)
layer_name = "s2_conv3_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([1, 5, ch[0], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d(s2_conv2_1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s2_conv3_1 = tf.nn.dropout(tanh, keep_prob)

# Asymmetric convolution (5x1)
layer_name = "s2_conv3_2"
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 1, ch[1], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d(s2_conv3_1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s2_conv3_2 = tf.nn.dropout(tanh, keep_prob)

# Dilated convolution (rate 4)
layer_name = "s2_conv4_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([1, 5, ch[1], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d_dilated(s2_conv3_2, W, b, 4)

  tanh = tf.nn.tanh(conv)
  s2_conv4_1 = tf.nn.dropout(tanh, keep_prob)

# Asymmetric convolution (1x5)
layer_name = "s2_conv5_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([1, 5, ch[0], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d(s2_conv4_1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s2_conv5_1 = tf.nn.dropout(tanh, keep_prob)

# Asymmetric convolution (5x1)
layer_name = "s2_conv5_2"
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 1, ch[1], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d(s2_conv5_1, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s2_conv5_2 = tf.nn.dropout(tanh, keep_prob)

# Dilated convolution (rate 8)
layer_name = "s2_conv6_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([1, 5, ch[1], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d_dilated(s2_conv5_2, W, b, 2)

  tanh = tf.nn.tanh(conv)
  s2_conv6_1 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 3

# Convolution
layer_name = "s3_conv1_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 5, ch[1], 1])
  b = utils.bias_variable([1])
  conv = utils.conv2d(s2_conv6_1, W, b, 1)
  
  s3_conv1_1 = conv

##############################

# Form logits and a prediction
y = tf.squeeze(s3_conv1_1, squeeze_dims=[3])
prediction = tf.cast(tf.round(tf.sigmoid(y)), tf.bool)

# Image summarries
tf.image_summary("input", x, max_images=2)
tf.image_summary("label", tf.expand_dims(tf.cast(y_, tf.float32), dim=3), max_images=2)
tf.image_summary("prediction", tf.expand_dims(tf.cast(prediction, tf.float32), dim=3), max_images=2)
