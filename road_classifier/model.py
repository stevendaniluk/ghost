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
#   -Downsampling - 5x5 kernel, 16 output channels
#   -Bottleneck regular convolution - 5x5 kernel
#   -Bottleneck regular convolution - 5x5 kernel
#
#   SECTION 3:
#   -Downsampling - 5x5 kernel, 32 output channels
#   -Bottleneck regular convolution - 5x5 kernel
#   -Bottleneck regular convolution - 5x5 kernel
#
#   SECTION 4:
#   -Bottleneck dilated convolution - 5x5 kernel, rate 2
#   -Bottleneck asymmetric convolution - 5x5 kernel
#   -Bottleneck dilated convolution - 5x5 kernel, rate 4
#   -Bottleneck asymmetric convolution - 5x5 kernel
#   -Bottleneck dilated convolution - 5x5 kernel, rate 8
#   -Bottleneck asymmetric convolution - 5x5 kernel
#   -Bottleneck dilated convolution - 5x5 kernel, rate 16
#   -Bottleneck asymmetric convolution - 5x5 kernel
#
#   SECTION 5:
#   -Upsampling - 5x5 kernel, 16 output channels
#   -Bottleneck regular convolution - 5x5 kernel
#   -Bottleneck regular convolution - 5x5 kernel
#
#   SECTION 6:
#   -Upsampling - 5x5 kernel, 8 output channels
#   -Bottleneck regular convolution - 5x5 kernel
#   -Bottleneck regular convolution - 5x5 kernel
#
#   SECTION 7:
#   -Convolution - 5x5 kernel, 1 output channel

import tensorflow as tf
import TensorflowUtils as utils
import scipy
import parameters as params

training = tf.placeholder(tf.bool, name='training')

# Dropout variable
with tf.name_scope('dropout'):
	keep_prob = tf.placeholder(tf.float32)

# Make input and output variables
x = tf.placeholder(tf.float32, shape=[None, params.res["height"], params.res["width"], 3])
y_ = tf.placeholder(tf.bool, shape=[None, params.res["height"], params.res["width"]])

ch = [8, 16, 32, 32, 16, 8]  # Number of channels for each section

##############################
# Section 1

# Convolution
layer_name = "s1_conv1_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 5, 3, ch[0]])
  b = utils.bias_variable([ch[0]])
  conv = utils.conv2d(x, W, b, 1)

  tanh = tf.nn.tanh(conv)
  s1_conv1_1 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 2

# Downsampling
layer_name = "s2_conv1_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 5, ch[0], ch[1]])
  b = utils.bias_variable([ch[1]])
  conv = utils.conv2d(s1_conv1_1, W, b, 2)

  tanh = tf.nn.tanh(conv)
  s2_conv1_1 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Regular convolution
layer_name = "s2_bottleneck1_1"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s2_conv1_1, [5, 5], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s2_bottleneck1_1 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Regular convolution
layer_name = "s2_bottleneck1_2"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s2_bottleneck1_1, [5, 5], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s2_bottleneck1_2 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 3

# Downsampling
layer_name = "s3_conv1_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 5, ch[1], ch[2]])
  b = utils.bias_variable([ch[2]])
  conv = utils.conv2d(s2_bottleneck1_2, W, b, 2)

  tanh = tf.nn.tanh(conv)
  s3_conv1_1 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Regular convolution
layer_name = "s3_bottleneck1_1"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s3_conv1_1, [5, 5], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s3_bottleneck1_1 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Regular convolution
layer_name = "s3_bottleneck1_2"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s3_bottleneck1_1, [5, 5], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s3_bottleneck1_2 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 4

# Bottleneck - Dilated convolution
layer_name = "s4_bottleneck1_1"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s3_bottleneck1_2, [5, 5], rate=2, training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_1 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Asymmetric convolution
layer_name = "s4_bottleneck1_2"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_1, [10, 1], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_2 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Asymmetric convolution
layer_name = "s4_bottleneck1_3"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_2, [1, 10], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_3 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Dilated convolution
layer_name = "s4_bottleneck1_4"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_3, [5, 5], rate=4, training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_4 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Asymmetric convolution
layer_name = "s4_bottleneck1_5"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_4, [10, 1], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_5 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Asymmetric convolution
layer_name = "s4_bottleneck1_6"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_5, [1, 10], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_6 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Dilated convolution
layer_name = "s4_bottleneck1_7"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_6, [5, 5], rate=8, training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_7 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Asymmetric convolution
layer_name = "s4_bottleneck1_8"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_7, [10, 1], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_8 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Asymmetric convolution
layer_name = "s4_bottleneck1_9"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_8, [1, 10], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_9 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Dilated convolution
layer_name = "s4_bottleneck1_10"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_9, [5, 5], rate=8, training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_10 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Asymmetric convolution
layer_name = "s4_bottleneck1_11"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_10, [10, 1], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_11 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Asymmetric convolution
layer_name = "s4_bottleneck1_12"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s4_bottleneck1_11, [1, 10], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s4_bottleneck1_12 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 5

# Upsampling
layer_name = "s5_conv1_1"
s5_shape = tf.shape(s2_conv1_1)
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 5, ch[4], ch[3]])
  b = utils.bias_variable([ch[4]])
  conv = utils.conv2d_transpose(s4_bottleneck1_8, W, b, s5_shape, 2)

  tanh = tf.nn.tanh(conv)
  s5_conv1_1 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Regular convolution
layer_name = "s5_bottleneck1_1"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s5_conv1_1, [5, 5], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s5_bottleneck1_1 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Regular convolution
layer_name = "s5_bottleneck1_2"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s5_bottleneck1_1, [5, 5], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s5_bottleneck1_2 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 6

# Upsampling
layer_name = "s6_conv1_1"
s6_shape = tf.shape(s1_conv1_1)
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 5, ch[5], ch[4]])
  b = utils.bias_variable([ch[5]])
  conv = utils.conv2d_transpose(s5_bottleneck1_2, W, b, s6_shape, 2)

  tanh = tf.nn.tanh(conv)
  s6_conv1_1 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Regular convolution
layer_name = "s6_bottleneck1_1"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s6_conv1_1, [5, 5], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s6_bottleneck1_1 = tf.nn.dropout(tanh, keep_prob)

# Bottleneck - Regular convolution
layer_name = "s6_bottleneck1_2"
with tf.name_scope(layer_name):
  b = utils.bottleneck(s6_bottleneck1_1, [5, 5], training=training, name=layer_name)
  tanh = tf.nn.tanh(b)
  s6_bottleneck1_2 = tf.nn.dropout(tanh, keep_prob)

##############################
# Section 7

# Convolution
layer_name = "s7_conv1_1"
with tf.name_scope(layer_name):
  W = utils.weight_variable([5, 5, ch[5], 1)
  b = utils.bias_variable([1])
  conv = utils.conv2d(s6_bottleneck1_2, W, b, 1)
  
  tanh = tf.nn.tanh(conv)
  s7_conv1_1 = tf.nn.dropout(tanh, keep_prob)

##############################

# Form logits and a prediction
y = tf.squeeze(s7_conv1_1, squeeze_dims=[3])
prediction = tf.cast(tf.round(tf.sigmoid(y)), tf.bool)

# Image summarries
tf.image_summary("input", x, max_images=2)
tf.image_summary("label", tf.expand_dims(tf.cast(y_, tf.float32), dim=3), max_images=2)
tf.image_summary("prediction", tf.expand_dims(tf.cast(prediction, tf.float32), dim=3), max_images=2)
