# Road classifier model
#
# Full Convolutional Network based on "Fully Convolutional Networks for Semantic Segmentation"
#
# Code based off of: https://github.com/shekkizh/FCN.tensorflow
#
# Architecture:
#		-VGG16 Encoder
#		-Decoder layer 1: Convolution + ReLu with pooling and dropout 
#				- 7x7 kernel, same-padding, stide 1, 512 input channels, 4096 output channels
#		-Decoder layer 2: Convolution + ReLu with pooling and dropout 
#				- 1x1 kernel, same-padding, stride 1, 4096 input channels, 4096 output channels
#		-Decoder layer 3: Convolution + ReLu 
#				- 1x1 kernel, same-padding, stride 1, 4096 input channels, 1 output channel
#		-Decoder layer 4: Transpose convolution with skip connection from VGG layer 4 
#				- 4x4 kernel, same-padding, stride 2, ? input channels, 1 output channel
#		-Decoder layer 5: Transpose convolution with skip connection from VGG layer 3 
#				- 4x4 kernel, same-padding, stride 2, 1 input channels, 1 output channel
#		-Decoder layer 6: Transpose convolution with skip connection from VGG layer 4 
#				- 16x16 kernel, same-padding, stride 8, 1 input channels, 1 output channel

import tensorflow as tf
import scipy
import os, sys
from six.moves import urllib
import numpy as np
import scipy.io
import parameters as params

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string("model_dir", "trained_models/", "Path to vgg model mat")
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
NUM_OF_CLASSESS = 1

###############################################
# Functions originally from TensorflowUtils.py

def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var

def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def add_activation_summary(var):
    if var is not None:
        tf.histogram_summary(var.op.name + "/activation", var)
        tf.scalar_summary(var.op.name + "/sparsity", tf.nn.zero_fraction(var))

def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def get_vgg_model(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)

    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data

###############################################

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = get_variable(bias.reshape(-1), name=name + "_b")
            current = conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                add_activation_summary(current)
        elif kind == 'pool':
            current = avg_pool_2x2(current)
        net[name] = current

    return net

training = tf.placeholder(tf.bool, name='training')

# Make input and output variables
x = tf.placeholder(tf.float32, shape=[None, params.res["height"], params.res["width"], 3])
y_ = tf.placeholder(tf.bool, shape=[None, params.res["height"], params.res["width"]])
keep_prob = tf.placeholder(tf.float32)

# VGG net expects images with pixel values [0,255]
x_image = tf.mul(x, 255)

print("Setting up VGG layers.")
model_data = get_vgg_model(FLAGS.model_dir, MODEL_URL)

mean = model_data['normalization'][0][0][0]
mean_pixel = np.mean(mean, axis=(0, 1))
processed_image = x_image - mean_pixel

# Form the vgg net, and get the pooled output out of layer 5
weights = np.squeeze(model_data['layers'])
image_net = vgg_net(weights, processed_image)
conv_final_layer = image_net["conv5_3"]
pool5 = max_pool_2x2(conv_final_layer)

# Decoder layer 1: One strided convolution with dropout
W6 = weight_variable([7, 7, 512, 4096], name="W6")
b6 = bias_variable([4096], name="b6")
conv6 = conv2d_basic(pool5, W6, b6)
relu6 = tf.nn.relu(conv6, name="relu6")
if FLAGS.debug:
    add_activation_summary(relu6)
relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

# Decoder layer 2: One strided convolution with dropout
W7 = weight_variable([1, 1, 4096, 4096], name="W7")
b7 = bias_variable([4096], name="b7")
conv7 = conv2d_basic(relu_dropout6, W7, b7)
relu7 = tf.nn.relu(conv7, name="relu7")
if FLAGS.debug:
    uadd_activation_summary(relu7)
relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

# Decoder layer 3: One strided convolution
W8 = weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
b8 = bias_variable([NUM_OF_CLASSESS], name="b8")
conv8 = conv2d_basic(relu_dropout7, W8, b8)

# Decoder layer 4: Transpose convolution with skip connection from vgg layer 4
deconv_shape1 = image_net["pool4"].get_shape()
W_t1 = weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
conv_t1 = conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

# Decoder layer 5: Transpose convolution with skip connection from vgg layer 3
deconv_shape2 = image_net["pool3"].get_shape()
W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

# Decoder layer 6: Transpose convolution up to original resolution
shape = tf.shape(x_image)
deconv_shape3 = tf.pack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
W_t3 = weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
b_t3 = bias_variable([NUM_OF_CLASSESS], name="b_t3")
conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

# Form logits and a prediction
y = tf.squeeze(conv_t3, squeeze_dims=[3])
prediction = tf.cast(tf.round(tf.sigmoid(y)), tf.bool)

# Image summarries
tf.image_summary("input", x_image, max_images=1)
tf.image_summary("label", tf.expand_dims(tf.cast(y_, tf.float32), dim=3), max_images=1)
tf.image_summary("prediction", tf.expand_dims(tf.cast(prediction, tf.float32), dim=3), max_images=1)
