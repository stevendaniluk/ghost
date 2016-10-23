# Simple script for inspecting the model's performance

import scipy.misc
import numpy as np
import tensorflow as tf
import data_loader as data
import parameters as params
import model as model

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "trained_models/test/model.ckpt")

# Get images
num_imgs = 15
img, label = data.LoadValBatch(num_imgs)
#img = data.LoadTestBatch(num_imgs)

for i in range(num_imgs):
	# Get prediciton
  mask = model.prediction.eval(feed_dict={model.x: [img[i]], model.keep_prob: 1.0, model.training:False})
  mask = tf.reshape(mask, [params.res["height"], params.res["width"]]).eval()

  # Tile raw, label, and predicted images
  combo_img = np.concatenate((img[i], np.tile(np.atleast_3d(label[i]), [1,1,3]), np.tile(np.atleast_3d(mask), [1,1,3])), axis=1)
  scipy.misc.imshow(combo_img)
  