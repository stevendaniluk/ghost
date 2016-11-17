# Tests the road classifier model
#
# Given a model, and the dataset defined in parameters, it performs inference on
# every image in the test set. The LoadOrderedTrainBatch function is used, so in 
# parameters train_frac should be set to 1.0.
#
# The model being tested must be set in the import call and model_name.
#
# The following metrics are evaluated and displayed
#  -Accuracy
#  -True Positives
#  -True Negatives
#  -False Positives
#  -False Negatives
#  -Intersection of Union
#  -Inference Time (averaged of 10 inferences)

import numpy as np
import tensorflow as tf
import data_loader as data
import parameters as params
import model as model
import time

# Set the name of the model filename to be tested
model_filename = "model"

sess = tf.InteractiveSession()

# Load the model
saver = tf.train.Saver()
saver.restore(sess, "trained_models/" + model_filename + "/model.ckpt")
print "Loaded model: " + model_filename + "."

# Ops to be performed during testing
with tf.name_scope('prediction_stats'):
	accuracy = tf.reduce_mean(tf.cast(tf.equal(model.prediction, model.y_), tf.float32))
	true_pos = tf.reduce_mean(tf.cast(tf.logical_and(model.prediction, model.y_), tf.float32))
	true_neg = tf.reduce_mean(tf.cast(tf.logical_and(tf.logical_not(model.prediction), tf.logical_not(model.y_)), tf.float32))
	false_pos = tf.reduce_mean(tf.cast(tf.logical_and(model.prediction, tf.logical_not(model.y_)), tf.float32))
	false_neg = tf.reduce_mean(tf.cast(tf.logical_and(tf.logical_not(model.prediction), model.y_), tf.float32))
	IoU = true_pos/(true_pos + false_pos + false_neg)

# Initialize previous prediction
prev_pred = np.full((1, params.res["height"], params.res["width"]), 0.0)

# Performance stats to be filled
acc_stat = []
true_pos_stat = []
true_neg_stat = []
false_pos_stat = []
false_neg_stat = []
IoU_stat = []

# Set initial dataset
dataset_num = data.train_dataset_num

# Cycle through entire test set
n = data.num_raws
print "Beginning testing on {0} images.\n".format(n)
for i in range(n):
	x, y = data.LoadOrderedTrainBatch()
	# If datasets have changed, the previous predicition must be reset
	if dataset_num != data.train_dataset_num:
		dataset_num = data.train_dataset_num
		prev_pred = np.full((1, params.res["height"], params.res["width"]), 0.0)

	feed_dict = {model.x:x, model.y_:y, model.prev_y:prev_pred, model.keep_prob:1.0, model.training:False}
	prev_pred, acc_i, true_pos_i, true_neg_i, false_pos_i, false_neg_i, IoU_i = sess.run([model.prediction, accuracy, true_pos, true_neg, false_pos, false_neg, IoU], feed_dict=feed_dict)

	# Log metrics
	acc_stat.append(acc_i)
	true_pos_stat.append(true_pos_i)
	true_neg_stat.append(true_neg_i)
	false_pos_stat.append(false_pos_i)
	false_neg_stat.append(false_neg_i)
	IoU_stat.append(IoU_i)

# Perform inference 10 times to get an average computation time (done separately so
# the time to compute all the metrics is not included)
prev_pred = np.random.rand(1, params.res["height"], params.res["width"]).tolist()
inf_time = []
for i in range(10):
	x, y = data.LoadOrderedTrainBatch()
	feed_dict = {model.x:x, model.prev_y:prev_pred, model.keep_prob:1.0, model.training:False}

	start_time = time.time()
	prev_pred = sess.run(model.prediction, feed_dict=feed_dict)
	inf_time.append(time.time() - start_time)

# Display all the results
print "------- Results --------"
print "Accuracy:        {0:5.2f}%".format(100*np.sum(acc_stat)/n)
print "True Positives:  {0:5.2f}%".format(100*np.sum(true_pos_stat)/n)
print "True Negatives:  {0:5.2f}%".format(100*np.sum(true_neg_stat)/n)
print "False Positives: {0:5.2f}%".format(100*np.sum(false_pos_stat)/n)
print "False Negatives: {0:5.2f}%".format(100*np.sum(false_neg_stat)/n)
print "IoU:             {0:5.2f}%".format(100*np.sum(IoU_stat)/n)
print "Inference time:  {0:5.3f}s".format(np.sum(inf_time)/10)
print "------------------------\n"
