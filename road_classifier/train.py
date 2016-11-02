# Trains the road classifier model
#
# Every step the training loss is outputted and summaries are recorded.
# Accuracy on the validation set is checked every 20 steps, and Metadata
# is recorded every 100 steps. Once training is complete, the model is saved.
#
# Summaries are saved for the following data:
#   -Cross entropy
#   -Prediction accuracy
#   -Confusion matrix (TP, TN, FP, and FN)
#   -IoU (Intersection over union)

import os, os.path
import scipy.misc
import numpy as np
import tensorflow as tf
import data_loader as data
import model as model
import parameters as params

# Check for log data
if tf.gfile.Exists(params.log_dir):
  tf.gfile.DeleteRecursively(params.log_dir)
tf.gfile.MakeDirs(params.log_dir)
tf.gfile.MakeDirs(params.log_dir + "/images")

sess = tf.InteractiveSession()

# Use weighted cross entropy as the loss function
with tf.name_scope('cross_entropy'):
  num_positives = tf.maximum(tf.reduce_sum(tf.cast(model.y_, tf.int32)), 1)
  num_negatives = tf.sub(tf.size(model.y_), num_positives)
  class_ratio = tf.cast(num_negatives, tf.float32)/tf.cast(num_positives, tf.float32)
  diff = tf.nn.weighted_cross_entropy_with_logits(tf.clip_by_value(model.y, 1e-10, 1.0), tf.cast(model.y_, tf.float32), class_ratio)
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
  tf.scalar_summary('cross entropy', cross_entropy)

# Add optimizer to the graph to minimize cross entropy
with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(model.prediction, model.y_)
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary('accuracy', accuracy)

with tf.name_scope('prediction_stats') as scope:
  true_pos = tf.reduce_mean(tf.cast(tf.logical_and(model.prediction, model.y_), tf.float32))
  true_neg = tf.reduce_mean(tf.cast(tf.logical_and(tf.logical_not(model.prediction), tf.logical_not(model.y_)), tf.float32))
  false_pos = tf.reduce_mean(tf.cast(tf.logical_and(model.prediction, tf.logical_not(model.y_)), tf.float32))
  false_neg = tf.reduce_mean(tf.cast(tf.logical_and(tf.logical_not(model.prediction), model.y_), tf.float32))
  IoU = true_pos/(true_pos + false_pos + false_neg)

  tf.scalar_summary(scope + 'true_pos', true_pos)
  tf.scalar_summary(scope + 'true_neg', true_neg)
  tf.scalar_summary(scope + 'false_pos', false_pos)
  tf.scalar_summary(scope + 'false_neg', false_neg)
  tf.scalar_summary(scope + 'IoU', IoU)
  
# Merge all the summaries and write them out
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(params.log_dir + '/train', sess.graph)

print "Initializing variables."
tf.initialize_all_variables().run()
saver = tf.train.Saver()

# Initialize previous prediction
prev_prediction = np.random.rand(1, params.res["height"], params.res["width"]).tolist()
prev_label = np.random.rand(1, params.res["height"], params.res["width"]).tolist()

# Set initial dataset
train_dataset_num = data.train_dataset_num

# Train the model, and also write summaries.
# Every 100th step save meta data
print "Beginning training (max {0} steps).".format(params.max_steps)
for i in range(params.max_steps):
  # Get training data
  xs, ys = data.LoadTrainBatch()

  # If datasets have changed, the previous predicition must be randomized
  if train_dataset_num != data.train_dataset_num:
    train_dataset_num = data.train_dataset_num
    prev_prediction = np.random.rand(1, params.res["height"], params.res["width"]).tolist()
    prev_label = np.random.rand(1, params.res["height"], params.res["width"]).tolist()
    print "New dataset."

  if i % 100 == 99:
    # Record execution stats 
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    feed_dict = {model.x: xs, model.y_: ys, model.prev_y:prev_prediction, model.keep_prob: 1.0, model.training:True}
    summary, loss, _ = sess.run([merged, cross_entropy, train_step], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
    
    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    train_writer.add_summary(summary, i)
    print "Saved metadata for step {0}.".format(i)
  else:
    # Record a summary
    feed_dict = {model.x:xs, model.y_:ys, model.prev_y:prev_prediction, model.keep_prob:params.dropout, model.training:True}
    summary, loss, _ = sess.run([merged, cross_entropy, train_step], feed_dict=feed_dict)
    train_writer.add_summary(summary, i)

  print "Training step {0} loss:{1:.3f}".format(i, loss)

  # Get the prediction for next step
  feed_dict = {model.x:xs, model.y_:ys, model.prev_y:prev_prediction, model.keep_prob:1.0, model.training:False}
  prev_prediction = sess.run(model.prediction, feed_dict=feed_dict)
  
  # save the label for the next step
  prev_label = ys

  # Save sample predictions
  if i % 20 == 19:
    scipy.misc.imsave((params.log_dir + "/images/step_" + str(i) + "_raw.png"), xs[0])
    scipy.misc.imsave((params.log_dir + "/images/step_" + str(i) + "._label.png"), ys[0])
    scipy.misc.imsave((params.log_dir + "/images/step_" + str(i) + "._pred.png"), prev_prediction[0])
    print "Saved sample images."

train_writer.close()
print "Training complete." 

if (params.save_model):
  # Save the model
  if not os.path.exists(params.model_ckpt_dir):
    os.makedirs(params.model_ckpt_dir)
  checkpoint_path = os.path.join(params.model_ckpt_dir, "model.ckpt")
  filename = saver.save(sess, checkpoint_path)
  print "Model saved in file: {0}.".format(filename)
