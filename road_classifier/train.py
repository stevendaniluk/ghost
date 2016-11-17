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

# Make model directory
if (params.save_model or params.early_stopping):
  if not os.path.exists(params.model_ckpt_dir):
    os.makedirs(params.model_ckpt_dir)

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
val_writer = tf.train.SummaryWriter(params.log_dir + '/val', sess.graph)

print "Initializing variables."
tf.initialize_all_variables().run()
saver = tf.train.Saver()

if (params.sequential):
  # Initialize previous prediction
  prev_prediction = np.full((1, params.res["height"], params.res["width"]), 0.0)

  # Set initial dataset
  train_dataset_num = data.train_dataset_num

# Train the model, write summaries, and check accuracy on the entire validations set
# Sample predictions and metadata will be periodically saved.
best_val_acc = 0
best_val_acc_step = 0
print "Beginning training (max {0} steps).".format(params.max_steps)
for i in range(params.max_steps):

  # Get randomized or ordered data
  if (params.sequential):
    # Get training data
    xs, ys = data.LoadOrderedTrainBatch()
    # If datasets have changed, the previous predicition must be reset
    if train_dataset_num != data.train_dataset_num:
      train_dataset_num = data.train_dataset_num
      prev_prediction = np.full((1, params.res["height"], params.res["width"]), 0.0)
      print "Next dataset started."

    feed_dict = {model.x:xs, model.y_:ys, model.prev_y:prev_prediction, model.keep_prob:params.dropout, model.training:True}
  else:
    xs, ys = data.LoadTrainBatch(params.batch_size)
    feed_dict = {model.x:xs, model.y_:ys, model.keep_prob:params.dropout, model.training:True}

  # Train operation
  if i % 500 == 499:
    # Record execution stats 
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    summary, loss, _ = sess.run([merged, cross_entropy, train_step], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
    
    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    train_writer.add_summary(summary, i)
    print "Saved metadata for step {0}.".format(i)
  else:
    # Record a summary
    summary, loss, _ = sess.run([merged, cross_entropy, train_step], feed_dict=feed_dict)
    train_writer.add_summary(summary, i)

  if i % 10 == 0:
    print "Training step {0} loss:{1:.3f}".format(i, loss)
  
  # Measure validation set accuracy (over entire set)
  if (i % 100 == 0):
    acc_count = 0

    # Loop through each image in the validation set.
    # Write a summary for the last image, and run one extra time and to rotate through
    # the dataset so the summary isn't always on the same image
    for j in range(data.num_val_imgs + 1):
      if (params.sequential):
        xs, ys, prev_ys = data.LoadOrderedValBatch()
        feed_dict = {model.x:xs, model.y_:ys, model.prev_y:prev_ys, model.keep_prob:1.0, model.training:False}
      else:
        xs, ys = data.LoadValBatch(1)
        feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0, model.training:False}

      if j == data.num_val_imgs:
        summary = sess.run(merged, feed_dict=feed_dict)
      else:
        acc_count += sess.run(accuracy, feed_dict=feed_dict)

    acc = acc_count/data.num_val_imgs
    print "Validation set accuracy: {0:.3f}".format(acc)
    val_writer.add_summary(summary, i)

    # Save model when it has improved
    if params.early_stopping and acc > best_val_acc:
      best_val_acc = acc
      best_val_acc_step = i

      checkpoint_path = os.path.join(params.model_ckpt_dir, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
      print "Model saved in file: {0}.".format(filename)

  # Save sample predictions
  if i % 100 == 0:
    if (params.sequential):
      feed_dict = {model.x:xs, model.y_:ys, model.prev_y:prev_prediction, model.keep_prob:1.0, model.training:False}
    else:
      feed_dict = {model.x:xs, model.y_:ys, model.keep_prob:1.0, model.training:False}
    
    prediction = sess.run(model.prediction, feed_dict=feed_dict)

    scipy.misc.imsave((params.log_dir + "/images/step_" + str(i) + "_raw.png"), np.squeeze(xs[0]))
    scipy.misc.imsave((params.log_dir + "/images/step_" + str(i) + "._label.png"), ys[0])
    scipy.misc.imsave((params.log_dir + "/images/step_" + str(i) + "._pred.png"), prediction[0])
    print "Saved sample images."

  # Early stopping
  if params.early_stopping and (i - best_val_acc_step) > 1000:
    print "Stopping at step {0}.".format(i)
    break

  # When training sequentially, get the prediction and label for the next step
  if (params.sequential):
    # Get the prediction and label for the next step
    feed_dict = {model.x:xs, model.y_:ys, model.prev_y:prev_prediction, model.keep_prob:1.0, model.training:False}
    prev_prediction = tf.sigmoid(sess.run(model.y, feed_dict=feed_dict)).eval()

train_writer.close()
val_writer.close()
print "Training complete." 

# Save the final model (if desired)
if (params.save_model and not params.early_stopping):
  checkpoint_path = os.path.join(params.model_ckpt_dir, "model.ckpt")
  filename = saver.save(sess, checkpoint_path)
  print "Model saved in file: {0}.".format(filename)
