# Trains the road classifier model
#
# Training summaries are recorded every step, accuracy on the validations set 
# is measured every 5 steps and a summary recorded, while Metadata is recorded
# every 20 steps.

import os, os.path
import tensorflow as tf
import model
import data_loader as data
import parameters as params

# Check for summaries data
if tf.gfile.Exists(params.summaries_dir):
  tf.gfile.DeleteRecursively(params.summaries_dir)
tf.gfile.MakeDirs(params.summaries_dir)

sess = tf.InteractiveSession()

# Use cross entropy as the loss function
with tf.name_scope('cross_entropy'):
  diff = tf.nn.sigmoid_cross_entropy_with_logits(model.y, tf.cast(model.y_, tf.float32))
  with tf.name_scope('total'):
    cross_entropy = tf.reduce_mean(diff)
  tf.scalar_summary('cross entropy', cross_entropy)

# Add optimizer to the graph to minimize cross entropy
with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.cast(tf.round(model.y), tf.bool), model.y_)
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary('accuracy', accuracy)

# Merge all the summaries and write them out
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(params.summaries_dir + '/train', sess.graph)
val_writer = tf.train.SummaryWriter(params.summaries_dir + '/val')

print "Initializing variables."
tf.initialize_all_variables().run()
saver = tf.train.Saver()

# Train the model, and also write summaries.
# Every 5th step, measure validation-set accuracy, and write validation summaries
# All other steps, run train_step on training data, and add training summaries
print "Beginning training (max {0} steps).".format(params.max_steps)
for i in range(params.max_steps):
  print "Training step: {0}".format(i)
  # Get training data
  xs, ys = data.LoadTrainBatch(params.batch_size)

  if i % 20 == 19:
    # Record execution stats 
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step],
                          feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0},
                          options=run_options,
                          run_metadata=run_metadata)
    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    train_writer.add_summary(summary, i)
    print "Adding metadata for run {0}.".format(i)
  else:
    # Record a summary
    summary, _ = sess.run([merged, train_step], feed_dict={model.x: xs, model.y_: ys, model.keep_prob:params.dropout})
    train_writer.add_summary(summary, i)

  if (i % 5 == 0): 
    # Measure validation set accuracy
    xs, ys = data.LoadValBatch(params.batch_size)
    summary, acc = sess.run([merged, accuracy], feed_dict={model.x: xs, model.y_: ys, model.keep_prob:params.dropout})
    val_writer.add_summary(summary, i)
    print "Validation accuracy at step {0}: {1:.3f}".format(i, acc)

train_writer.close()
val_writer.close()
print "Training complete." 

if (params.save_model):
  # Save the model
  if not os.path.exists(params.model_ckpt_dir):
    os.makedirs(params.model_ckpt_dir)
  checkpoint_path = os.path.join(params.model_ckpt_dir, "model.ckpt")
  filename = saver.save(sess, checkpoint_path)
  print "Model saved in file: {0}.".format(filename)
