# Loads all data for training and testing the road classifier model
#
# Raw images are converted to greyscale, while labels are converted to boolean.
#
# Scans the dataset folders set in parameters.py for raw and labeled images.
# In each dataset folder the images must be in the following named folders:
#   -raw
#   -labels
#
# All of the raw images with labels are arranged into ordered lists, as well as 
# a randomly shuffled list. For sequential training the entire dataset is used as
# the training set, while every 1/n (n=val_frac) image is used for validation.
# For training with randomized images the dataset is split into separate parts
# according to the fractions set in the parameters.
#
# LoadTrainBatch and LoadValBatch provide the interface to fetch images for 
# training in a random order. LoadOrderedTrainBatch and LoadOrderedValBatch
# are for getting sequential images.

import os, os.path
import random
import math
import scipy.misc
import parameters as params
import numpy as np

# Pointers for the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0
test_batch_pointer = 0

valid_images = [".jpg", ".png"]

# Temporary lists for getting image names
temp_x_names = [] 
temp_y_names = []

# Full lists of image names (randomized)
x = []
y = []
prev_x = []
train_x = []
train_y= []
train_prev_x = []
val_x = []
val_y = []
val_prev_y = []

for name in params.datasets:
  print "Scanning dataset: {0}".format(name)

  # Form directories
  raw_path = os.path.join("datasets", name, "raw")
  label_path = os.path.join("datasets", name, "labels")

  # Get the name of all training images in the folder
  for raw_name in os.listdir(raw_path):
    # Make sure its an image
    ext = os.path.splitext(raw_name)[1]
    if ext.lower() not in valid_images:
      continue
    # Check for the labeled image
    label_name = os.path.splitext(raw_name)[0] + "_label" + os.path.splitext(raw_name)[1]
    if os.path.isfile(os.path.join(label_path, label_name)):
      temp_x_names.append(os.path.join(raw_path, raw_name))
      temp_y_names.append(os.path.join(label_path, label_name))
    else:
      print "No label found for {0}. Will be ignored.".format(label_name)

  # Add to image lists
  x += sorted(temp_x_names, key=str.lower)
  y += sorted(temp_y_names, key=str.lower)
  prev_x.append(None)
  prev_x += sorted(temp_x_names, key=str.lower)
  prev_x.pop()

  # Clear for next loop
  temp_x_names = []
  temp_y_names = []

# Get total image numbers to print
num_raws = len(x)
num_labels = len(y)

# Shuffle the list of images for randomized lists
if (num_raws > 0 and num_labels > 0):
  c = list(zip(x, y, prev_x))
  random.shuffle(c)
  x, y, prev_x = zip(*c)

# Split randomized images into training and validation sets
num_train_imgs = int(math.ceil(num_raws * params.train_frac))
num_val_imgs = int(math.floor(num_raws * params.val_frac))
train_x = x[:num_train_imgs]
train_y = y[:num_train_imgs]
train_prev_x = prev_x[:num_train_imgs]
val_x = x[-num_val_imgs:]
val_y = y[-num_val_imgs:]
val_prev_x = prev_x[-num_val_imgs:]

# Error checking
if num_raws != num_labels:
  print "WARNING: Number of raw and labeled images do not match. {0} raws, and {1} labels found".format(num_raws, num_labels)

print "Found {0} images with labels.\n".format(num_raws)

####################################

# Pre-processing for images (cropping, resizing, whitening, and type conversion)
def process_img(img, label):
  # Get image dimensions
  h_in, w_in, channels = np.atleast_3d(img).shape

  # Make sure the original is larger than the new size
  if (w_in < params.res["width"] or h_in < params.res["height"]):
    print "\nERROR: Original image resolution smaller than desired resolution."
    print "Original: {0}x{1}, Desired: {2}x{3}\n".format(w_in, h_in, params.res["width"], params.res["height"])
    return

  # Aspect ratios
  current_ar = float(w_in)/float(h_in);
  desired_ar = float(params.res["width"])/float(params.res["height"])

  # Default indices
  w_low = 0
  w_high = w_in - 1
  h_low = 0
  h_high = h_in - 1

  if current_ar > desired_ar:
    # Reduce image width (centred with original)
    w_new = int(round(desired_ar*h_in))
    w_low = int(math.floor((w_in - w_new)/2))
    w_high = w_new + w_low - 1
  elif current_ar < desired_ar:
    # Reduce image height (from top of image)
    h_new = int(round(w_in/desired_ar))
    h_low = h_in - h_new

  # Crop
  img = img[h_low:h_high, w_low:w_high]

  # Resize
  img = scipy.misc.imresize(img, [params.res["height"], params.res["width"]])

  # Processing differs between 3-channel raw images and labels
  if label:    
    # Convert to boolean
    img = np.round(img).astype(bool)
  else:
    # Whiten (zero mean, and unit stddev)
    mean = np.mean(img)
    stddev = np.sqrt(np.var(img))
    adjusted_stddev = max(stddev, 1.0/np.sqrt(img.size))
    img = (img - mean)/adjusted_stddev
    img = np.atleast_3d(img)
  
  return img

# Function for getting randomized training images
def LoadTrainBatch(batch_size):
  global train_batch_pointer
  x_out = []
  y_out = []
  prev_x_out = []
  for i in range(0, batch_size):
      x_out.append(process_img(scipy.misc.imread(train_x[(train_batch_pointer + i) % num_train_imgs], 'L'), label=False))
      y_out.append(process_img(scipy.misc.imread(train_y[(train_batch_pointer + i) % num_train_imgs]), label=True))
      
      if train_prev_x[(train_batch_pointer + i) % num_train_imgs] == None:
        prev_x_out.append(np.full((params.res["height"], params.res["width"], 1), 0.5))
      else:
        prev_x_out.append(process_img(scipy.misc.imread(train_prev_x[(train_batch_pointer + i) % num_train_imgs], 'L'), label=False))

  train_batch_pointer += batch_size
  return x_out, y_out, prev_x_out

# Function for getting randomized validation images
def LoadValBatch(batch_size):
  global val_batch_pointer
  x_out = []
  y_out = []
  prev_x_out = []
  for i in range(0, batch_size):
      x_out.append(process_img(scipy.misc.imread(val_x[(val_batch_pointer + i) % num_val_imgs], 'L'), label=False))
      y_out.append(process_img(scipy.misc.imread(val_y[(val_batch_pointer + i) % num_val_imgs]), label=True))

      if val_prev_x[(val_batch_pointer + i) % num_val_imgs] == None:
        prev_x_out.append(np.full((params.res["height"], params.res["width"], 1), 0.5))
      else:
        prev_x_out.append(process_img(scipy.misc.imread(val_prev_x[(val_batch_pointer + i) % num_val_imgs], 'L'), label=False))

  val_batch_pointer += batch_size
  return x_out, y_out, prev_x_out
