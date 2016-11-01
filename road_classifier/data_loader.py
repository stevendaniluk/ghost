# Loads all data for training and testing the road classifier model
#
# Scans the dataset folders set in parameters.py for raw, labeled, and test images.
# In each dataset folder the images must be in the following named folders:
#   -raw
#   -labels
#   -test (not required)
#
# All of the raw images with labels are randomly shuffled, then split into training
# and validation sets according to the ratios in parameters.py.
#
# LoadTrainBatch, LoadValBatch, and LoadTestBatch provide the interface to fetch 
# images for training or testing.

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

train_dataset_num = 0
val_dataset_num = 0
test_dataset_num = 0

valid_images = [".jpg", ".png"]

# Temporary lists for getting image names
x_img_names = [] 
y_img_names = []
test_img_names = []

# Full lists of image names
train_x = []
train_y = []
val_x = []
val_y = []
test = []

# List of sizes for each dataset
train_set_sizes = []
val_set_sizes = []
test_set_sizes = []

for name in params.datasets:
  print "Scanning dataset: {0}".format(name)

  # Form directories
  raw_path = os.path.join("datasets", name, "raw")
  label_path = os.path.join("datasets", name, "labels")
  test_path = os.path.join("datasets", name, "test")

  # Get the name of all training images in the folder
  for raw_name in os.listdir(raw_path):
    # Make sure its an image
    ext = os.path.splitext(raw_name)[1]
    if ext.lower() not in valid_images:
      continue
    # Check for the labeled image
    label_name = os.path.splitext(raw_name)[0] + "_label" + os.path.splitext(raw_name)[1]
    if os.path.isfile(os.path.join(label_path, label_name)):
      x_img_names.append(os.path.join(raw_path, raw_name))
      y_img_names.append(os.path.join(label_path, label_name))
    else:
      print "No label found for {0}. Will be ignored.".format(label_name)

  # Find split size for this set
  set_num_raws = len(x_img_names)
  set_num_train_imgs = int(math.ceil(set_num_raws * params.train_frac))
  set_num_val_imgs = int(math.floor(set_num_raws * params.val_frac))

  # Record set sizes
  train_set_sizes.append(set_num_train_imgs)
  val_set_sizes.append(set_num_val_imgs)

  # Add to image lists
  train_x += x_img_names[:set_num_train_imgs]
  train_y += y_img_names[:set_num_train_imgs]
  val_x += x_img_names[-set_num_val_imgs:]
  val_y += y_img_names[-set_num_val_imgs:]

  # Clear for next loop
  x_img_names = []
  y_img_names = []

  # Get the name of all test images in the folder
  if os.path.exists(test_path):
    for name in os.listdir(test_path):
      # Make sure its an image
      ext = os.path.splitext(name)[1]
      if ext.lower() not in valid_images:
        continue
      # Check for the labeled image
      test_img_names.append(os.path.join(test_path, name))

  if len(test_img_names) > 0:
    # Record the set sizes
    test_set_sizes.append(len(test_img_names))
    # Add to image list
    test += test_img_names
    # Clear for next loop
    test_img_names = []

# Get total image numbers to print
num_raws = len(train_x) + len(val_x)
num_labels = len(train_y) + len(val_y)
num_test_imgs = len(test)
num_train_imgs = len(train_x)
num_val_imgs = len(val_x)

# Error checking
if num_raws != num_labels:
  print "WARNING: Number of raw and labeled images do not match. {0} raws, and {1} labels found".format(num_raws, num_labels)

print "Found {0} images with labels, and {1} unlabeled test images.".format(num_raws, num_test_imgs)

print "Split into {0} training images, and {1} validation images.\n".format(num_train_imgs, num_val_imgs)

# Pre-processing for images (cropping, resizing, whitening, and type conversion)
def process_img(img):
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

  # Map to [0,1]
  img = img/255.0

  # Processing differs between 3-channel raw images and labels
  if channels == 3:
    # Crop
    img = img[h_low:h_high, w_low:w_high, :]
    # Whiten (zero mean, and unit stddev)
    mean = np.mean(img)
    stddev = np.sqrt(np.var(img))
    adjusted_stddev = max(stddev, 1.0/np.sqrt(img.size))
    img = (img - mean)/adjusted_stddev
  else:
    # Crop
    img = img[h_low:h_high, w_low:w_high]
    # Convert to boolean
    img = np.round(img).astype(bool)

  # Resize
  img = scipy.misc.imresize(img, [params.res["height"], params.res["width"]])
  return img

# Function for getting training images
def LoadTrainBatch():
  global train_batch_pointer
  global train_dataset_num

  x_out = []
  y_out = []
  x_out.append(process_img(scipy.misc.imread(train_x[train_batch_pointer % num_train_imgs])))
  y_out.append(process_img(scipy.misc.imread(train_y[train_batch_pointer % num_train_imgs])))
  
  train_batch_pointer += 1

  # Reset the dataset number
  if train_batch_pointer % train_set_sizes[train_dataset_num] == 0:
    train_dataset_num = (train_dataset_num + 1) % len(train_set_sizes)

  return x_out, y_out

# Function for getting validation images
def LoadValBatch():
  global val_batch_pointer
  global val_dataset_num

  x_out = []
  y_out = []
  
  x_out.append(process_img(scipy.misc.imread(val_x[val_batch_pointer % num_val_imgs])))
  y_out.append(process_img(scipy.misc.imread(val_y[val_batch_pointer % num_val_imgs])))

  val_batch_pointer += 1

  # Reset the dataset number
  if val_batch_pointer % val_set_sizes[val_dataset_num] == 0:
    val_dataset_num = (val_dataset_num + 1) % len(val_set_sizes)

  return x_out, y_out

# Function for getting testing images
def LoadTestBatch():
  global test_batch_pointer
  global test_dataset_num

  x_out = []
  x_out.append(process_img(scipy.misc.imread(test[test_batch_pointer % num_test_imgs])))
      
  test_batch_pointer += 1

  # Reset the dataset number
  if test_batch_pointer % test_set_sizes[test_dataset_num] == 0:
    test_dataset_num = (test_dataset_num + 1) % len(test_set_sizes)

  return x_out
