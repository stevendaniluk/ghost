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

valid_images = [".jpg", ".png"]

x_img_names = [] 
y_img_names = []
test_img_names = []

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

  # Get the name of all test images in the folder
  if os.path.exists(test_path):
    for name in os.listdir(test_path):
      # Make sure its an image
      ext = os.path.splitext(name)[1]
      if ext.lower() not in valid_images:
        continue
      # Check for the labeled image
      test_img_names.append(os.path.join(test_path, name))

num_raws = len(x_img_names)
num_labels = len(y_img_names)
num_test_imgs = len(test_img_names)

# Error checking
if num_raws != num_labels:
  print "WARNING: Number of raw and labeled images do not match. {0} raws, and {1} labels found".format(num_raws, num_labels)

print "Found {0} images with labels, and {1} unlabeled test images.".format(num_raws, num_test_imgs)

# Shuffle the list of images
c = list(zip(x_img_names, y_img_names))
random.shuffle(c)
x_img_names, y_img_names = zip(*c)

# Split into training and validation sets
num_train_imgs = int(math.ceil(num_raws * params.train_frac))
num_val_imgs = int(math.floor(num_raws * params.val_frac))

train_x = x_img_names[:num_train_imgs]
train_y = y_img_names[:num_train_imgs]
val_x = x_img_names[-num_val_imgs:]
val_y = y_img_names[-num_val_imgs:]

print "Split into {0} training images, and {1} validation images.\n".format(num_train_imgs, num_val_imgs)

# Function for cropping and resizing an image
def resize_img(img):
  # Get image dimensions
  h_in, w_in, dims = np.atleast_3d(img).shape
  is_rgb_img = (dims == 3)

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

  # Crop with indices
  if is_rgb_img:
    img = img[h_low:h_high, w_low:w_high, :]
  else:
    img = img[h_low:h_high, w_low:w_high]

  # Resize
  img = scipy.misc.imresize(img, [params.res["height"], params.res["width"]])
  return img

# Function for getting training images
def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(resize_img(scipy.misc.imread(train_x[(train_batch_pointer + i) % num_train_imgs])) / 255.0)
        y_out.append(np.round(resize_img(scipy.misc.imread(train_y[(train_batch_pointer + i) % num_train_imgs])) / 255.0).astype(bool))
        
    train_batch_pointer += batch_size
    return x_out, y_out

# Function for getting validation images
def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(resize_img(scipy.misc.imread(val_x[(val_batch_pointer + i) % num_val_imgs])) / 255.0)
        y_out.append(np.round(resize_img(scipy.misc.imread(val_y[(val_batch_pointer + i) % num_val_imgs])) / 255.0).astype(bool))

    val_batch_pointer += batch_size
    return x_out, y_out

# Function for getting testing images
def LoadTestBatch(batch_size):
    global test_batch_pointer
    x_out = []
    for i in range(0, batch_size):
        x_out.append(resize_img(scipy.misc.imread(test_img_names[(test_batch_pointer + i) % num_test_imgs])) / 255.0)
        
    test_batch_pointer += batch_size
    return x_out
