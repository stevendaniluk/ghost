# Checks datasets for all black labels, then deletes them and their raw image

import os, os.path
import random
import math
import scipy.misc
import numpy as np

datasets = ["cityscapes", "camvid", "kitti"]
valid_images = [".jpg", ".png"]
label_names = []
raw_names = []

for name in datasets:
  print "Scanning dataset: {0}".format(name)

  # Form directory
  raw_path = os.path.join(name, "raw")
  label_path = os.path.join(name, "labels")

  # Get the name of all training images in the folder
  for raw_name in os.listdir(raw_path):
    # Make sure its an image
    ext = os.path.splitext(raw_name)[1]
    if ext.lower() not in valid_images:
      continue

    # Check for the labeled image
    label_name = os.path.splitext(raw_name)[0] + "_label" + os.path.splitext(raw_name)[1]
    if os.path.isfile(os.path.join(label_path, label_name)):
      raw_names.append(os.path.join(raw_path, raw_name))
      label_names.append(os.path.join(label_path, label_name))

num_labels = len(label_names)
print "Number of labels: {0}".format(num_labels)

# Check all images
print "Checking images."
bad_labels = []
bad_raws = []
for i in range(num_labels):
  if i % 500 == 0:
    print "At image {0}".format(i)

  img = scipy.misc.imread(label_names[i])
  num_pos = np.sum(img)
  if num_pos == 0:
    bad_labels.append(label_names[i])
    bad_raws.append(raw_names[i])

num_bad_imgs = len(bad_labels)
print "Done, found {0} bad images to be deleted.".format(num_bad_imgs)

if num_bad_imgs > 0:
  # Sort alphabetically
  bad_labels = sorted(bad_labels, key=str.lower)
  bad_raws = sorted(bad_raws, key=str.lower)

  print "List of bad images that will be deleted:"
  for i in range(len(bad_labels)):
    print bad_labels[i]
    os.remove(bad_labels[i])
    os.remove(bad_raws[i])
