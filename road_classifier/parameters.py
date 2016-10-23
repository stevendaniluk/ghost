# All Settings for the road classifier model

run_name = "test"	# Name which everything will be saved under

# Directories
datasets = ["cityscapes", "camvid", "kitti"]
log_dir = "logs/" + run_name
model_ckpt_dir = "trained_models/" + run_name	 

# Input Data Settings
res = {"width": 64, "height": 64}   # Resolution to downsample to
train_frac = 0.8                    # Percent of dataset for training
val_frac = 0.2                      # Percent of dataset for validation

# Training Settings
max_steps = 10            # Number of steps to run trainer.
batch_size = 1            # Number of images in each training batch
dropout = 0.95            # Keep probability for training dropout.
learning_rate = 0.001     # Initial learning rate
save_model = True         # If the final model should be saved