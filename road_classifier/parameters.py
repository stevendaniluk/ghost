# All Settings for the road classifier model

# Directories
datasets = ["camvid"]   # Folder(s) containing datasets
summaries_dir = "logs"										  # Directory to save logs in
model_ckpt_dir = "trained_models"					  # Directory to save model checkpoints in

# Input Data Settings
res = {"width": 120, "height": 80}   # Resolution to downsample to
train_frac = 0.6                     # Percent of dataset for training
val_frac = 0.2                       # Percent of dataset for validation

# Training Settings
max_steps = 11 							# Number of steps to run trainer.
batch_size = 5 							# Number of images in each training batch
dropout = 0.9               # Keep probability for training dropout.
learning_rate = 0.001       # Initial learning rate
save_model = False					# If the final model should be saved