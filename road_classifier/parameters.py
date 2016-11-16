# All Settings for the road classifier model

run_name = "test"	# Name which everything will be saved under

# Directories
datasets = ["dataset"]
log_dir = "logs/" + run_name
model_ckpt_dir = "trained_models/" + run_name	 

# Input Data Settings
res = {"width": 300, "height": 100}  # Resolution to downsample to
train_frac = 0.90                     # Percent of dataset for training
val_frac = 0.10                       # Percent of dataset for validation

# Training Settings
max_steps = 10000         # Number of steps to run trainer.
batch_size = 3            # Number of images in each (randomized) training batch
sequential = True				  # Train on ordered or random images
dropout = 0.95            # Keep probability for training dropout.
learning_rate = 1e-5      # Initial learning rate
early_stopping = True     # Implement early stopping
save_model = False        # If the final model should be saved