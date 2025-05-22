import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import torch_safe_load # Use the safe loader
from ultralytics.utils import DEFAULT_CFG_DICT, ASSETS, checks
from pathlib import Path
import pprint

# --- Configuration ---
# 1. Path to YOUR model's YAML configuration
your_model_yaml = 'yolo12n.yaml' # Make sure this path is correct

# 2. Path to the OFFICIAL .pt file you want to get hyperparameters FROM
#    Choose the official model closest in size/type to yours (e.g., yolov8n.pt for yolo11n.yaml)
weights_path = 'yolov12n.pt'

# 3. Your dataset configuration file
dataset_yaml = 'coco.yaml' # Or your actual dataset YAML path


# --- End Configuration ---


try:
    # Load the checkpoint dictionary safely
    ckpt, _ = torch_safe_load(weights_path, safe_only=False)
    if 'train_args' in ckpt and ckpt['train_args'] is not None:
        # Convert Namespace to dict if necessary
        official_train_args = vars(ckpt['train_args']) if not isinstance(ckpt['train_args'], dict) else ckpt['train_args']
        print("\n--- Official Training Arguments Loaded ---")
        # pprint.pprint(official_train_args)
    else:
        print(f"Warning: 'train_args' not found in {weights_path}. Using Ultralytics defaults.")
        official_train_args = {} # Fall back to defaults

except Exception as e:
    print(f"Error loading args from {weights_path}: {e}. Using Ultralytics defaults.")
    official_train_args = {}

# --- Prepare Arguments for Your Training ---

# Start with the loaded official arguments
training_args = official_train_args.copy()

# **Crucial:** Remove arguments that define the model/weights/run setup,
# as we want to use YOUR yaml and train from SCRATCH.
# keys_to_remove = [
#     'model',        # We specify this with YOLO(your_model_yaml)
#     'weights',      # We want random initialization (from scratch)
#     'cfg',          # We are loading config from your_model_yaml
#     'resume',       # Not resuming
#     'save_dir',     # Let Ultralytics create a new one
#     'project',      # Let Ultralytics create a new one
#     'name',         # Let Ultralytics create a new one
#     'task',         # Usually inferred, or you can set it explicitly if needed
#     'mode',         # We are calling .train() explicitly
#     # 'data' and 'epochs' MUST be provided by the user below, remove any loaded ones
#     'data',
#     'epochs',
#     # Device is often best left for auto-detection or explicit setting in train()
#     'device',
#     # You might want to remove others depending on your needs, e.g., optimizer if you want a specific one
#     # 'optimizer',
# ]
# for key in keys_to_remove:
#     training_args.pop(key, None) # Remove key if it exists, ignore if not

# **Essential:** Add/Override necessary arguments for your run
training_args['data'] = dataset_yaml
# training_args['epochs'] = num_epochs
# training_args['imgsz'] = 640 # Set explicitly if needed, otherwise uses default
# training_args['batch'] = 16 # Set explicitly if needed
# training_args['device'] = 0 # Or 'cpu', or None for auto

# Optional: Override specific hyperparameters if desired
# training_args['lr0'] = 0.005 # Example: override learning rate
# training_args['mosaic'] = 0.5 # Example: override mosaic based on your previous findings

print("\n--- Arguments Prepared for New Training ---")
pprint.pprint(training_args)


