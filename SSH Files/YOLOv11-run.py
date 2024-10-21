from ultralytics import YOLO
import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
import torch

# Set wandb project and run names
wandb_project_name = 'Edutech'

wandb.init(
    project=wandb_project_name,
    config={
        'epochs': 5,
        'batch_size': 80,
        'learning_rate': 0.001,
        'fp16': True,  # Add this to track FP16 usage
        'compile': True,  # Add this to track compilation usage
    }
)

# Load the YOLO11 extra-large detection model
model = YOLO('yolo11x.pt')

# Enable torch.compile (available in PyTorch 2.0+)
if torch.__version__ >= "2" and wandb.config.compile:
    print("Compiling model with torch.compile()...")
    model.model = torch.compile(model.model)

# Add WandB callback for logging
add_wandb_callback(model)

# Path to the combined dataset's data.yaml
data_yaml_path = os.path.join('combined_dataset', 'data.yaml')

# Verify that data.yaml exists
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"Error: data.yaml not found at {data_yaml_path}")
else:
    print(f"Using data.yaml at {data_yaml_path}")

# Train the model with wandb logging enabled and FP16
model.train(
    data=data_yaml_path,
    epochs=5,
    imgsz=640,
    batch=80,
    workers=16,
    exist_ok=True,
    device='0',
    project=wandb_project_name,
    verbose=True,
    amp=True,  # This enables mixed precision (FP16) training
)

# Run inference using the best model
results = model.predict(
    source=os.path.join('combined_dataset', 'images/val'),
    save=True,
    project='runs/predict',
    name='yolo11x_predict',
    exist_ok=True
)



import matplotlib.pyplot as plt
import glob
import os

def visualize_predictions(result_dir):
    image_paths = glob.glob(os.path.join(result_dir, '*.jpg'))
    num_images = min(4, len(image_paths))

    plt.figure(figsize=(15, 12))
    for i, image_path in enumerate(image_paths[:num_images]):
        image = plt.imread(image_path)
        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions('runs/predict/yolo11x_predict')
