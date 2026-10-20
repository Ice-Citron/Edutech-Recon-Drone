from ultralytics import YOLO
import matplotlib.pyplot as plt
import glob
import os

def visualize_predictions(result_dir):
    """Visualize up to four prediction results."""
    image_paths = glob.glob(os.path.join(result_dir, '*.jpg'))
    num_images = min(4, len(image_paths))

    if num_images == 0:
        print("No images found for visualization.")
        return

    plt.figure(figsize=(15, 12))
    for i, image_path in enumerate(image_paths[:num_images]):
        image = plt.imread(image_path)
        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def run_inference(checkpoint_path, inference_source='combined_dataset/images/valid', inference_name='yolo_infer_last'):
    """Run inference using the saved checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint '{checkpoint_path}' does not exist. Please ensure the path is correct.")
        return

    print(f"Loading the model from '{checkpoint_path}'...")
    try:
        # Load the model with the saved weights
        model = YOLO(checkpoint_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Verify inference source
    if not os.path.exists(inference_source):
        print(f"Inference source '{inference_source}' does not exist. Please provide a valid path.")
        return

    print(f"Running inference on '{inference_source}'...")
    try:
        results = model.predict(
            source=inference_source,
            save=True,
            project='runs/predict',
            name=inference_name,
            exist_ok=True
        )
        print("Inference completed.")
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # Visualize predictions
    visualize_predictions(os.path.join('runs', 'predict', inference_name))

def main():
    # Define the path to the checkpoint
    checkpoint_path = 'EduTech-YOLOv11/weights/best.pt'  # final model, from the HF clone

    # Run inference
    run_inference(checkpoint_path, inference_name='yolo_infer_last')

if __name__ == "__main__":
    main()
