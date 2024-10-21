from roboflow import Roboflow
import os
import shutil

rf = Roboflow(api_key="TZCi2SCltMQkbaVAtOqT")

# List of datasets to download (workspace, dataset, version)
datasets_info = [
    ("hongmo", "sea-ezx3q", 44),
    ("kim-borma", "marine-debris-vb2qm", 16),
    ("myfyp", "new-taco", 3),
    ("anas-fd5bu", "-kg", 5),
    ("trash-detection-eka7g", "aeriel-trash-detection", 5),
    ("fyp-bfx3h", "yolov8-trash-detections", 6),
    ("taco-ihjgk", "yolov8-trash-detections-kgnug", 11),
]

"""
    ("trash-ai-27vay", "ultimate-rqkdd", 1),
    ("iqram", "trashdetectionv8", 2),
    ("smart-india-hackathon-2023", "garbage_best", 1),
    # New datasets (replace VERSION with the correct version number)
    ("kim-borma", "conttttt", 1),
    ("adamson-university-rkm6w", "manila-bay", 3),
    ("davids-workspace", "garbage-detection-mnj17", 1),
    ("majasworkspace", "p_eine_klasse", 1),
    ("ukm-wcn", "ml2-wcn-ukm", 23),
    ("vit-akped", "tacodetecron-2", 1),
    ("k2k", "imported-f38bi", 4),
    ("python-project-5oizd", "trash-detection-mzljs", 7),
    ("project-mqoot", "plastic-bag-detection", 1),
    ("trash-dataset-for-oriented-bounded-box", "trash-detection-1fjjc", 14),
    ("nam-nhat", "trash-dvdrr", 5),
    ("litter-beach-detection", "beach-garbage", 2),
    ("dechets-qlhbp", "tacoo-cj6rf", 1),
    ("cscsi218", "garbageclassification-47d7t", 1),
    ("hust-xz9js", "trashbot-wahhb", 1),
    ("furqan-sayyed-veuct", "taco-ppujr", 1),
    ("abhijeet-beedikar-pbc0x", "garbage-classification-taco", 1)
"""

dest_dir = os.path.abspath("datasets")
print(f"Destination directory: {dest_dir}")

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for (workspace, dataset_name, version) in datasets_info:
    print(f"Processing {dataset_name}...")
    try:
        project = rf.workspace(workspace).project(dataset_name)
        print(f"Attempting to download version {version} of {dataset_name}")

        # Specify the download location
        download_dir = os.path.join(dest_dir, dataset_name)

        # Download the dataset to the specified location
        dataset = project.version(version).download("yolov11", location=download_dir)
        print(f"Successfully downloaded {dataset_name} to {dataset.location}")

        # List the contents of the downloaded directory
        if os.path.exists(dataset.location):
            print(f"Contents of {dataset.location}: {os.listdir(dataset.location)}")
        else:
            print(f"Warning: Downloaded directory not found at {dataset.location}")

    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
    print("-" * 50)

print("Download process completed. Please check the following directory for your files:")
print(dest_dir)








import os
import shutil
import yaml
import glob

def merge_datasets(datasets_dir, combined_dataset_dir):
    # Create directories for the combined dataset
    os.makedirs(combined_dataset_dir, exist_ok=True)
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(os.path.join(combined_dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(combined_dataset_dir, 'labels', split), exist_ok=True)

    combined_classes = []
    class_mappings = {}

    # First pass: Collect all class names and build combined_classes list
    for dataset_name in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        if not os.path.exists(data_yaml_path):
            print(f"data.yaml not found in {dataset_path}. Skipping this dataset.")
            continue

        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)

        dataset_classes = data_yaml.get('names', [])
        if not dataset_classes:
            print(f"No 'names' field found in {data_yaml_path}. Skipping this dataset.")
            continue

        # Update combined_classes and create class mappings for this dataset
        mapping = {}
        for idx, class_name in enumerate(dataset_classes):
            if class_name not in combined_classes:
                combined_classes.append(class_name)
            mapping[idx] = combined_classes.index(class_name)
        class_mappings[dataset_name] = mapping

    print(f"Combined classes: {combined_classes}")

    # Second pass: Copy images and labels, remap class IDs
    for dataset_name in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        mapping = class_mappings.get(dataset_name)
        if mapping is None:
            continue  # Skip datasets that were not included in combined_classes

        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(dataset_path, split)
            if not os.path.exists(split_dir):
                print(f"Split '{split}' not found in {dataset_name}. Skipping.")
                continue

            # Check if images and labels are under 'images' and 'labels' directories
            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'labels')
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                # Maybe images and labels are directly under split_dir
                images_dir = split_dir
                labels_dir = split_dir
                # Check if there are images and labels in split_dir
                image_files = glob.glob(os.path.join(images_dir, '*.jpg')) + \
                              glob.glob(os.path.join(images_dir, '*.jpeg')) + \
                              glob.glob(os.path.join(images_dir, '*.png')) + \
                              glob.glob(os.path.join(images_dir, '*.bmp')) + \
                              glob.glob(os.path.join(images_dir, '*.tif')) + \
                              glob.glob(os.path.join(images_dir, '*.tiff'))
                label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
                if len(image_files) == 0:
                    print(f"No images found in {images_dir}. Skipping split '{split}' in dataset '{dataset_name}'.")
                    continue
            else:
                # Images and labels are under 'images' and 'labels' directories
                image_files = glob.glob(os.path.join(images_dir, '*.*'))
                label_files = glob.glob(os.path.join(labels_dir, '*.txt'))

            for image_file in image_files:
                image_filename = os.path.basename(image_file)
                image_name, image_ext = os.path.splitext(image_filename)

                # Corresponding label file
                label_file = os.path.join(labels_dir, f"{image_name}.txt")
                if not os.path.exists(label_file):
                    print(f"Label file {label_file} not found. Creating empty label.")
                    label_lines = []
                else:
                    with open(label_file, 'r') as lf:
                        label_lines = lf.readlines()

                # Remap class IDs in label file
                remapped_labels = []
                for line in label_lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"Invalid label format in {label_file}. Skipping line.")
                        continue
                    try:
                        class_id = int(parts[0])
                        new_class_id = mapping.get(class_id)
                        if new_class_id is None:
                            print(f"Class ID {class_id} not found in mapping for {dataset_name}. Skipping line.")
                            continue
                        parts[0] = str(new_class_id)
                        remapped_labels.append(' '.join(parts))
                    except ValueError:
                        print(f"Invalid class ID in {label_file}. Skipping line.")
                        continue

                # Copy image
                dest_image_name = f"{dataset_name}_{image_filename}"
                target_split = split  # Use the same split name
                dest_image_path = os.path.join(combined_dataset_dir, 'images', target_split, dest_image_name)
                shutil.copy(image_file, dest_image_path)

                # Write remapped label
                dest_label_name = f"{dataset_name}_{image_name}.txt"
                dest_label_path = os.path.join(combined_dataset_dir, 'labels', target_split, dest_label_name)
                with open(dest_label_path, 'w') as lf:
                    lf.write('\n'.join(remapped_labels))

    # Create combined data.yaml
    combined_data = {
        'path': os.path.abspath(combined_dataset_dir),
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'valid'),  # Use 'valid' as validation set
        'names': combined_classes
    }
    with open(os.path.join(combined_dataset_dir, 'data.yaml'), 'w') as f:
        yaml.dump(combined_data, f)

    print("Dataset merging complete.")

# Usage
datasets_dir = 'datasets'  # Directory containing individual datasets
combined_dataset_dir = 'combined_dataset'  # Directory to store the combined dataset

merge_datasets(datasets_dir, combined_dataset_dir)





import yaml
import os
from collections import defaultdict

def consolidate_classes(combined_dataset_dir):
    data_yaml_path = os.path.join(combined_dataset_dir, 'data.yaml')
    
    # Load the original data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Define class mappings (adjust this based on your specific classes)
    class_mappings = {
        'Glass': ['Glass', 'glass', 'Glass bottle', 'glass bottle'],
        'Metal': ['Metal', 'metal', 'Can', 'can', 'aluminum', 'Aluminium foil', 'Pop tab', 'pop tab'],
        'Plastic': ['Plastic', 'plastic', 'PET', 'PET_Bottle', 'Plastic_ETC', 'plastic bottle', 'plastic container',
                    'Plastic bag', 'plastic bag', 'Plastic_Buoy', 'Plastic_Buoy_China', 'Bottle', 'Plastic vessels'],
        'Styrofoam': ['Styrofoam', 'Styrofoam_Box', 'Styrofoam_Buoy', 'Styrofoam_Piece', 'styrofoam', 'Styrofoam cup'],
        'Paper': ['Paper', 'paper', 'cardboard', 'cardboard boxes and cartons', 'Carton'],
        'Rope': ['Rope', 'Net'],
        'Food Waste': ['food - others', 'Decomposable'],
        'Other': ['Other litter', 'other', 'Non-decomposable', 'Non-decomposable-', 'utensils and straw', 'Straw',
                  'fabrics', 'wood', 'Cigarette', 'Wrapper', 'Lid', 'cap or lid', 'Bottle cap', 'plastic bottle cap']
    }

    # Create reverse mapping
    reverse_mapping = {}
    for new_class, old_classes in class_mappings.items():
        for old_class in old_classes:
            reverse_mapping[old_class.lower()] = new_class

    # Function to update labels in a file
    def update_labels(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                old_class = int(parts[0])
                if old_class < len(data['names']):
                    old_class_name = data['names'][old_class].lower()
                    new_class_name = reverse_mapping.get(old_class_name, 'Other')
                    new_class_id = list(class_mappings.keys()).index(new_class_name)
                    new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
        
        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    # Update all label files
    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(combined_dataset_dir, 'labels', split)
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                update_labels(os.path.join(label_dir, filename))

    # Update data.yaml with new classes
    data['names'] = list(class_mappings.keys())
    
    # Save the updated data.yaml
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f)

    print(f"Updated data.yaml saved to {data_yaml_path}")
    print(f"New classes: {data['names']}")

# Usage
datasets_dir = 'datasets'  # Directory containing individual datasets
combined_dataset_dir = 'combined_dataset'  # Directory storing the combined dataset

# First, run your existing merge_datasets function
merge_datasets(datasets_dir, combined_dataset_dir)

# Then, run the class consolidation
consolidate_classes(combined_dataset_dir)

# Now you can proceed with your YOLO training using the consolidated dataset