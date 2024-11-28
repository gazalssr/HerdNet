import os
import pandas as pd
import shutil

# Load the CSV file
csv_file = '/herdnet/DATASETS/CAH_no_margins_30/val/Val_binary_gt.csv'  # Update with the correct local path to your CSV file
data = pd.read_csv(csv_file)

# Define paths for the original and new folders
original_folder = '/herdnet/DATASETS/CAH_no_margins_30/val'  # Update with your original image folder path
empty_folder = '/herdnet/DATASETS/empty_folder'  # Update with desired path for empty images
non_empty_folder = '/herdnet/DATASETS/non_empty_folder'  # Update with desired path for non-empty images

# Ensure target directories exist
os.makedirs(empty_folder, exist_ok=True)
os.makedirs(non_empty_folder, exist_ok=True)

# Process the CSV and copy images to the respective folders
for _, row in data.iterrows():
    image_name = row['images']
    label = row['binary']
    
    # Define source and destination paths
    source_path = os.path.join(original_folder, image_name)
    destination_folder = non_empty_folder if label == 1 else empty_folder
    destination_path = os.path.join(destination_folder, image_name)
    
    # Copy the image if it exists
    if os.path.exists(source_path):
        shutil.copy2(source_path, destination_path)
    else:
        print(f"Image {image_name} not found in the source folder.")

print("Images have been successfully copied to the 'empty' and 'non_empty' folders.")
