import os

# Replace 'path/to/your/folder' with the actual path to your folder
folder_path = '/herdnet/DATASETS/TEST_patches_no_margin_5'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Count the number of files (excluding directories)
num_files = len([f for f in files if os.path.isfile(os.path.join(folder_path, f))])

print(f"There are {num_files} files in the folder '{folder_path}'.")
import os
import shutil

# Specify the source folder where 'gt.csv' might exist
source_folder = '/herdnet/DATASETS/TEST_patches_no_margin/'  # Replace with the actual path to your folder

# Specify the destination folder where you want to copy 'gt.csv'
destination_folder = '/herdnet/DATASETS'

# Construct the full path to 'gt.csv' in the source folder
source_file = os.path.join(source_folder, 'TEST_gt_no_margin.csv')

# Check if 'gt.csv' exists in the source folder
if os.path.exists(source_file):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Construct the full path to the destination file
    destination_file = os.path.join(destination_folder, '/herdnet/DATASETSTEST_gt_no_margin.csv')
    
    # Copy 'gt.csv' to the destination folder
    shutil.copy(source_file, destination_file)
    print(f"'gt.csv' has been copied to '{destination_folder}'.")
else:
    print(f"'gt.csv' does not exist in '{source_folder}'.")