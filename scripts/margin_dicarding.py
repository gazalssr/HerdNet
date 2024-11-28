import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import shutil
def load_image_patch(file_path, grayscale=False):
    """
    Load an image patch. Optionally, convert to grayscale for margin detection.

    Parameters:
    file_path (str): Path to the image file.
    grayscale (bool): Whether to load the image in grayscale.

    Returns:
    numpy array: The image patch as a numpy array.
    """
    try:
        image = Image.open(file_path)
        if grayscale:
            image = image.convert('L')  # Convert to grayscale if specified
        patch_array = np.array(image)
    except UnidentifiedImageError:
        print(f"Warning: PIL couldn't read {file_path}. Trying with OpenCV.")
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: OpenCV also couldn't read {file_path}. Skipping.")
            return None
        patch_array = image
    return patch_array

def is_margin_patch(patch, min_connected_area=0.1):
    """
    Check if a patch contains large contiguous regions of pure white or pure black pixels.

    Parameters:
    patch (numpy array): The image patch.
    min_connected_area (float): Minimum size of connected component (relative to patch size) to consider.

    Returns:
    bool: True if the patch contains a large contiguous region of pure white or black pixels.
    """
    # Define minimum area threshold for connected components
    min_area = min_connected_area * patch.size  # Convert to number of pixels

    # Detect large contiguous white regions (pure white pixels, value 255)
    binary_patch_white = (patch == 255).astype(np.uint8)  # Pure white only
    num_labels_white, labels_white = cv2.connectedComponents(binary_patch_white)
    for i in range(1, num_labels_white):  # Ignore the background component
        if np.sum(labels_white == i) > min_area:
            return True  # Large contiguous white area found

    # Detect large contiguous black regions (pure black pixels, value 0)
    binary_patch_black = (patch == 0).astype(np.uint8)  # Pure black only
    num_labels_black, labels_black = cv2.connectedComponents(binary_patch_black)
    for i in range(1, num_labels_black):  # Ignore the background component
        if np.sum(labels_black == i) > min_area:
            return True  # Large contiguous black area found

    return False  # No large contiguous white or black areas found

def copy_no_margin_patches_and_update_gt(patch_dir, gt_file_path, margin_dir, no_margin_dir, min_connected_area=0.1):
    """
    Copy patches without large contiguous regions of pure white or black pixels to a new folder,
    and update the GT file to include only those patches.

    Parameters:
    patch_dir (str): Directory containing the original patches.
    gt_file_path (str): Path to the GT file.
    margin_dir (str): Directory to save the margin patches.
    no_margin_dir (str): Directory to save patches without margins.
    min_connected_area (float): Minimum size of connected component (relative to patch size) to consider.
    """
    if not os.path.exists(margin_dir):
        os.makedirs(margin_dir)
    
    # Ensure no_margin_dir exists
    if not os.path.exists(no_margin_dir):
        os.makedirs(no_margin_dir)
    
    gt_df = pd.read_csv(gt_file_path)
    patches_with_margins = []

    patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir) if f.endswith('.jpg')]
    for patch_file in patch_files:
        patch_id = os.path.basename(patch_file)

        # Retrieve the label for the patch from the GT file
        label = gt_df.loc[gt_df['images'] == patch_id, 'binary'].values[0] if patch_id in gt_df['images'].values else None

        # Load the patch in grayscale for margin detection
        patch_gray = load_image_patch(patch_file, grayscale=True)

        # Skip files that couldn't be loaded in grayscale
        if patch_gray is None:
            print(f"Skipping {patch_file} because it couldn't be loaded.")
            continue

        # Check if patch has a large contiguous margin
        has_margin = is_margin_patch(patch_gray, min_connected_area)

        # If label is 1 (object present), always copy to no_margin_dir
        if label == 1 or not has_margin:
            # Load the original (color) patch to save in no_margin_dir
            patch_color = load_image_patch(patch_file, grayscale=False)
            if patch_color is not None:
                # Copy patches without margins or with objects to the no_margin_dir
                no_margin_save_path = os.path.join(no_margin_dir, patch_id)
                save_patch(patch_color, no_margin_save_path)
        else:
            # If it has a margin and label is 0, consider it a margin patch
            patches_with_margins.append(patch_id)
            save_path = os.path.join(margin_dir, patch_id)
            save_patch(patch_gray, save_path)  # Save grayscale version to margin_dir

    # Update the GT file to include only patches without margins or with label 1
    gt_df_filtered = gt_df[~gt_df['images'].isin(patches_with_margins)]
    gt_df_filtered.to_csv(gt_file_path, index=False)

    print("Patches copied to no_margin_dir (without margins or label 1):")
    for patch in gt_df_filtered['images']:
        print(patch)

    return len(patch_files), len(patches_with_margins)

def save_patch(patch, file_path):
    """
    Save a numpy array as an image file.

    Parameters:
    patch (numpy array): The image patch.
    file_path (str): Path to save the image file.
    """
    # Check if the patch is grayscale or color and save accordingly
    if len(patch.shape) == 2:  # Grayscale
        image = Image.fromarray(patch)
    else:  # Color
        image = Image.fromarray(patch.astype(np.uint8), 'RGB')
    image.save(file_path)

# Example usage
patch_dir = '/herdnet/DATASETS/TRAIN_Patches_5'  # Replace with your actual patch directory path
gt_file_path = '/herdnet/DATASETS/TRAIN_Patches_5/Train_binary_gt.csv'   # Replace with your GT file path
margin_dir = '/herdnet/DATASETS/margin_patches_TRAIN'  # Directory to save margin patches
no_margin_dir = '/herdnet/DATASETS/TRAIN_patches_no_margin'  # Directory to save patches without margins

# Set min_connected_area to 10% of patch size
original_count, margin_count = copy_no_margin_patches_and_update_gt(
    patch_dir, gt_file_path, margin_dir, no_margin_dir, 
    min_connected_area=0.07
)
# Copy the final GT file to the central directory
central_dir = '/herdnet/DATASETS'
final_gt_copy_path = os.path.join(central_dir, 'TRAIN_gt_no_margin.csv')
shutil.copy(gt_file_path, final_gt_copy_path)
print(f"GT file also copied to: {final_gt_copy_path}")
print(f"Original number of patches: {original_count}")
print(f"Number of patches with margins: {margin_count}")
print(f"Patches without margins or with label 1 saved to: {no_margin_dir}")
# Then copy the new no margins gt  manually to the folder containing no margin patches