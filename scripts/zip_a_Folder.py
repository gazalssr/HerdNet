import zipfile
import os

def zip_folder(folder_path, output_zip_path):
    """
    Zip the contents of a folder.

    Parameters:
    folder_path (str): Path to the folder you want to zip.
    output_zip_path (str): Path where the output zip file should be saved.
    """
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)  # Relative path inside the zip
                zipf.write(file_path, arcname)
    print(f"Folder zipped successfully to: {output_zip_path}")

# Example usage
folder_to_zip = '/herdnet/DATASETS/VAL_patches_no_margin_5'  # Replace with the path to your folder with patches
output_zip_file = '/herdnet/DATASETS/P_DATA_ZIP/VAL_patches_no_margin_5.zip'  # Replace with where you want to save the zip file
zip_folder(folder_to_zip, output_zip_file)
