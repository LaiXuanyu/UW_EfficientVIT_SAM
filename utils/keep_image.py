import os
import shutil

# Define the paths for mask and image directories
mask_directory = "datasets/SUIM/train_val/train_val/masks"
image_directory = "datasets/SUIM/train_val/train_val/images"
output_directory = 'image'  # Output directory to save matching images

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get the filenames in the mask directory (excluding extensions)
mask_filenames = {os.path.splitext(filename)[0] for filename in os.listdir(mask_directory)}

# Iterate through the files in the image directory
for filename in os.listdir(image_directory):
    # Get the filename without the extension
    name_without_extension = os.path.splitext(filename)[0]
    if name_without_extension in mask_filenames:
        # If the filename is in the list of mask filenames, copy it to the output directory
        source_path = os.path.join(image_directory, filename)
        destination_path = os.path.join(output_directory, filename)
        shutil.copy(source_path, destination_path)
        print(f'Copied: {source_path} to {destination_path}')
