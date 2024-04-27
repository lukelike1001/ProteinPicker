import mrcfile
import numpy as np
from tifffile import imsave

# Step 1: Read MRC File
def read_mrc_file(file_path):
    with mrcfile.open(file_path) as mrc:
        data = mrc.data  # Read MRC data into a NumPy array
    return data

# Step 2: Extract Images (if applicable)
def extract_images(data):
    # Example: Extract each slice as a separate NumPy array
    return [slice_data for slice_data in np.moveaxis(data, -1, 0)]

# Step 3: Save as TIFF
def save_as_tiff(images, output_dir):
    for i, image in enumerate(images):
        file_name = f"image_{i}.tiff"
        imsave(f"{output_dir}/{file_name}", image)


def convert_mrc_to_tiff(mrc_file_path, output_directory):
    mrc_data = read_mrc_file(mrc_file_path)
    image_slices = extract_images(mrc_data)
    save_as_tiff(image_slices, output_directory)
    

"""
We need to run convert_mrc_to_tiff for all proteins that we use.
1. Coagulation
2. 2FAK
"""
convert_mrc_to_tiff("mrc/2fak/atlas_fix_2fak.mrc", "tiff-imgs/2fak")
convert_mrc_to_tiff("mrc/coagulation/atlas_AF_Q4R562_F1_model_v4.mrc", "tiff-imgs/coagulation")