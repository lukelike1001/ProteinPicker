import mrcfile
from PIL import Image

def mrc_to_png(input_mrc_file, output_directory):
    with mrcfile.open(input_mrc_file, permissive=True) as mrc:
        for i, data_slice in enumerate(mrc.data):
            # Convert to uint8 and scale to 0-255 range
            data_slice = ((data_slice - data_slice.min()) / (data_slice.max() - data_slice.min()) * 255).astype('uint8')

            # Convert to PIL Image
            image = Image.fromarray(data_slice)

            # Save as PNG
            image.save(f"{output_directory}/slice_{i:03d}.png")

# Generate PNG images from MRC file
mrc_to_png("mrc/2fak/atlas_fix_2fak.mrc", "png-imgs/2fak")
mrc_to_png("mrc/coagulation/atlas_AF_Q4R562_F1_model_v4.mrc", "png-imgs/coagulation")