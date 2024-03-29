import mrcfile
from PIL import Image

def mrc_to_png(input_mrc_file, filename, output_directory):
    with mrcfile.open(input_mrc_file, permissive=True) as mrc:
        for i, data_slice in enumerate(mrc.data):
            # Convert to uint8 and scale to 0-255 range
            data_slice = ((data_slice - data_slice.min()) / (data_slice.max() - data_slice.min()) * 255).astype('uint8')

            # Convert to PIL Image
            image = Image.fromarray(data_slice)

            # Save as PNG
            image.save(f"{output_directory}/{filename}_slice_{i:03d}.png")
