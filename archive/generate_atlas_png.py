# import python 3.9.2
from mrc_to_png import *
import os

def generate_batch_png(input_filepath, output_filepath):
    for filename in os.listdir(input_filepath):
        if filename.endswith('.mrc'):
            mrc_filepath = os.path.join(input_filepath, filename)
            mrc_to_png(mrc_filepath, filename[:-4], output_filepath)

# Find the current working directory
cwd = os.getcwd()

# Generate PNG images for BLTP2, Coagulation, and RIF1
# Batch 1: Atlas files (w/out membrane)
generate_batch_png("mrc/atlas-mrc/bltp2", "imgs/atlas-imgs/bltp2")
generate_batch_png("mrc/atlas-mrc/coagulation", "imgs/atlas-imgs/coagulation")
generate_batch_png("mrc/atlas-mrc/rif1", "imgs/atlas-imgs/rif1")

# Batch 2: Tomogram files (w/out membrane)
generate_batch_png("mrc/tomogram-mrc/bltp2", "imgs/tomogram-imgs/bltp2")
generate_batch_png("mrc/tomogram-mrc/coagulation", "imgs/tomogram-imgs/coagulation")
generate_batch_png("mrc/tomogram-mrc/rif1", "imgs/tomogram-imgs/rif1")