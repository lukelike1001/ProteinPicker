import mrcfile
import numpy as np

file_path = 'mrc/atlas-mrc/rif1/atlas_RIF1_4.mrc'
with mrcfile.open(file_path) as mrc:
    mrc_data = mrc.data
    print(mrc_data.shape)
    
    first_slice = mrc_data[0]
    print(type(first_slice[0][0]))
    
    num_zeros = num_ones = 0
    tolerance = 1e-6
    
    for i in range(100):
        for j in range(100):
            if np.abs(first_slice[i][j]) < tolerance:
                num_zeros += 1
            else:
                num_ones += 1
    
    print(f"There are {num_zeros} 0s in the first slice.")
    print(f"There are {num_ones} 1s in the first slice.")
