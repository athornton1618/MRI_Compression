# Leverages the fastmri api, some key ideas borrowed from https://github.com/facebookresearch/fastMRI/blob/main/fastMRI_tutorial.ipynb
import h5py
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import fastmri
from fastmri.data import transforms as T
import os

def create_image(volume_kspace,slice_idx,output_file):
    slice_kspace = volume_kspace[slice_idx]
    slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
    slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
    slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image
    slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
    matplotlib.image.imsave(output_file, slice_image_rss)

if __name__ == '__main__':
    data_dir = '../data/multicoil_test/'
    output_dir = '../data/images/'
    #iterate through directory
    ii = 0
    for file in os.listdir(data_dir):
        print("Processing Scan "+str(ii))
        f = os.path.join(data_dir, file)
        hf = h5py.File(f)
        volume_kspace = hf['kspace'][()]
        n_slices = volume_kspace.shape[0]
        print('Number of Slices for this file: '+str(n_slices))
        filename = file.split('.')
        for slice_idx in range(n_slices):
            output_file = output_dir+filename[0]+'_'+str(slice_idx)+'.jpg'
            create_image(volume_kspace,slice_idx,output_file)
        ii+=1
    