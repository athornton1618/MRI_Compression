import pandas as pd
import numpy as np
import os
import h5py
import sys
sys.path.append('../')
from common_fcns import combine_all_coils, resize_scan, k_encode

if __name__ == '__main__':
    # Load in the scans, compute k_encoding, save off
    data_dir = '../data/multicoil_test/'
    slice_name = []
    slice = []
    K = 3000 #fix K for 'good' value
    #iterate through directory
    ii = 0
    for file in os.listdir(data_dir):
        print("Processing Scan "+str(ii))
        f = os.path.join(data_dir, file)
        hf = h5py.File(f)
        volume_kspace = hf['kspace'][()]
        n_slices = volume_kspace.shape[0]
        filename = file.split('.')
        for slice_idx in range(n_slices):
            slice_name.append(filename[0]+'_'+str(slice_idx))
            X_raw = combine_all_coils(volume_kspace,slice_idx)
            X = resize_scan(X_raw)
            X_encode = np.array(k_encode(K,X)).flatten()
            slice.append(X_encode)
        ii+=1

    # Make a nice DataFrame of the samples
    d = {'slice_name': slice_name, 'slice': slice}
    df = pd.DataFrame(data=d)
    df.to_pickle("../data/training_set.pkl",protocol=4) # Google colab is dumb and needs protocol 4 as of 4/24/2022
    # Also want json
    result = df.to_json(r'../data/training_set.json')

    