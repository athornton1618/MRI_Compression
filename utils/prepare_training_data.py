import pandas as pd
import numpy as np
import os
import h5py
import sys
sys.path.append('../')
from common_fcns import combine_all_coils, resize_scan, k_encode, quantize

if __name__ == '__main__':
    # Load in the scans, compute k_encoding, save off
    data_dir = '../data/multicoil_test/'
    train_slice_name = []
    train_slice = []
    test_slice_name = []
    test_slice = []
    K = 10000 #fix K for 'good' value
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
            X_raw = combine_all_coils(volume_kspace,slice_idx)
            X = resize_scan(X_raw)
            X = quantize(X)
            X_encode = np.array(k_encode(K,X)).flatten()
            if ii==0:
                test_slice_name.append(filename[0]+'_'+str(slice_idx))
                test_slice.append(X_encode)
            else:
                train_slice_name.append(filename[0]+'_'+str(slice_idx))
                train_slice.append(X_encode)
            #print(X_encode.shape)
        ii+=1

    # Make a nice DataFrame of the samples
    train_d = {'slice_name': train_slice_name, 'slice': train_slice}
    train_df = pd.DataFrame(data=train_d)
    train_result = train_df.to_json(r'../data/train_set.json')
    test_d = {'slice_name': test_slice_name, 'slice': test_slice}
    test_df = pd.DataFrame(data=test_d)
    test_result = test_df.to_json(r'../data/test_set.json')

    