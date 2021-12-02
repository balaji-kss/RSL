import scipy.io
import os, sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def read_body_model():

    full_name = '/home/ubuntu/Documents/US/NEU/RA/skeletal_action_recognition_code/data/UTKinect/body_model.mat'
    #full_name = 'body_model_mat.mat'
    
    print('full_name ',full_name)
    mat = scipy.io.loadmat(full_name, simplify_cells=True)
    print(mat.keys())
    #bm = mat['body_model']
    bm = mat

    for key in bm.keys():
        bm[key] = np.array([bm[key]])
        print(key, bm[key], bm[key].shape)

read_body_model()