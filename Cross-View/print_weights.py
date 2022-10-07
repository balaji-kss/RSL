from torch.utils.data import DataLoader
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from testClassifier_CV import testing, getPlots
import time 

def get_dict_params(stateDict):

    Drr = stateDict['sparseCoding.rr'].float()
    Dtheta = stateDict['sparseCoding.theta'].float()
    
    return Drr, Dtheta

def get_tenc_keys(dict_keys):

    req_str = "transformer_encoder"
    tenc_keys = []
    for key in dict_keys:
        if req_str in key:
            tenc_keys.append(key)

    return tenc_keys

def get_transformer_params(stateDict, rand_val):
    
    tenc_keys = get_tenc_keys(stateDict.keys())

    print('key: ', tenc_keys[rand_val])

    return stateDict[tenc_keys[rand_val]]

def check_transformer_params(model_path1, model_path2):

    state_dict1 = torch.load(model_path1, map_location=map_loc)['state_dict']
    state_dict2 = torch.load(model_path2, map_location=map_loc)['state_dict']

    tenc_key_idx = random.randint(0, 108) # total tenc keys
    print('tenc_key_idx ', tenc_key_idx)
    val1 = get_transformer_params(state_dict1, tenc_key_idx)
    val2 = get_transformer_params(state_dict2, tenc_key_idx)

    print('val1: ', val1)
    print('val2: ', val2)

    if torch.equal(val1, val2): print('TENC MATCHING')
    else: print('TENC NOT MATCHING')

if __name__ == "__main__":

    gpu_id = 1
    map_loc = "cuda:" + str(gpu_id)
    pt_model_path = '/home/balaji/Documents/code/RSL/Thesis/RSL/Cross-View/pretrained/setup1/Single/pretrainedDyan.pth'
    model_path = '/home/balaji/Documents/code/RSL/Thesis/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/dyan_debug/T36_fista01_openpose/20.pth'

    print('model_path1: ', pt_model_path)
    print('model_path2: ', model_path)

    ## Check dict params
    stateDict1 = torch.load(pt_model_path, map_location=map_loc)['state_dict']
    Drr1, Dtheta1 = get_dict_params(stateDict1)
    stateDict2 = torch.load(model_path, map_location=map_loc)['state_dict']
    Drr2, Dtheta2 = get_dict_params(stateDict2)

    if torch.equal(Drr1, Drr2): print('DRR MATCHING')
    else: print('DRR NOT MATCHING')

    if torch.equal(Dtheta1, Dtheta2): print('DTHETA MATCHING')
    else: print('DTHETA NOT MATCHING')


    ## Check transformer params
    # for i in range(5):
    #     check_transformer_params(pt_model_path, model_path)