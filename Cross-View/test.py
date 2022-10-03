import torch
from modelZoo.BinaryCoding import *

gpu_id = 0
map_loc = "cuda:" + str(gpu_id)
num_class = 10
dataType = '2D'
N = 80 * 2

P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

net = Tenc_SparseC_Cl(num_class=num_class, Npole=N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2, fistaLam=0.1, gpu_id=gpu_id).cuda(gpu_id)

def load_pretrainedModel(stateDict, net):

    new_dict = net.state_dict()
    stateDict = stateDict['state_dict']
    pre_dict = {k: v for k, v in stateDict.items() if k in new_dict}
    print('pre_dict keys ', pre_dict.keys())
    new_dict.update(pre_dict)

    net.load_state_dict(new_dict)

    return net

dy_pretrain = '/home/balaji/Documents/code/RSL/Thesis/RSL/Cross-View/pretrained/setup1/Single/pretrainedDyan.pth'
tenc_pretrain = '/home/balaji/Documents/code/RSL/Thesis/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/tenc_recon_mask_wt/20.pth' 

def freeze_params(model):

    for param in model.parameters():
        param.requires_grad = False

def load_pretrain_models(net, dy_pretrain, tenc_pretrain):

    dy_state_dict = torch.load(dy_pretrain, map_location=map_loc)
    tenc_state_dict = torch.load(tenc_pretrain, map_location=map_loc)

    print('**** load pretrained dyan ****')
    net = load_pretrainedModel(dy_state_dict, net)
    
    print('**** load pretrained tenc ****')
    net = load_pretrainedModel(tenc_state_dict, net)

    print('**** freeze transformer_encoder params ****')
    freeze_params(net.transformer_encoder)

load_pretrain_models(net, dy_pretrain, tenc_pretrain)