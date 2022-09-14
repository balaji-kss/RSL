from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from modelZoo.BinaryCoding import DynamicStream_Transformer
from utils import *
import scipy.io
from modelZoo.networks import *
from torch.autograd import Variable
from scipy.spatial import distance
import torch.nn
from modelZoo.sparseCoding import sparseCodingGenerator
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from dataset.NUCLA_viewProjection_CS import *
from dataset.NTURGBDsubject import *
from modelZoo.DyanOF import *
from dataset.NTU_viewProjection import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from lossFunction import binaryLoss

from dataset.JHMDB_dloader import *
import modelZoo.keyframe.networks as kf

gpu_id = 0
num_workers = 4
fistaLam = 0.1
print('gpu_id: ',gpu_id)
print('num_workers: ',num_workers)
print('fistaLam: ',fistaLam)

PRE = 0
T = 36
dataset = 'NUCLA'

N = 80*2
num_class = 10
dataType = '2D'

clip = 'Multi'
fusion = False

def load_kftest_data():

    dataRoot = '/data/Yuexi/JHMDB/'
    trainAnnot, testAnnot = get_train_test_annotation(dataRoot)
    testSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='test',if_occ=False)
    test_loader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=4)

    return test_loader


def load_test_data():

    if dataset == 'NUCLA':
        num_class = 10
        path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
        
        'CS:'

        'CV:'
        testSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='1,2', T=T,
                                    target_view='view_2',
                                    project_view='view_1', test_view='view_3')
        testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=num_workers)

    elif dataset == 'NTU':
        num_class = 60
        if dataType == '3D':
            root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
        else:
            root_skeleton = "/data/NTU-RGBD/poses"
        nanList = list(np.load('./NTU_badList.npz')['x'])
        'CS:'

        'CV:'
        testSet = NTURGBD_viewProjection(root_skeleton, root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
                                    phase='train', T=36, target_view='C002', project_view='C001', test_view='C003')
        testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=num_workers)

    return testSet, testloader

def load_train_data():

    if dataset == 'NUCLA':
        num_class = 10
        path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
        
        'CS:'

        'CV:'
        trainSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='train', cam='2,1', T=T,
                                   target_view='view_2', project_view='view_1', test_view='view_3')
        trainloader = DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=num_workers)

    elif dataset == 'NTU':
        num_class = 60
        if dataType == '3D':
            root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
        else:
            root_skeleton = "/data/NTU-RGBD/poses"
        nanList = list(np.load('./NTU_badList.npz')['x'])
        'CS:'

        'CV:'
        trainSet = NTURGBD_viewProjection(root_skeleton=root_skeleton,
                                root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
                                phase='train', T=36, target_view='C002', project_view='C003', test_view='C001')

        trainloader = torch.utils.data.DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    return trainSet, trainloader

def load_kf_model():

    FRA = 30
    T = 40
    modelPath = '/home/balaji/Documents/code/RSL/NewCV/Key-Frame-Proposal-Network-for-Efficient-Pose-Estimation-in-Videos/models'
    modelFile = os.path.join(modelPath, 'kfpn_jhmdb_online.pth')
    map_location = torch.device(gpu_id)
    state_dict = torch.load(modelFile, map_location=map_location)['state_dict']
    Drr = state_dict['K_FPN.Drr']
    Dtheta = state_dict['K_FPN.Dtheta']
    net = kf.onlineUpdate(FRA=FRA, PRE=PRE,T=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)

    net.load_state_dict(state_dict)
    net.eval()
    net.cuda(gpu_id)

    return net

def load_model(model_path):

    map_location = torch.device(gpu_id)
    stateDict = torch.load(model_path, map_location=map_location)['state_dict']

    return stateDict

def load_net(num_class, stateDict, transformer=False):

    Drr = stateDict['sparseCoding.rr']
    Dtheta = stateDict['sparseCoding.theta']
    kinetics_pretrain = './pretrained/i3d_kinetics.pth'
    # net = twoStreamClassification(num_class=num_class, Npole=(1*N+1), num_binary=(1*N+1), Drr=Drr, Dtheta=Dtheta,
    #                         dim=2, gpu_id=gpu_id, inference=True, fistaLam=0.1, dataType=dataType, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

    if transformer:
        net = DynamicStream_Transformer(num_class=num_class, Npole=1*N+1, num_binary=1*N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True,
                            gpu_id=gpu_id, fistaLam=0.1).cuda(gpu_id)

    else:
        net = DynamicStream(num_class=num_class, Npole=1*N+1, num_binary=1*N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True,
                            gpu_id=gpu_id, fistaLam=0.1).cuda(gpu_id)
        

    net.load_state_dict(stateDict)
    net.eval()

    return net

class GradCamModel_RGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        stateDict = load_model()
        self.net = load_net(num_class, stateDict)

        for name, param in self.net.named_parameters():
            param.requires_grad = True
            #print(name, param.data.shape)

        # self.layerhook.append(self.net.RGBClassifier.featureExtractor.base_model[0].conv3d.register_forward_hook(self.forward_hook()))
        self.layerhook.append(self.net.RGBClassifier.layer1.register_forward_hook(self.forward_hook()))
        
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        
        return self.gradients

    def forward_hook(self):

        def hook(module, inp, out):    
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
            
        return hook

    def forward(self, input_clip, inputImg_clip, t, fusion=False):
        label_clip, b, outClip_v = self.net(input_clip, inputImg_clip, t, fusion)
        return label_clip, b, outClip_v, self.selected_out

class GradCamModel_DYN(nn.Module):
    def __init__(self, onlyds = False):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.onlyds = onlyds
        stateDict = load_model()
        self.net = load_net(num_class, stateDict)

        for name, param in self.net.named_parameters():
            param.requires_grad = True
            #print(name, param.data.shape)

        if self.onlyds:
            self.layerhook.append(self.net.data.register_forward_hook(self.forward_hook()))
        else:
            self.layerhook.append(self.net.dynamicsClassifier.sparseCoding.register_forward_hook(self.forward_hook()))
        
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        
        return self.gradients

    def forward_hook(self):

        def hook(module, inp, out):
            out = out[0]
            #print('hook out shape: ', out.shape)    
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
            
        return hook

    def forward(self, input_clip, inputImg_clip, t, fusion=False):

        if self.onlyds:
            label_clip, b, outClip_v, _, _ = self.net(input_clip, t)
        else:
            label_clip, b, outClip_v, _, _ = self.net(input_clip, inputImg_clip, t, fusion)

        return label_clip, b, outClip_v, self.selected_out

if __name__ == '__main__':
    
    gcmodel = GradCamModel_RGB().to('cuda:0')