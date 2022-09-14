from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
import time
from lossFunction import binaryLoss
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

gpu_id = 0
num_workers = 1
PRE = 0

T = 36
dataset = 'NUCLA'
# dataset = 'NTU'
N = 80

dataType = '2D'
# clip = 'Single'
clip = 'Multi'
fusion = False

# modelPath = '/home/balaji/Documents/code/RSL/NewCV/Cross-View/crossViewModel/NUCLA/1110/dynamicStream_fista01_reWeighted_noBI_sqrC_T36_UCLA/100.pth'
modelPath = '/home/balaji/Documents/code/RSL/NewCV/Cross-View/crossViewModel/NUCLA/freeze_tenc_full/100.pth'

map_location = torch.device(gpu_id)
stateDict = torch.load(modelPath, map_location=map_location)['state_dict']


Drr = stateDict['sparseCoding.rr']
Dtheta = stateDict['sparseCoding.theta']
print('Drr shape: ',Drr.shape)
print('Dtheta shape: ',Dtheta.shape)

data = {'Drr': Drr.cpu().numpy(), 'Dtheta':Dtheta.cpu().numpy()}



if dataset == 'NUCLA':
    num_class = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    
    print('test dataset:', dataset, 'cross view experiment')
    testSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T,
                                  target_view='view_2',
                                  project_view='view_1', test_view='view_3')
    testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=num_workers)

elif dataset == 'NTU':
    num_class = 60
    # root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    if dataType == '3D':
        root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    else:
        root_skeleton = "/data/NTU-RGBD/poses"
    nanList = list(np.load('./NTU_badList.npz')['x'])
    'CS:'

    testSet = NTURGBDsubject(root_skeleton, nanList, dataType=dataType, clip=clip, phase='test', T=36)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

net = DynamicStream_Transformer(num_class=num_class, Npole=2*N+1, num_binary=1*N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True,
                         gpu_id=gpu_id, fistaLam=0.2).cuda(gpu_id)

net.load_state_dict(stateDict)
# net.eval()
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

count = 0
pred_cnt = 0
ACC = []
classLabel = [[] for i in range(0, num_class)]
classGT = [[] for i in range(0, num_class)]

binaryCode = []
origCoeff = []

with torch.no_grad():
    for s, sample in enumerate(testloader):
        
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        inputImage = sample['input_image'].float().cuda(gpu_id)

        t = inputSkeleton.shape[2]
        y = sample['action'].data.item()
        label = torch.zeros(inputSkeleton.shape[1], num_class)

        result_bi = []
        result_coffe = []
        result_y = []

        start = time.time()
        for i in range(0, inputSkeleton.shape[1]):
        # for i in range(0, 1):

            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            inputImg_clip = inputImage[:, i, :, :, :]

            if fusion:
                label_clip, _, _ = net.dynamicsClassifier(input_clip, t)  # two stream, dynamcis branch
            else:
                label_clip, bi, yr, coeff, d = net(input_clip, t)# DY+BI
                

                if y == 5:
                    result_bi.append(bi.cpu().numpy())
                    result_coffe.append(coeff.cpu().numpy())
                    

            label[i] = label_clip
        label = torch.mean(label, 0, keepdim=True)
        end = time.time()
        print('time:', (end-start), 'time/clip:', (end-start)/inputSkeleton.shape[1])

        pred = torch.argmax(label).data.item()
        print('sample:',s, 'pred:', pred, 'gt:', y)
        count += 1
        

        if pred == y:
            pred_cnt += 1
            binaryCode = binaryCode+result_bi
            origCoeff = origCoeff + result_coffe

        # if len(binaryCode) ==12:
        #     origCoeff = np.asarray(origCoeff)
        #     binaryCode = np.asarray(binaryCode)

        #     dict = {'bi_action': binaryCode, 'coeff_action':origCoeff}
            
        #     scipy.io.savemat('/data/Yuexi/Cross_view/1129/setup1/coeff_bi_v3_action05.mat', mdict=dict)

        #     print('check')

    Acc = pred_cnt / count
    print('Acc: ', Acc)

print('done')
