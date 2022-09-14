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
from PIL import Image

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

gpu_id = 0
num_workers = 2
PRE = 0

T = 36
dataset = 'NUCLA'
# dataset = 'NTU'
Alpha = 0.1
lam1 = 2
lam2 = 1

N = 80*2

dataType = '2D'
# clip = 'Single'
clip = 'Multi'
fusion = False
# fusion = True
#
modelRoot = './crossViewModel/'

#modelPath = modelRoot + dataset + '/1110/dynamicStream_fista01_reWeighted_noBI_sqrC_T36_UCLA/'
modelPath = modelRoot + dataset + '/dyn_reproduce_fix/'
modelPath = os.path.join(modelPath, '100.pth')
plot_dir = modelRoot + dataset + '/dyn_reproduce_fix/plots/'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

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
    'CS:'

    'CV:'
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

    'CV:'
    
net = Fullclassification(num_class=num_class, Npole=1*N+1, num_binary=1*N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True,
                         gpu_id=gpu_id, fistaLam=0.1).cuda(gpu_id)

net.load_state_dict(stateDict)
# net.eval()
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

count = 0
pred_cnt = 0
classLabel = [[] for i in range(0, num_class)]
classGT = [[] for i in range(0, num_class)]
bis = []
coeffs = []
coeffsqrs = []


def get_plot(coeff_sqr, bi, idx, sname):

    xdata = [i for i in range(161)]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Square of Coefficient and Binary code for: ' + str(idx))
    ax1.set_title('Square of coefficient')
    ax2.set_title('Binary')
    ax1.set_ylim(0, 1.25)
    ax1.scatter(xdata, coeff_sqr, color ='tab:blue')
    ax2.scatter(xdata, bi, color ='tab:red') 
    plt.savefig(sname)
    plt.clf()

def mosaic_plot(bi, coeff):

    bi_np = bi[0].cpu().numpy()
    coeff_np = coeff[0].cpu().numpy()
    coeff_sqr_np = coeff_np ** 2

    num_pts = bi_np.shape[-1]
    tile = (10, 5)
    fig_size = (60, 80)
    mosaic = np.zeros((fig_size[0] * tile[0], fig_size[1] * tile[1], 3), dtype='uint8')
    
    for idx in range(num_pts):
        xidx, yidx = idx%tile[1], idx//tile[1]
        bi_np_idx = bi_np[:, idx]
        coeff_np_idx = coeff_np[:, idx]
        coeff_sqr_np_idx = coeff_sqr_np[:, idx]
        plot_sname = plot_dir + '_' + str(idx) + '.png'
        get_plot(coeff_sqr_np_idx, bi_np_idx, idx, plot_sname)
        img = cv2.imread(plot_sname)
        
        # cv2.imshow(str(idx), img)
        # cv2.waitKey(-1)
        img = cv2.resize(img, None, fx=0.125, fy=0.125)
        sx, sy, ex, ey = xidx * fig_size[1], yidx * fig_size[0], (xidx + 1) * fig_size[1], (yidx + 1) * fig_size[0] 
        print('sx, sy, ex, ey ', sx, sy, ex, ey)
        mosaic[sy: ey, sx: ex] = img
        
    cv2.imshow('mosaic', mosaic)
    cv2.waitKey(-1)

with torch.no_grad():
    for s, sample in enumerate(testloader):
        # print('testing:', s)
        # input = sample['test_view_multiClips'].float().cuda(gpu_id)
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        # inputSkeleton = sample['project_skeleton'].float().cuda(gpu_id)

        # inputSkeleton = sample['test_velocity'].float().cuda(gpu_id)
        inputImage = sample['input_image'].float().cuda(gpu_id)

        t = inputSkeleton.shape[2]
        y = sample['action'].data.item()
        label = torch.zeros(inputSkeleton.shape[1], num_class)

        print('inputSkeleton shape: ', inputSkeleton.shape)

        start = time.time()
        for i in range(0, inputSkeleton.shape[1]):
        # for i in range(0, 1):

            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            inputImg_clip = inputImage[:, i, :, :, :]
            # label_clip, _, _ = net(input_clip, t) # DY+BL+CL
            # label_clip, _ = net(input_clip, t) # DY+CL

            if fusion:
                label_clip, _, _ = net.dynamicsClassifier(input_clip, t)  # two stream, dynamcis branch
            else:
                # label_clip, _, _,coeff, d = net(input_clip, inputImg_clip, t, fusion)
                # label_clip, _ = net(input_clip, t) #DY
                label_clip, bi, yr, coeff, d = net(input_clip, t)# DY+BI
                mosaic_plot(bi, coeff)

            label[i] = label_clip
            
        label = torch.mean(label, 0, keepdim=True)
        end = time.time()
        print('time:', (end-start), 'time/clip:', (end-start)/inputSkeleton.shape[1])
    
        pred = torch.argmax(label).data.item()
        print('sample:', s, 'pred:', pred, 'gt:', y)
        count += 1

        if pred == y:
            pred_cnt += 1

    Acc = pred_cnt / count

    print('Acc:%.4f' % Acc, 'count:', count, 'pred_cnt:', pred_cnt)
    

print('done')
