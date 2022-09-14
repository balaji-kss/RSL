from queue import Full
from tkinter.tix import Tree
from torch.utils.data import DataLoader
from modelZoo.BinaryCoding import DynamicStream_Transformer, Dyan_Autoencoder, Dyan_BC_Autoencoder
from utils import *
import scipy.io
from modelZoo.networks import *
from torch.autograd import Variable
from scipy.spatial import distance
import torch.nn
from modelZoo.sparseCoding import sparseCodingGenerator
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from modelZoo.DyanOF import *
from dataset.NTU_viewProjection import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from lossFunction import binaryLoss
# torch.backends.cudnn.enabled = False
from lossFunction import hashingLoss, CrossEntropyLoss
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
import time

gpu_id = 0
num_workers = 2
PRE = 0

T = 36
dataset = 'NUCLA'

N = 80*2
Epoch = 100
dataType = '2D'
# clip = 'Single'
clip = 'Multi'
fusion = False
transformer = True

modelRoot = './crossViewModel/'
saveModel = modelRoot + dataset + '/dyanbc_recon/'
print('saveModel: ', saveModel)

if not os.path.exists(saveModel):
    os.makedirs(saveModel)
map_location = torch.device(gpu_id)

'load pre-trained DYAN'
preTrained = modelRoot + dataset + '/1110/dynamicStream_fista01_reWeighted_noBI_sqrC_T36_UCLA/'
#
stateDict = torch.load(os.path.join(preTrained, '100.pth'), map_location=map_location)['state_dict']

reduced = 0
scratch = 1

if scratch:
    P,Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
elif reduced:
    rr = stateDict['sparseCoding.rr'].cpu().numpy()
    theta = stateDict['sparseCoding.theta'].cpu().numpy()
    _, r_reduced,theta_reduced = get_reducedDictionary(rr, theta, THD_distance=0.05)

    Drr = torch.from_numpy(np.asarray(r_reduced)).float()
    Dtheta = torch.from_numpy(np.asarray(theta_reduced)).float()
    N = (Drr.shape[0])*2
else:
    Drr = stateDict['sparseCoding.rr'].float()
    Dtheta = stateDict['sparseCoding.theta'].float()
    print('Drr: ', Drr)
    print('Dtheta: ', Dtheta)

if dataset == 'NUCLA':
    num_class = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_3d'
    trainSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='train', cam='3,2', T=T,
                                   target_view='view_2', project_view='view_3', test_view='view_1')
    # #
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    valSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='3,2', T=T,
                                  target_view='view_1',
                                  project_view='view_2', test_view='view_1')
    valloader = DataLoader(valSet, batch_size=1, shuffle=True, num_workers=num_workers)

elif dataset == 'NTU':
    num_class = 60
    # num_class = 120
    # nanList = list(np.load('./NTU_badList.npz')['x'])
    with open('/data/NTU-RGBD/ntu_rgb_missings_60.txt', 'r') as f:
        nanList = f.readlines()
        nanList = [line.rstrip() for line in nanList]

    if dataType == '3D':
        root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    else:
        root_skeleton = "/data/NTU-RGBD/poses_60"

    trainSet = NTURGBD_viewProjection(root_skeleton=root_skeleton,
                                root_list="/data/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
                                phase='train', T=36, target_view='C002', project_view='C003', test_view='C001')

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    valSet = NTURGBD_viewProjection(root_skeleton=root_skeleton,
                                root_list="/data/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
                                phase='test', T=36, target_view='C002', project_view='C003', test_view='C001')
    valloader = DataLoader(valSet, batch_size=1, shuffle=True, num_workers=num_workers)


#Network
net = Dyan_BC_Autoencoder(Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, \
                    Inference=True, gpu_id=gpu_id, fistaLam=0.2).cuda(gpu_id)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, weight_decay=0.001,momentum=0.9)
net.train()

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.1)
mseLoss = torch.nn.MSELoss()

LOSS = []
ACC = []
print('training dataset:', dataset)

for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    start_time = time.time()
    loss_dyan = []
    loss_inp_recon = []
    total_loss = []

    for i, sample in enumerate(trainloader):

        optimizer.zero_grad()
        input_v1 = sample['target_skeleton'].float().cuda(gpu_id)
        input_v2 = sample['project_skeleton'].float().cuda(gpu_id)
        input_v1_img = sample['target_image'].float().cuda(gpu_id)
        input_v2_img = sample['project_image'].float().cuda(gpu_id)

        y = sample['action'].cuda(gpu_id)
        t1 = input_v1.shape[2]
        t2 = input_v2.shape[2]

        dyan_mse1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
        dyan_mse2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)

        input_mse1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
        input_mse2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)

        clipBI1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
        clipBI2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)

        for clip in range(0, input_v2.shape[1]):

            v1_clip = input_v1[:,clip,:,:,:].reshape(1, t1, -1)
            v2_clip = input_v2[:,clip,:,:,:].reshape(1, t2, -1)

            img1_clip = input_v1_img[:,clip,:,:,:]
            img2_clip = input_v2_img[:,clip,:,:,:]

            tdec_out1, dyan_out1, tenc_out1, bi_out1 = net(v1_clip, t1)
            tdec_out2, dyan_out2, tenc_out2, bi_out2 = net(v2_clip, t2)

            dyan_mse1[clip] = mseLoss(dyan_out1, tenc_out1)
            input_mse1[clip] = mseLoss(tdec_out1, v1_clip)

            dyan_mse2[clip] = mseLoss(dyan_out2, tenc_out2)
            input_mse2[clip] = mseLoss(tdec_out2, v2_clip)

            bi_gt1 = torch.zeros_like(b1).cuda(gpu_id)
            bi_gt2 = torch.zeros_like(b2).cuda(gpu_id)
            
        loss1 = torch.mean(dyan_mse1) + torch.mean(input_mse1)
        loss2 = torch.mean(dyan_mse2) + torch.mean(input_mse2)
        
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        total_loss.append(loss.data.item())
        loss_dyan.append((torch.mean(dyan_mse1)).data.item() + (torch.mean(dyan_mse2)).data.item())
        loss_inp_recon.append((torch.mean(input_mse1)).data.item() + (torch.mean(input_mse2)).data.item())

    total_loss_avg = np.mean(np.array(total_loss))
    loss_dyan_avg = np.mean(np.array(loss_dyan))
    loss_inp_recon_avg = np.mean(np.array(loss_inp_recon))

    end_time = time.time()
    time_per_epoch = (end_time - start_time)/60.0 #mins
    print('epoch: ', epoch, ' |time: ', np.round(time_per_epoch, 3), ' |loss:', np.round(total_loss_avg, 3), ' |dyan_mse:', np.round(loss_dyan_avg, 3), ' |inp_recon_mse:', np.round(loss_inp_recon_avg, 3))
    
    scheduler.step()

    if epoch % 5 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

    if epoch % 5 == 0:
        print('start validating:')
        val_total_loss, val_dyan_loss, val_input_loss = [], [], []
        
        with torch.no_grad():
            for i, sample in enumerate(valloader):
                
                inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
                inputImage = sample['input_image'].float().cuda(gpu_id)
                t = inputSkeleton.shape[2]

                dyan_mse = torch.zeros(inputSkeleton.shape[1]).cuda(gpu_id)
                input_mse = torch.zeros(inputSkeleton.shape[1]).cuda(gpu_id)
                
                for i in range(0, inputSkeleton.shape[1]):

                    input_clip = inputSkeleton[:,i, :, :, :].reshape(1, t, -1)
                    inputImg_clip = inputImage[:,i, :, :, :]

                    tdec_out, dyan_out, tenc_out = net(input_clip, t)
                    
                    dyan_mse[clip] = mseLoss(dyan_out, tenc_out)
                    input_mse[clip] = mseLoss(tdec_out, input_clip)

                dyan_mse_avg = torch.mean(dyan_mse)
                input_mse_avg = torch.mean(input_mse)
                val_loss = dyan_mse_avg + input_mse_avg    
                
                val_total_loss.append(val_loss.data.item())                
                val_dyan_loss.append(dyan_mse_avg.data.item())
                val_input_loss.append(input_mse_avg.data.item())

        val_total_loss_avg = np.round(np.mean(np.array(val_total_loss)), 3)
        val_dyan_loss_avg = np.round(np.mean(np.array(val_dyan_loss)), 3)
        val_input_loss_avg = np.round(np.mean(np.array(val_input_loss)), 3)

        print('validation: epoch: ', epoch, '  |loss:', val_total_loss_avg, ' |dyan_mse:', val_dyan_loss_avg, ' |inp_recon_mse:', val_input_loss_avg)
        
print('done')
