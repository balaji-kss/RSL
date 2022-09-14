from queue import Full
from tkinter.tix import Tree
from torch.utils.data import DataLoader
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
num_workers = 10
PRE = 0

T = 36
dataset = 'NUCLA'
# dataset = 'NTU'
Alpha = 0.1
# for BI
lam1 = 2  # for CL
lam2 = 1 # for MSE

N = 80*2
Epoch = 100
dataType = '2D'
# clip = 'Single'
clip = 'Multi'
fusion = False
transformer = True

modelRoot = './crossViewModel/'
saveModel = modelRoot + dataset + '/tenc_mask_full_0.5/'
print('saveModel: ', saveModel)

if not os.path.exists(saveModel):
    os.makedirs(saveModel)
map_location = torch.device(gpu_id)

'load pre-trained DYAN'
pretrained = '/home/balaji/Documents/code/RSL/NewCV/Cross-View/crossViewModel/NUCLA/1110/dynamicStream_fista01_reWeighted_noBI_sqrC_T36_UCLA/100.pth'
# pretrained = '/home/balaji/Documents/code/RSL/NewCV/Cross-View/crossViewModel/NUCLA/dyan_recon_mask/90.pth'
#
stateDict = torch.load(pretrained, map_location=map_location)['state_dict']
reduced = 0
scratch = 0

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
    # Drr = stateDict['sparse_coding.rr'].float()
    # Dtheta = stateDict['sparse_coding.theta'].float()
    print('Drr: ', Drr)
    print('Dtheta: ', Dtheta)

if dataset == 'NUCLA':
    num_class = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_3d'
    trainSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='train', cam='2,1', T=T,
                                   target_view='view_2', project_view='view_1', test_view='view_3')
    # #
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    valSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T,
                                  target_view='view_2',
                                  project_view='view_1', test_view='view_3')
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


def freeze_params(model):

    for param in model.parameters():
        param.requires_grad = False

#Network
if transformer:

    net = DynamicStream_Transformer(num_class=num_class, Npole=1*N+1, num_binary=1*N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True,
                            gpu_id=gpu_id, fistaLam=0.1).cuda(gpu_id)

    print('***** load pretrained transformer encoder *****')
    pretrain_transformer_path = './crossViewModel/NUCLA/dyan_recon_mask/100.pth'
    net.load_state_dict(torch.load(pretrain_transformer_path), strict=False)

    print('**** freeze transformer_encoder params ****')
    freeze_params(net.transformer_encoder)

    optimizer = torch.optim.SGD([
                                {'params':filter(lambda x: x.requires_grad, net.transformer_encoder.parameters()), 'lr':1e-5},
                                {'params':filter(lambda x: x.requires_grad, net.sparseCoding.parameters()), 'lr':1e-5},
                                {'params':filter(lambda x: x.requires_grad, net.Classifier.parameters()), 'lr':1e-3}
                                ], weight_decay=0.001, momentum=0.9)
else:

    net = DynamicStream(num_class=num_class, Npole=1*N+1, num_binary=1*N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True,
                            gpu_id=gpu_id, fistaLam=0.1).cuda(gpu_id)

    optimizer = torch.optim.SGD([
                                {'params':filter(lambda x: x.requires_grad, net.sparseCoding.parameters()), 'lr':1e-6},
                                {'params':filter(lambda x: x.requires_grad, net.Classifier.parameters()), 'lr':1e-3}
                                ], weight_decay=0.001, momentum=0.9)


net.train()


scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()
# Criterion = torch.nn.BCELoss()
LOSS = []
ACC = []
print('training dataset:', dataset)
print('alpha:', Alpha, 'lam1:', lam1, 'lam2:', lam2)
# print('cls:bi:reconst=2:0.3:1')
for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    start_time = time.time()
    lossVal = []
    lossCls = []
    lossBi = []
    lossMSE = []
    for i, sample in enumerate(trainloader):

        # print('sample:', i)
        optimizer.zero_grad()
        # input_v1 = sample['target_view_multiClips'].float().cuda(gpu_id)
        # input_v2 = sample['project_view_multiClips'].float().cuda(gpu_id)
        '2S'

        input_v1 = sample['target_skeleton'].float().cuda(gpu_id)
        input_v2 = sample['project_skeleton'].float().cuda(gpu_id)
        input_v1_img = sample['target_image'].float().cuda(gpu_id)
        input_v2_img = sample['project_image'].float().cuda(gpu_id)

        # input_v1 = sample['target_velocity'].float().cuda(gpu_id)
        # input_v2 = sample['project_velocity'].float().cuda(gpu_id)


        y = sample['action'].cuda(gpu_id)
        t1 = input_v1.shape[2]
        t2 = input_v2.shape[2]

        label1 = torch.zeros(input_v1.shape[1], num_class).cuda(gpu_id)
        label2 = torch.zeros(input_v2.shape[1], num_class).cuda(gpu_id)
        clipBI1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
        clipBI2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)
        clipMSE1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
        clipMSE2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)

        label_rgb1 = torch.zeros(input_v1.shape[1], num_class).cuda(gpu_id)
        label_rgb2 = torch.zeros(input_v2.shape[1], num_class).cuda(gpu_id)
        label_dym1 = torch.zeros(input_v1.shape[1], num_class).cuda(gpu_id)
        label_dym2 = torch.zeros(input_v2.shape[1], num_class).cuda(gpu_id)

        for clip in range(0, input_v2.shape[1]):

            v1_clip = input_v1[:,clip,:,:,:].reshape(1,t1,-1)
            v2_clip = input_v2[:,clip,:,:,:].reshape(1,t2, -1)

            'two stream model'
            #
            img1_clip = input_v1_img[:,clip,:,:,:]
            img2_clip = input_v2_img[:,clip,:,:,:]

            'Full Model'

            label_clip1, b1, outClip_v1, c1, tenc_out1 = net(v1_clip, t1)
            label_clip2, b2, outClip_v2, c2, tenc_out2 = net(v2_clip, t2)
            bi_gt1 = torch.zeros_like(b1).cuda(gpu_id)
            bi_gt2 = torch.zeros_like(b2).cuda(gpu_id)

            label1[clip] = label_clip1
            label2[clip] = label_clip2

            clipBI1[clip] = L1loss(b1, bi_gt1)

            clipMSE1[clip] = mseLoss(outClip_v1, tenc_out1)
            clipBI2[clip] = L1loss(b2, bi_gt2)
            clipMSE2[clip] = mseLoss(outClip_v2, tenc_out2)

        label1 = torch.mean(label1, 0, keepdim=True)
        label2 = torch.mean(label2, 0, keepdim=True)

        loss1 = lam1 * Criterion(label1, y) + Alpha * (torch.mean(clipBI1)) + lam2 * (torch.mean(clipMSE1))
        loss2 = lam1 * Criterion(label2, y) + Alpha * (torch.mean(clipBI2)) + lam2 * (torch.mean(clipMSE2))
        
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()
        lossVal.append(loss.data.item())
        lossCls.append(Criterion(label1, y).data.item() + Criterion(label2, y).data.item())
        lossBi.append((torch.mean(clipBI1)).data.item() + (torch.mean(clipBI2)).data.item())
        lossMSE.append((torch.mean(clipMSE1)).data.item() + (torch.mean(clipMSE2)).data.item())

    loss_val = np.mean(np.array(lossVal))

    end_time = time.time()
    time_per_epoch = (end_time - start_time)/60.0 #mins
    print('epoch:', epoch, '|time:', np.round(time_per_epoch, 3), '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|Bi:', np.mean(np.array(lossBi)),
          '|mse:', np.mean(np.array(lossMSE)))
    
    scheduler.step()

    if epoch % 5 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

    if epoch % 5 == 0:
        print('start validating:')
        count = 0
        pred_cnt = 0
        Acc = []
        with torch.no_grad():
            for i, sample in enumerate(valloader):
                # input = sample['test_view_multiClips'].float().cuda(gpu_id)
                inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)

                # inputSkeleton = sample['test_velocity'].float().cuda(gpu_id)
                inputImage = sample['input_image'].float().cuda(gpu_id)

                t = inputSkeleton.shape[2]
                y = sample['action'].data.item()
                label = torch.zeros(inputSkeleton.shape[1], num_class)
                for i in range(0, inputSkeleton.shape[1]):

                    input_clip = inputSkeleton[:,i, :, :, :].reshape(1, t, -1)
                    inputImg_clip = inputImage[:,i, :, :, :]
                    # label_clip, _, _ = net(input_clip, t) # DY+BL+CL
                    # label_clip, _ = net(input_clip, t) # DY+CL

                    if fusion:
                        label_clip, _, _ = net.dynamicsClassifier(input_clip, t) # two stream, dynamcis branch
                    else:
                        # label_clip, _, _,_,_ = net(input_clip, inputImg_clip, t, fusion)
                        # label_clip,_ = net(input_clip, t) #DY
                        label_clip, _, _, _,_ = net(input_clip, t) # DY+BI

                    label[i] = label_clip
                label = torch.mean(label, 0, keepdim=True)

                # c, _ = Encoder.forward2(input, T)
                # c = c.reshape(1, N + 1, int(input.shape[-1]/2), 2)
                # label = net(c)  # CL only
                # label, _ = net(c) # 'BI + CL'

                # label, _ = net(input, T) # 'DY + CL'

                pred = torch.argmax(label).data.item()
                # print('sample:',i, 'pred:', pred, 'gt:', y)
                count += 1
                # if pred1 == y:
                #     pred_cnt +=1
                # elif pred2 == y:
                #     pred_cnt += 1
                if pred == y:
                    pred_cnt += 1

                # for n in range(0, label.shape[0]):
                #     pred = torch.argmax(label[n]).data.item()
                #     if pred == y[0]:
                #         count+= 1
                # acc = count/label.shape[0]
            # Acc.append(acc)
            # Acc = count/valSet.__len__()
            Acc = pred_cnt/count

            print('epoch:', epoch, 'Acc:%.4f'% Acc, 'count:',count, 'pred_cnt:', pred_cnt)
            ACC.append(Acc)

# data = {'LOSS':np.asarray(LOSS), 'Acc':np.asarray(ACC)}
# scipy.io.savemat('./matFile/Classifier_train_v12_test_v3_10cls.mat', mdict=data)

print('done')
