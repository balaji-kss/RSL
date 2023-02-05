from torch.utils.data import DataLoader
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from testClassifier_CV import testing, getPlots
import time

gpu_id = 0
map_loc = "cuda:" + str(gpu_id)

# T = 36
dataset = 'NUCLA'

Alpha = 0.01 # bi loss
lam1 = 2 # cls loss
lam2 = 1 # mse loss

N = 80 * 2
Epoch = 100
# num_class = 10
dataType = '2D'
clip = 'Multi'

if clip == 'Single':
    num_workers = 8
    bz = 32
else:
    num_workers = 4
    bz = 6

T = 36 # input clip length

fusion = False
'initialized params'
P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

modelRoot = './ModelFile/crossView_NUCLA/'
mode = '/dyan_bi_tenc_cl_rhdyan/'

saveModel = modelRoot + clip + mode + 'T36_fista01_openpose/'
if not os.path.exists(saveModel):
    os.makedirs(saveModel)
print('model:',mode, 'model path:', saveModel)
num_class = 10
setup = 'setup1' # v1,v2 train, v3 test;
path_list = './data/CV/' + setup + '/'
# root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
trainSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, clip=clip, phase='train', cam='2,1', T=T,
                                setup=setup)
# #

trainloader = DataLoader(trainSet, batch_size=bz, shuffle=True, num_workers=num_workers)

testSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T, setup=setup)
testloader = DataLoader(testSet, batch_size=bz, shuffle=True, num_workers=num_workers)

'dy+bi+cl'
dy_pretrain = './pretrained/' + setup + '/' + clip + '/pretrainedRHdyan_BI.pth'
# net = Fullclassification(num_class=10, Npole=N+1, num_binary=N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True, gpu_id=gpu_id, fistaLam=0.1).cuda(gpu_id)
net = Dyan_Tenc_multi(num_class=num_class, Npole=N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2, fistaLam=0.1, Inference=True, gpu_id=gpu_id).cuda(gpu_id)

stateDict = torch.load(dy_pretrain, map_location=map_loc)
net = load_pretrainedModel(stateDict, net)

net.train()
lr = 5e-4  # classifier
lr_2 = 1e-4  # sparse codeing

'for dy+bi+cl'
optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.sparseCoding.parameters()), 'lr':lr_2},
{'params':filter(lambda x: x.requires_grad, net.transformer_encoder.parameters()), 'lr':lr},
                             {'params':filter(lambda x: x.requires_grad, net.Classifier.parameters()), 'lr':lr}], weight_decay=0.001, momentum=0.9)


scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()
L1loss = torch.nn.SmoothL1Loss()


LOSS = []
ACC = []
LOSS_CLS = []
LOSS_MSE = []
LOSS_BI = []
print('Experiment config(setup, clip, lam1, lam2, lr, lr_2):', setup, clip, Alpha, lam1, lam2, lr, lr_2)
for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    start_time = time.time()
    lossVal = []
    lossCls = []
    lossBi = []
    lossMSE = []
    count = 0
    pred_cnt = 0
    for i, sample in enumerate(trainloader):

        # print('sample:', i)
        optimizer.zero_grad()

        skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
        input_images = sample['input_images'].float().cuda(gpu_id)
        gt_label = sample['action'].cuda(gpu_id)

        if clip == 'Single':
            t = skeletons.shape[1]
            input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)

        else:
            t = skeletons.shape[2]
            input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz, clip, T, 50

        'dy+bi+cl:'
        actPred, binaryCode, output_skeletons = net(input_skeletons, t)
        bi_gt = torch.zeros_like(binaryCode).cuda(gpu_id)

        if clip == 'Single':
            actPred = actPred
            pred = torch.argmax(actPred, 1)

        else:
            actPred = actPred.reshape(skeletons.shape[0], skeletons.shape[1], num_class)
            actPred = torch.mean(actPred, 1)
            pred = torch.argmax(actPred, 1)

        cls_loss = Criterion(actPred, gt_label)
        mse_loss = mseLoss(output_skeletons, input_skeletons)
        bi_loss = L1loss(binaryCode, bi_gt)

        loss = lam1 * cls_loss + lam2 * mse_loss \
               + Alpha * bi_loss
        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())
        lossCls.append(cls_loss.data.item())
        lossMSE.append(mse_loss.data.item())
        lossBi.append(bi_loss.data.item())

        ## Train acc
        correct = torch.eq(gt_label, pred).int()
        count += gt_label.shape[0]
        pred_cnt += torch.sum(correct).data.item()

    loss_val = np.mean(np.array(lossVal))
    LOSS.append(loss_val)
    LOSS_CLS.append(np.mean(np.array((lossCls))))
    LOSS_MSE.append(np.mean(np.array(lossMSE)))
    LOSS_BI.append(np.mean(np.array(lossBi)))

    end_time = time.time()
    train_acc = pred_cnt/count
    time_per_epoch = (end_time - start_time)/60.0 #mins

    print('epoch:', epoch, ' |time: ', np.round(time_per_epoch, 3), '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|mse:', np.mean(np.array(lossMSE)), '|acc:', train_acc, '|bi:', np.mean(np.array(lossBi)))

    scheduler.step()
    if epoch % 10 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')
    
        Acc = testing(testloader, net, gpu_id, clip)

        print('testing epoch:',epoch, 'Acc:%.4f'% Acc)
        ACC.append(Acc)


'plotting results:'
# getPlots(LOSS,LOSS_CLS, LOSS_MSE, LOSS_BI, ACC,fig_name='DY_CL.pdf')

print('done')
