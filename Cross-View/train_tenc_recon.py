from torch.utils.data import DataLoader
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from testClassifier_CV import test_reconstruct, getPlots
import time

gpu_id = 0
map_loc = "cuda:" + str(gpu_id)

dataset = 'NUCLA'

N = 80 * 2
Epoch = 300
fistaLam = 0.1
dataType = '2D'
clip = 'Single'

if clip == 'Single':
    num_workers = 8
    bz = 32
else:
    num_workers = 4
    bz = 8

T = 36 # input clip length

'initialized params'
P, Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

modelRoot = './ModelFile/crossView_NUCLA/'

saveModel = modelRoot + clip +  '/tenc_recon_n2_mask_nopadloss/'
if not os.path.exists(saveModel):
    os.makedirs(saveModel)
print('model path:', saveModel)

num_class = 10
setup = 'setup1' # v1,v2 train, v3 test;
path_list = './data/CV/' + setup + '/'
# root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
trainSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, clip=clip, phase='train', cam='2,1', T=T,
                                setup=setup)
# #
trainloader = DataLoader(trainSet, batch_size=bz, shuffle=True, num_workers=num_workers)

testSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T, setup=setup)
testloader = DataLoader(testSet, batch_size=bz, shuffle=False, num_workers=num_workers)

net = Dyan_Autoencoder(Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, \
                    Inference=True, gpu_id=gpu_id, fistaLam=fistaLam).cuda(gpu_id)

lr = 5e-3

optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=lr, weight_decay=0.001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.4)
mseLoss = torch.nn.MSELoss()
maskedmseLoss = MaskedMSELoss()

LOSS = []
ACC = []
lam1, lam2 = 1, 10

print('Experiment config setup, clip, lam1, lam2, lr, lr_2, fistaLam: ', setup, clip, lam1, lam2, lr, fistaLam)

for epoch in range(0, Epoch + 1):

    print('start training epoch:', epoch)
    start_time = time.time()
    loss_dyan = []
    loss_inp_recon = []
    total_loss = []

    net.train()

    for i, sample in enumerate(trainloader):

        # print('sample:', i)
        optimizer.zero_grad()

        skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
        input_images = sample['input_images'].float().cuda(gpu_id)
        gt_label = sample['action'].cuda(gpu_id)
        lengths = sample['lengths'].cuda(gpu_id)

        pad_mask = net.key_padding_mask(lengths, max_len=T)
        pad_mask_expand = pad_mask.unsqueeze(2).repeat(1, 1, 50) 
        #print('pad_mask_expand ', pad_mask_expand, pad_mask_expand.shape) #(B, 36, 50)

        if clip == 'Single':
            t = skeletons.shape[1]
            input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)
        else:
            t = skeletons.shape[2]
            input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz, clip, T, 50

        dyan_out, tenc_out, tdec_out = net(input_skeletons, t, lengths)
        
        dyan_mse = mseLoss(dyan_out, tenc_out)
        input_mse = maskedmseLoss(tdec_out, input_skeletons, pad_mask_expand)

        loss = lam1 * dyan_mse + lam2 * input_mse 
        loss.backward()

        optimizer.step()
        total_loss.append(loss.data.item())
        loss_dyan.append(dyan_mse.data.item())
        loss_inp_recon.append(input_mse.data.item())

    total_loss_avg = np.mean(np.array(total_loss))
    loss_dyan_avg = np.mean(np.array(loss_dyan))
    loss_inp_recon_avg = np.mean(np.array(loss_inp_recon))

    end_time = time.time()
    time_per_epoch = (end_time - start_time)/60.0 #mins

    print('epoch: ', epoch, ' |time: ', np.round(time_per_epoch, 3), ' mins |loss:', np.round(total_loss_avg, 3), ' |dyan_mse:', np.round(loss_dyan_avg, 3), ' |inp_recon_mse:', np.round(loss_inp_recon_avg, 3))

    scheduler.step()
    if epoch % 10 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')
        dyn_recon_loss, input_recon_loss = test_reconstruct(testloader, net, gpu_id, clip)
        print('testing epoch:', epoch, ' |dyan_mse:', dyn_recon_loss, ' |inp_recon_mse:', input_recon_loss)

print('done')
