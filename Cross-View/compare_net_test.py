from torch.utils.data import DataLoader
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
import matplotlib.pyplot as plt

mseLoss = torch.nn.MSELoss()

def compare_net_out(dataloader, dyan_net, tran_dyan_net, gpu_id, clip):
    count = 0
    pred_cnt = 0
    tpred_cnt = 0

    with torch.no_grad():
        for i, sample in enumerate(dataloader):

            skeleton = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)

            y = sample['action'].cuda(gpu_id)

            if clip == 'Single':
                t = skeleton.shape[1]
                input = skeleton.reshape(skeleton.shape[0], t, -1)

            else:
                t = skeleton.shape[2]
                input = skeleton.reshape(skeleton.shape[0]*skeleton.shape[1], t, -1)

            trans_inp = input.clone()   

            print('*** DYAN ***')
            label, dyan_out = dyan_net(input, t) # 'DY + CL'
            recon_loss = mseLoss(dyan_out, input).data.item()
            recon_loss = np.round(recon_loss, 5)
            pred = torch.argmax(label, 1)

            print('*** TENC + DYAN ***')
            tlabel, tdyan_out, tdyan_inp = tran_dyan_net(input, t) # 'Tenc + DY + CL'
            trecon_loss = mseLoss(tdyan_out, tdyan_inp).data.item()
            trecon_loss = np.round(trecon_loss, 5)
            tpred = torch.argmax(tlabel, 1)

            print('recon_loss: ', recon_loss, ' trecon_loss: ', trecon_loss)            
            print(' gt: ', y,' pred: ', pred, ' tpred: ', tpred)    
            
            correct = torch.eq(y, pred).int()
            tcorrect = torch.eq(y, tpred).int()
            count += y.shape[0]
            pred_cnt += torch.sum(correct).data.item()
            tpred_cnt += torch.sum(tcorrect).data.item()
            pred_avg = np.round(pred_cnt/count, 3)
            tpred_avg = np.round(tpred_cnt/count, 3)

            print('pred_avg: ', pred_avg, ' tpred_avg: ', tpred_avg)

def load_net(net, ckpt):

    stateDict = torch.load(ckpt, map_location="cuda:" + str(gpu_id))['state_dict']
    net.load_state_dict(stateDict)

    return net

if __name__ == "__main__":

    gpu_id = 1
    num_workers = 4
    'initialized params'

    N = 80*2
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    T = 36
    dataset = 'NUCLA'
    clip = 'Single'
    setup = 'setup1' # v1,v2 train, v3 test;
    path_list = './data/CV/' + setup + '/'
    testSet = NUCLA_CrossView(root_list=path_list, dataType='2D', clip='Single', phase='test', cam='2,1', T=T,
                              setup=setup)
    testloader = DataLoader(testSet, batch_size=32, shuffle=True, num_workers=num_workers)

    dyan_net = classificationWSparseCode(num_class=10, Npole=N + 1, Drr=Drr, Dtheta=Dtheta, dataType='2D', dim=2,
                                    fistaLam=0.1, gpu_id=gpu_id).cuda(gpu_id)
    ckpt = '/home/balaji/Documents/code/RSL/Thesis/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/dyan_cl/T36_fista01_openpose/50.pth'
    dyan_net = load_net(dyan_net, ckpt)
    
    transformer_dyan_net = Tenc_SparseC_Cl(num_class=10, Npole=N + 1, Drr=Drr, Dtheta=Dtheta, dataType='2D', dim=2, fistaLam=0.1, gpu_id=gpu_id).cuda(gpu_id)
    tckpt = '/home/balaji/Documents/code/RSL/Thesis/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/tenc_dyan_cl2/T36_fista01_openpose/100.pth'
    transformer_dyan_net = load_net(transformer_dyan_net, tckpt)

    compare_net_out(testloader, dyan_net, transformer_dyan_net, gpu_id, clip)