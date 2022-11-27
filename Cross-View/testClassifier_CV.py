from torch.utils.data import DataLoader
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
import matplotlib.pyplot as plt

mseLoss = torch.nn.MSELoss()

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def testing(dataloader, net, gpu_id, clip):
    count = 0
    pred_cnt = 0
    global_recon_loss = 0
    T = 36
    with torch.no_grad():
        for i, sample in enumerate(dataloader):

            skeleton = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
            y = sample['action'].cuda(gpu_id)
            lengths = sample['lengths']
            pad_mask = padding_mask(lengths, max_len=T).cuda(gpu_id) #(B, T)

            if clip == 'Single':
                t = skeleton.shape[1]
                input = skeleton.reshape(skeleton.shape[0], t, -1)

            else:
                t = skeleton.shape[2]
                input = skeleton.reshape(skeleton.shape[0]*skeleton.shape[1], t, -1)

            # print('input shape ', input.shape)
            #label, dyan_out = net(input, t) # 'DY + CL'
            #Edyan_inp = input

            label, dyan_out, dyan_inp = net(input, t, pad_mask) # 'Tenc + DY + CL'
            recon_loss = mseLoss(dyan_out, dyan_inp).data.item()
            global_recon_loss += recon_loss

            if clip == 'Single':
                label = label
                pred = torch.argmax(label, 1)

            else:
                num_class = label.shape[-1]
                label = label.reshape(skeleton.shape[0], skeleton.shape[1], num_class)

                label = torch.mean(label,1)
                pred = torch.argmax(label,1)

            correct = torch.eq(y, pred).int()
            count += y.shape[0]
            pred_cnt += torch.sum(correct).data.item()

            recon_loss_avg = global_recon_loss/count
            #print('recon_loss: ', np.round(recon_loss, 5), ' recon_loss_avg: ', np.round(recon_loss_avg, 5))

        Acc = pred_cnt/count
        recon_loss_avg = global_recon_loss/count
        print(' recon_loss_avg: ', np.round(recon_loss_avg, 5))
        
    return Acc

def visualize_res(dataloader, net, gpu_id, clip):

    count = 0
    pred_cnt = 0
    global_recon_loss = 0
    sparseCs = []

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

            # print('input shape ', input.shape)
            # label, sparseC, dyan_out = net(input, t) # 'DY + CL'
            # dyan_inp = input

            label, sparseC, dyan_out, dyan_inp = net(input, t) # 'Tenc + DY + CL'
            recon_loss = mseLoss(dyan_out, dyan_inp).data.item()
            global_recon_loss += recon_loss
            sc_lst = torch.flatten(sparseC[:, :, 0]).cpu().detach().numpy().tolist()
            sparseCs.append(sc_lst)

            if clip == 'Single':
                label = label
                pred = torch.argmax(label, 1)

            else:
                num_class = label.shape[-1]
                label = label.reshape(skeleton.shape[0], skeleton.shape[1], num_class)

                label = torch.mean(label,1)
                pred = torch.argmax(label,1)

            correct = torch.eq(y, pred).int()
            count += y.shape[0]
            pred_cnt += torch.sum(correct).data.item()

            recon_loss_avg = global_recon_loss/count
            #print('recon_loss: ', np.round(recon_loss, 5), ' recon_loss_avg: ', np.round(recon_loss_avg, 5))
            print('i ', i)            
            break

    with open(txt_path, 'w+') as f:
        for sparseC in sparseCs:
            scs = ", ".join([str(sc) for sc in sparseC])
            f.write(scs)
            
    Acc = pred_cnt/count
    recon_loss_avg = global_recon_loss/count
    print(' recon_loss_avg: ', np.round(recon_loss_avg, 5))
        
    return Acc

def getPlots(LOSS,LOSS_CLS, LOSS_MSE, LOSS_BI, ACC, fig_name):
    'x-axis: number of epochs'
    colors = ['#1f77b4',
              '#ff7f0e',
              '#2ca02c',
              '#d62728',
              '#9467bd',
              '#8c564b',
              '#e377c2',
              '#7f7f7f',
              '#bcbd22',
              '#17becf',
              '#1a55FF']

    fig, axs = plt.subplots(2,1)
    N = len(LOSS)
    axs[0].plot(N, LOSS, 'r-*', label='total loss')
    axs[0].plot(N, LOSS_CLS, 'b-*', label='cls loss')
    axs[0].plot(N, LOSS_MSE, 'g-*', label='mse loss')

    axs[0].set_title('loss v.s epoch')
    axs[0].set_xlabel('number of epochs')
    axs[0].set_ylabel('loss val')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(N, ACC, 'r-+', label='accuracy')
    axs[1].lagend()
    axs[1].set_xlabel('number of epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].set_title('classification accuracy v.s epoch')
    axs[1].grid(True)

    fig.tight_layout()
    fname = './figs/' + fig_name
    plt.savefig(fname)

if __name__ == "__main__":

    gpu_id = 1
    num_workers = 4
    N = 80*2
    T = 36
    num_class = 10
    transformer = 1
    dataset = 'NUCLA'
    clip = 'Single'
    setup = 'setup1' # v1,v2 train, v3 test;
    path_list = './data/CV/' + setup + '/'
    dataType = '2D'
    map_loc = "cuda:" + str(gpu_id)

    testSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T, setup=setup)
    testloader = DataLoader(testSet, batch_size=32, shuffle=False, num_workers=num_workers)

    if transformer:
        model_path = '/home/balaji/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/tenc_dyan_exp4_lam0.3_3/T36_fista01_openpose/300.pth'
    else:
        model_path = '/home/balaji/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/dyan_cl/T36_fista01_openpose/60.pth'

    txt_path = os.path.join(model_path.rsplit('/', 1)[0], 'sc.txt')
    stateDict = torch.load(model_path, map_location=map_loc)['state_dict']
    
    if transformer:
        Drr = stateDict['sparse_coding.rr'].float()
        Dtheta = stateDict['sparse_coding.theta'].float()
        net = Tenc_SparseC_Cl(num_class=num_class, Npole=N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2, fistaLam=0.3, gpu_id=gpu_id).cuda(gpu_id)
    else:
        Drr = stateDict['sparseCoding.rr'].float()
        Dtheta = stateDict['sparseCoding.theta'].float()
        net = classificationWSparseCode(num_class=10, Npole=N + 1, Drr=Drr, Dtheta=Dtheta, dataType='2D', dim=2,
                                        fistaLam=0.1, gpu_id=gpu_id).cuda(gpu_id)
                                        
    net.load_state_dict(stateDict)

    #Acc = testing(testloader, net, gpu_id, clip)
    Acc = visualize_res(testloader, net, gpu_id, clip)
    print('Acc:%.4f' % Acc)