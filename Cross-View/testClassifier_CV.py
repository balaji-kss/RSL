from torch.utils.data import DataLoader
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
import matplotlib.pyplot as plt

mseLoss = torch.nn.MSELoss()

def testing(dataloader, net, gpu_id, clip):
    count = 0
    pred_cnt = 0
    global_recon_loss = 0
    T = 36

    net.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):

            skeleton = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
            y = sample['action'].cuda(gpu_id)
            lengths = sample['lengths'].cuda(gpu_id)

            if clip == 'Single':
                t = skeleton.shape[1]
                input = skeleton.reshape(skeleton.shape[0], t, -1)

            else:
                t = skeleton.shape[2]
                input = skeleton.reshape(skeleton.shape[0]*skeleton.shape[1], t, -1)
		

            label, recon = net(input, t) # 'DY + CL'
            dyan_inp = input

            # label, bi, recon, dyan_inp = net(input, t, lengths) # 'Tenc + DY + CL'
            recon_loss = mseLoss(recon, dyan_inp).data.item()
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

def visualize_cls(dataloader, net, gpu_id, clip):

    count = 0
    pred_cnt = 0
    global_recon_loss = 0
    sparseCs, recons, dyan_inps, net_inps = [], [], [], []

    net.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):

            skeleton = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
        
            y = sample['action'].cuda(gpu_id)
            lengths = sample['lengths'].cuda(gpu_id)

            if clip == 'Single':
                t = skeleton.shape[1]
                input = skeleton.reshape(skeleton.shape[0], t, -1)

            else:
                t = skeleton.shape[2]
                input = skeleton.reshape(skeleton.shape[0]*skeleton.shape[1], t, -1)

            # print('input shape ', input.shape)
            # label, sparseC, dyan_out = net(input, t) # 'DY + CL'
            # dyan_inp = input

            label, sparseC, recon, dyan_inp = net(input, t, lengths) # 'Tenc + DY + CL
            
            recon_loss = mseLoss(recon, dyan_inp).data.item()
            global_recon_loss += recon_loss

            sc_lst = torch.flatten(sparseC[:, :, :]).cpu().detach().numpy().tolist()
            recon_lst = torch.flatten(recon[:, :, :]).cpu().detach().numpy().tolist()
            dyan_inp_lst = torch.flatten(dyan_inp[:, :, :]).cpu().detach().numpy().tolist()
            inp_lst = torch.flatten(input[:, :, :]).cpu().detach().numpy().tolist()
            
            sparseCs.append(sc_lst)
            recons.append(recon_lst)
            dyan_inps.append(dyan_inp_lst)
            net_inps.append(inp_lst)

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

    write_lst(sc_txt_path, sparseCs)
    write_lst(y_txt_path, recons)
    write_lst(ytxt_path, dyan_inps)
    write_lst(xtxt_path, net_inps)
    
    Acc = pred_cnt/count
    recon_loss_avg = global_recon_loss/count
    print(' recon_loss_avg: ', np.round(recon_loss_avg, 5))
        
    return Acc

def write_lst(txt_path, lsts):

    with open(txt_path, 'w+') as f:
        for i, lst in enumerate(lsts):
            els = ", ".join([str(el) for el in lst])
            if i<len(lsts) - 1:
                els += ", "
            f.write(els)

def test_reconstruct(dataloader, net, gpu_id, clip):

    count = 0
    global_input_loss, global_dyan_loss = 0.0,  0.0

    net.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):

            skeleton = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
        
            y = sample['action'].cuda(gpu_id)
            lengths = sample['lengths'].cuda(gpu_id)

            if clip == 'Single':
                t = skeleton.shape[1]
                input = skeleton.reshape(skeleton.shape[0], t, -1)

            else:
                t = skeleton.shape[2]
                input = skeleton.reshape(skeleton.shape[0]*skeleton.shape[1], t, -1)

            recon, bi, dyan_inp, tdec_out = net(input, t, lengths)
            
            dyan_mse = mseLoss(recon, dyan_inp).data.item()
            input_mse = mseLoss(tdec_out, input).data.item()

            global_input_loss += input_mse
            global_dyan_loss += dyan_mse

            count += y.shape[0]

    input_loss_avg = global_input_loss/count
    dyan_loss_avg = global_dyan_loss/count

    input_loss_avg = np.round(input_loss_avg, 6)
    dyan_loss_avg = np.round(dyan_loss_avg, )
    
    return dyan_loss_avg, input_loss_avg

def visualize_reconstruct(dataloader, net, gpu_id, clip):

    count = 0
    global_input_loss = 0
    global_dyan_loss = 0
    xs, ys, x_s, y_s = [], [], [], []

    net.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):

            skeleton = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
        
            y = sample['action'].cuda(gpu_id)
            lengths = sample['lengths'].cuda(gpu_id)

            if clip == 'Single':
                t = skeleton.shape[1]
                input = skeleton.reshape(skeleton.shape[0], t, -1)

            else:
                t = skeleton.shape[2]
                input = skeleton.reshape(skeleton.shape[0]*skeleton.shape[1], t, -1)

            recon, dyan_inp, tdec_out = net(input, t, lengths)
            
            dyan_mse = mseLoss(recon, dyan_inp).data.item()
            input_mse = mseLoss(tdec_out, input).data.item()

            global_input_loss += input_mse
            global_dyan_loss += dyan_mse

            xlst = torch.flatten(input).cpu().detach().numpy().tolist()
            ylst = torch.flatten(dyan_inp).cpu().detach().numpy().tolist()
            y_lst = torch.flatten(recon).cpu().detach().numpy().tolist()
            x_lst = torch.flatten(tdec_out).cpu().detach().numpy().tolist()
            
            xs.append(xlst)
            ys.append(ylst)
            y_s.append(y_lst)
            x_s.append(x_lst)

            count += y.shape[0]
            print('i ', i)            
            #break

    write_lst(xtxt_path, xs)
    write_lst(ytxt_path, ys)
    write_lst(x_txt_path, x_s)
    write_lst(y_txt_path, y_s)

    input_loss_avg = global_input_loss/count
    dyan_loss_avg = global_dyan_loss/count
    print(' input_loss_avg: ', input_loss_avg)
    print(' dyan_loss_avg: ', dyan_loss_avg)

    return 

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

    gpu_id = 0
    num_workers = 4
    N = 80*2
    T = 36
    num_class = 10
    transformer = 0
    recon = 1
    dataset = 'NUCLA'
    clip = 'Single'
    setup = 'setup1' # v1,v2 train, v3 test;
    path_list = './data/CV/' + setup + '/'
    dataType = '2D'
    map_loc = "cuda:" + str(gpu_id)

    testSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T, setup=setup)
    testloader = DataLoader(testSet, batch_size=32, shuffle=False, num_workers=num_workers)

    if recon:
        model_path = '/home/balaji/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/tenc_recon_n2_bi/300.pth'
    elif transformer:
        # model_path = '/home/balaji/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/tenc_exp9_dim50/T36_fista01_openpose/300.pth'
        model_path = '/home/balaji/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/sc_tenc_dyanf_exp10/T36_fista01_openpose/30.pth'

    else:
        model_path = '/home/balaji/RSL/Cross-View/ModelFile/crossView_NUCLA/Single/dyan_cl/T36_fista01_openpose/60.pth'

    sc_txt_path = os.path.join(model_path.rsplit('/', 1)[0], 'trsc.txt')
    xtxt_path = os.path.join(model_path.rsplit('/', 1)[0], 'trskel.txt')
    ytxt_path = os.path.join(model_path.rsplit('/', 1)[0], 'trdyan_inp.txt')
    y_txt_path = os.path.join(model_path.rsplit('/', 1)[0], 'trrecon.txt')
    x_txt_path = os.path.join(model_path.rsplit('/', 1)[0], 'trtdec_out.txt')
    stateDict = torch.load(model_path, map_location=map_loc)['state_dict']

    if recon:
        Drr = stateDict['sparse_coding.rr'].float()
        Dtheta = stateDict['sparse_coding.theta'].float()
        net = Dyan_Autoencoder(Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, \
                    Inference=True, gpu_id=gpu_id, fistaLam=0.1, is_binary=True).cuda(gpu_id)
    elif transformer:
        Drr = stateDict['sparseCoding.rr'].float()
        Dtheta = stateDict['sparseCoding.theta'].float()
        print('Drr ', Drr)
        print('Dtheta ', Dtheta)
        # net = Tenc_SparseC_Cl(num_class=num_class, Npole=N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2, fistaLam=0.1, gpu_id=gpu_id).cuda(gpu_id)
        net = Dyan_Tenc(num_class=num_class, Npole=N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2, fistaLam=fistaLam, gpu_id=gpu_id).cuda(gpu_id)
    else:
        Drr = stateDict['sparseCoding.rr'].float()
        Dtheta = stateDict['sparseCoding.theta'].float()
        net = classificationWSparseCode(num_class=10, Npole=N + 1, Drr=Drr, Dtheta=Dtheta, dataType='2D', dim=2,
                                        fistaLam=0.1, gpu_id=gpu_id).cuda(gpu_id)
                                        
    net.load_state_dict(stateDict)

    #Acc = testing(testloader, net, gpu_id, clip)
    # Acc = visualize_cls(testloader, net, gpu_id, clip)
    # print('Acc:%.4f' % Acc)
    visualize_reconstruct(testloader, net, gpu_id, clip)
    
