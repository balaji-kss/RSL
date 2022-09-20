from torch.utils.data import DataLoader
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
import matplotlib.pyplot as plt


def testing(dataloader,net, gpu_id, clip):
    count = 0
    pred_cnt = 0

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

            # label,_,_ = net(input, imageData, t, fusion) # 2S
            # label,_,_ = net.dynamicsClassifier(input, t) # fusion
            # label = net(imageData) #RGB only

            label, _ = net(input, t) # 'DY + CL'
            # label, _, _ = net(input, t) # DY+BL+CL

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

        Acc = pred_cnt/count

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
    gpu_id = 6
    num_workers = 4
    'initialized params'
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    T = 36
    dataset = 'NUCLA'
    clip = 'Single'
    testSet = NUCLA_CrossView(root_list=path_list, dataType='2D', clip='Single', phase='test', cam='2,1', T=T,
                              setup=setup)
    testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=num_workers)



    net = classificationWSparseCode(num_class=10, Npole=N + 1, Drr=Drr, Dtheta=Dtheta, dataType='2D', dim=2,
                                    fistaLam=0.1, gpu_id=gpu_id).cuda(gpu_id)

    ckpt = 'path/to/models/xxx.pth'
    stateDict = torch.load(ckpt, map_location="cuda:" + str(gpu_id))['state_dict']
    net.load_state_dict(stateDict)

    Acc = testing(testloader, net, gpu_id, clip)

    print('Acc:%.4f' % Acc)