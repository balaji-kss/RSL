import os, sys
import numpy as np
import random
import time
import torch
np.set_printoptions(suppress=True)
from torch import nn, load, save
from torch.optim import lr_scheduler, Adam
import torch.utils.data as dataset
from torch.utils.data import DataLoader
import gates
from synthetic_data_generator import gumbel_gen_syndata

def get_data_loader():

    train_dataset = gumbel_gen_syndata(Npole=2*N+1, num_sample=num_samples,\
                                    phase='train')
    train_dataloader = dataset.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    val_dataset = gumbel_gen_syndata(Npole=2*N+1, num_sample=num_samples,\
                                    phase='val')
    val_dataloader = dataset.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader

def get_network():

    in_channels = 2*N+1 
    network = gates.GumbelGate(in_channels)
    return network

def train():

    train_dataloader, val_dataloader = get_data_loader()
    network = get_network()
    network.cuda(gpu_id)

    loss_mse = nn.MSELoss()
    loss_bce = nn.BCELoss()

    optimizer = Adam(network.parameters(), lr=lr, weight_decay=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_drop)
    
    print('Total number of samples: ',num_samples)
    print('gpu_id: ', gpu_id)
    print('lr: ',lr)
    print('lr_steps: ',lr_steps)
    print('lr_drop: ',lr_drop)
    print('num_epochs: ',num_epochs)
    print('save dir ',save_dir)
    print('network ',network)

    for epoch in range(num_epochs):

        print('Epoch:', epoch, 'lr: ', scheduler.get_lr())
        start = time.time()
        MSE_loss = []
        BCE_loss = []

        for i, sample in enumerate(train_dataloader):

            optimizer.zero_grad()

            gt_coeff = sample['coeff'].cuda(gpu_id)
            gt_bi = sample['binary'].cuda(gpu_id)
            pred_bi = network(gt_coeff)
            loss = loss_bce(pred_bi, gt_bi)
            loss1 = loss_mse(pred_bi, gt_bi)

            loss.backward()
            optimizer.step()

            BCE_loss.append(loss.data.item())
            MSE_loss.append(loss1.data.item())

        loss_mse_np = np.mean(np.asarray(MSE_loss))
        loss_bce_np = np.mean(np.asarray(BCE_loss))

        train_mins = (time.time() - start) / 60 # 1 epoch training 
        print('epoch:', epoch, 'time: ', round(train_mins, 3), 'mins mse loss: ', round(loss_mse_np, 3), ' bce loss: ',round(loss_bce_np, 3))
        scheduler.step()

        if epoch % 2 == 0:
            save(network.state_dict(), save_dir + str(epoch) + '.pth')

def get_input_label():

    npole = 2*N+1
    nonzero_frac = random.uniform(0.2, 0.4)
    ids = [i for i in range(npole)]

    for i in range(5):
        random.shuffle(ids)

    num_nonzeros = int(nonzero_frac * npole)
    non_zero_ids = ids[:num_nonzeros]

    min_pert, max_pert = -1000, 1000
    non_zero_values = torch.randint(min_pert, max_pert, (len(non_zero_ids),), dtype=torch.float32) + torch.randn((len(non_zero_ids),), dtype=torch.float32)
    #non_zero_values = torch.randn((len(non_zero_ids)))

    ##co
    min_pert, max_pert = -1e-2, 1e-2
    #pert = torch.zeros((npole)) 
    pert = (max_pert - min_pert) * torch.rand((npole)) + min_pert
    co = torch.zeros((npole)) + pert
    co[non_zero_ids] = non_zero_values
    #co = co/1000

    ##bi
    bi = torch.zeros(co.shape)
    bi[non_zero_ids] = 1

    return co, bi

def test_model(num_images=1):

    network = get_network()

    model_path = '/home/fish/Balaji/Documents/code/RSL/gumbel/models/exp7/134.pth'
    print('model_path: ',model_path)
    network.load_state_dict(torch.load(model_path))
    network.eval()

    for i in range(num_images):

        data, label = get_input_label()
        #data = data.unsqueeze(0)
        pred = network(data).detach().numpy()

        data = np.asarray(data)
        label = np.asarray(label)
        pred =  np.asarray(pred)

        #pred = (pred>=0.5).astype('int')

        mse = ((label - pred)**2).mean()
        diff = abs(label - pred)
        
        # for param in network.parameters():
        #     print(param.name, param.data, param.data.shape, torch.mean(param.data))

        
        print('******* ',i,' ******')
        print('data: ',data)
        print('label: ',label)
        print('pred: ',pred)
        print('mse: ',mse)
        
        print('diff ids: ',np.where(diff > 0))
        print('data ids: ',data[np.where(diff > 0)])
        print('label ids: ',label[np.where(diff > 0)])
        print('pred ids: ',pred[np.where(diff > 0)])
        
        print('*******  ******')

if __name__ == "__main__":

    N = 80
    gpu_id = 1
    lr = 1e-2
    lr_steps = [30, 60, 90, 120]
    lr_drop = 0.1
    num_epochs = 150
    num_samples = 500000
    model_root = '/home/fish/Balaji/Documents/code/RSL/gumbel/models/'
    save_dir = model_root + '/exp8/'
    batch_size = 32
    num_workers = 24

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('Training ..')
    train()

    print('Testing ..')
    test_model()
    print('done')