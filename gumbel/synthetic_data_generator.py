import numpy as np
import torch
import random
from torch.utils.data import Dataset

class gumbel_gen_syndata(Dataset):

    def __init__(self, Npole, num_sample, phase, train_frac=1.0, nonzero_frac_range=(0.3, 0.6), eps=1e-3):

        self.Npole = Npole # dimension of c 
        self.num_sample = num_sample
        self.phase = phase
        self.train_frac = train_frac
        self.nonzero_frac_range = nonzero_frac_range
        train_samples = int(self.train_frac*self.num_sample)
        self.eps = eps

        if self.eps != -1:
            min_pert, max_pert = -self.eps, self.eps
            pert = torch.zeros((train_samples, self.Npole)) 
            # pert = (max_pert - min_pert) * torch.rand((train_samples, \
            #         self. Npole)) + min_pert
        else:
            pert = torch.zeros((train_samples, self.Npole)) 

        if self.phase == 'train':
            self.coeff = torch.zeros((train_samples, self.Npole)) + pert
        else:
            val_samples = int((1-self.train_frac)*self.num_sample)
            self.coeff = torch.zeros((val_samples, self.Npole))

    def __len__(self):
        return self.coeff.shape[0]

    def __getitem__(self, idx):
        
        nonzero_frac = random.uniform(self.nonzero_frac_range[0],    self.nonzero_frac_range[1])
        ids = [i for i in range(self.Npole)]

        for i in range(5):
            random.shuffle(ids)

        num_nonzeros = int(nonzero_frac * self.Npole)
        non_zero_ids = ids[:num_nonzeros]
        
        #non_zero_values = torch.randn((len(non_zero_ids)))
        min_pert, max_pert = -1000, 1000
        non_zero_values = torch.randint(min_pert, max_pert, (len(non_zero_ids),), dtype=torch.float32) + torch.randn((len(non_zero_ids),), dtype=torch.float32)
        
        co = self.coeff[idx]
        co[non_zero_ids] = non_zero_values
        bi = torch.zeros(co.shape)
        bi[non_zero_ids] = 1
        dict = {'coeff':co, 'binary':bi}

        return dict

if __name__ == "__main__":

    num_samples = 20
    N = 10
    train_dataset = gumbel_gen_syndata(Npole=2*N+1, num_sample=num_samples, \
                                        phase='train')
    
    for i in range(10):
        dict = train_dataset.__getitem__(i)
        co, bi = dict['coeff'], dict['binary'] 
        print('co, bi ', np.round(co, 4), np.round(bi, 4))