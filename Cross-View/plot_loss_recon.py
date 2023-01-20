import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

log_path = '/home/balajisundar/Documents/US/NEU/Courses/Fall2022/Thesis/new_exps/tenc_recon_n2_dim50.log'
name = 'TENC + DYAN + TDEC'

rows = open(log_path).read().strip()

print('log_path ', log_path)

epochs = set(re.findall(r'epoch: (\d+)', rows))
epochs = sorted([int(e) for e in epochs])
print('epochs ', epochs)

str_loss = r' \|loss: (\d+\.\d+)'
total_loss = re.findall(str_loss, rows)
total_loss = [float(tl) for tl in total_loss]

str_dyan_mse = r' \|dyan_mse: (\d+\.\d+)'
dyan_mse = re.findall(str_dyan_mse, rows)
dyan_mse = [float(dm) for dm in dyan_mse]

str_inp_recon_mse = r' \|inp_recon_mse: (\d+\.\d+)'
inp_recon_mse = re.findall(str_inp_recon_mse, rows)
inp_recon_mse = [float(ir) for ir in inp_recon_mse]

str_val_dyan_mse = r' \|val_dyan_mse: (\d+\.\d+)'
val_dyan_mse = re.findall(str_val_dyan_mse, rows)
val_dyan_mse = [float(dm) for dm in val_dyan_mse]

str_val_inp_recon_mse = r' \|val_inp_recon_mse: (\d+\.\d+)'
val_inp_recon_mse = re.findall(str_val_inp_recon_mse, rows)
val_inp_recon_mse = [float(ir) for ir in val_inp_recon_mse]

#plt.plot(epochs, total_loss, color ='b', label='train total loss')
plt.plot(epochs, dyan_mse, color ='r', label='train dyan mse') 
plt.plot(epochs, inp_recon_mse, color ='g', label='train inp mse')
plt.plot(epochs[::10], val_dyan_mse, color ='purple', label='val dyan mse') 
plt.plot(epochs[::10], val_inp_recon_mse, color ='magenta', label='val inp mse')

plt.title('Train and test loss - ' + name) 
#plt.ylim([0, 0.05])
plt.xlim([0, 300])
plt.legend()
plt.show()
plt.clf()