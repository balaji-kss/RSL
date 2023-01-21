import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

log_path = '/home/balajisundar/Documents/US/NEU/Courses/Fall2022/Thesis/new_exps/exp12/cls_loss.log'
name = 'TENC + DYAN'

rows = open(log_path).read().strip()

print('log_path ', log_path)

epochs = set(re.findall(r'epoch: (\d+)', rows))
epochs = sorted([int(e) for e in epochs])
print('epochs ', epochs)

str_loss = r' \|loss: (\d+\.\d+)'
total_loss = re.findall(str_loss, rows)
total_loss = [float(tl) for tl in total_loss]

str_cls = r' \|cls: (\d+\.\d+)'
cls_loss = re.findall(str_cls, rows)
cls_loss = [float(cl) for cl in cls_loss]

str_bi = r' \|Bi: (\d+\.\d+)'
bi_loss = re.findall(str_bi, rows)
bi_loss = [float(bl) for bl in bi_loss]

str_mse = r' \|mse: (\d+\.\d+)'
mse_loss = re.findall(str_mse, rows)
mse_loss = [float(ml) for ml in mse_loss]

str_acc = r' \|acc: (\d+\.\d+)'
train_acc = re.findall(str_acc, rows)
train_acc = [float(acc) for acc in train_acc]

str_acc = r'testing epoch: \d+ Acc:(\d+\.\d+)'
test_acc = re.findall(str_acc, rows)
test_acc = [float(acc) for acc in test_acc]

print('train_acc ', train_acc)
print('test_acc ', test_acc)

plt.plot(epochs, total_loss, color ='b', label='total loss')
plt.plot(epochs, cls_loss, color ='r', label='cls loss') 
plt.plot(epochs, mse_loss, color ='g', label='mse loss') 
plt.title('Total loss - ' + name) 
#plt.ylim([0, 6])
plt.xlim([0, 200])
plt.legend()
plt.show()
plt.clf()

# plt.plot(epochs, cls_loss, color ='tab:blue') 
# plt.title('Cls loss - TENC + DYAN + CL')
# #plt.ylim([0, 20])
# plt.show()
# plt.clf()

# # plt.plot(epochs, bi_loss, color ='tab:blue') 
# # plt.title('Bi loss - DYAN + CL')
# # #plt.ylim([0, 20])
# # plt.show()
# # plt.clf()

plt.plot(epochs, mse_loss, color ='tab:blue') 
plt.title('Mse loss - ' + name)
#plt.ylim([0, 20])
plt.show()
plt.clf()

plt.plot(epochs, train_acc, color='r', label='train acc')
plt.plot(epochs[:-1:10], test_acc, color='g', label='test acc')

#plt.plot(epochs, train_acc, color ='tab:blue') 
plt.title('Train test acc - ' + name)
plt.ylim([0, 1.2])
plt.xlim([0, 200])
plt.legend()
plt.show()
plt.clf()

# for epoch in range(101):

#     print(epoch, total_loss[epoch], cls_loss[epoch], bi_loss[epoch], mse_loss[epoch])
