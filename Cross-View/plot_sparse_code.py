import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def read_sc_all_traj(txt_path):
    txt_path = txt_path + '_sc.txt'
    rows = open(txt_path).read().strip().split(', ')
    rows = [float(row) for row in rows]
    rows = np.reshape(rows, (-1, npoles, num_traj))
    rows = rows[:, :, 20]
    
    return rows

def read_sc_one_traj(txt_path):

    txt_path = txt_path + '_sc.txt'
    rows = open(txt_path).read().strip().split(', ')
    rows = [float(row) for row in rows]
    rows = np.reshape(rows, (-1, npoles))
    
    return rows

def read_recon_alltraj(txt_path, name=None, num_traj=50):

    if name is not None:
        txt_path = txt_path + '_' + name + '.txt'

    rows = open(txt_path).read().strip().split(', ')
    rows = [float(row) for row in rows]
    rows = np.reshape(rows, (-1, T, num_traj))

    return rows 

def plot(ax, color, xs, ys):

    ax.plot(xs, ys, c = color)
    ax.set_ylim([-3.0, 3.0])

def plot_sparseC():

    sc1 = read_sc_all_traj(txt_path1)
    sc2 = read_sc_all_traj(txt_path2)
    sc3 = read_sc_one_traj(txt_path3)
    sc4 = read_sc_all_traj(txt_path4)
    xs = list(range(1, 162))

    # for i in range(batch):
        
    #     fig, axs = plt.subplots(2)
    #     sns.heatmap(sc1[i], ax=axs[0], vmin=-0.2, vmax=1).set(title='Yuexi sparse code lam = 0.1')
    #     sns.heatmap(sc2[i], ax=axs[1], vmin=-0.2, vmax=1).set(title='sparse code lam = 0.1')
    #     # sns.heatmap(sc3[i], ax=axs[2], vmin=-0.2, vmax=1).set(title='sparse code lam = 0.3')
    #     # sns.heatmap(sc4[i], ax=axs[3], vmin=-0.2, vmax=1).set(title='sparse code lam = 0.5 - 200 epoch')
    #     plt.show()

    for i in range(batch):
        
        fig, axs = plt.subplots(1)
        plot(axs, 'yuexi', 'blue', xs, sc1[i])
        plot(axs, 'yuexi', 'red', xs, sc2[i])
        plot(axs, 'yuexi', 'green', xs, sc3[i])
        #plot(axs[1], 'Sparce code traj 0 dyn + cl lam=0.1', 'red', xs, sc2[i])
        plt.show()

def draw_skeletons(skeletons_org):

    skeletons = skeletons_org.reshape((36, 25, 2))
    plt.title('Correct Plot:\nBut uses to many lines to unpack li')

    for frame in range(T):
        skeleton = skeletons[frame]
        sx = skeleton[:, 0].tolist()
        sy = (2 - skeleton[:, 1]).tolist()
        plt.scatter(sx, sy)
        plt.xlim([-2, 2])
        for i, (x, y) in enumerate(zip(sx, sy)):
            plt.text(x, y, i+1, color="red", fontsize=12)
        
def get_recon_data(paths, index1, index2=None, embed_traj=50):

    if index2 is None:
        index2 = index1

    mse_x = read_recon_alltraj(paths[0])
    mse_y = read_recon_alltraj(paths[1], num_traj=embed_traj)
    mse_y_ = read_recon_alltraj(paths[2], num_traj=embed_traj)
    mse_x_ = read_recon_alltraj(paths[3])

    print('index1: ', index1, ' index2: ', index2)
    print('mse_x ', mse_x.shape)
    print('mse_y ', mse_y.shape)
    print('mse_y_ ', mse_y_.shape)
    print('mse_x_ ', mse_x_.shape)

    mse_x = mse_x[:,:,index1]
    mse_y = mse_y[:,:,index2]
    mse_y_ = mse_y_[:,:,index2]
    mse_x_ = mse_x_[:,:,index1]

    return [mse_x, mse_y, mse_y_, mse_x_]

def get_sc_data(paths, index1, index2=None, embed_traj=50):

    if index2 is None:
        index2 = index1

    mse_x = read_recon_alltraj(paths[0])
    mse_y = read_recon_alltraj(paths[1], num_traj=embed_traj)
    mse_y_ = read_recon_alltraj(paths[2], num_traj=embed_traj)

    print('index1: ', index1, ' index2: ', index2)
    print('mse_x ', mse_x.shape)
    print('mse_y ', mse_y.shape)
    print('mse_y_ ', mse_y_.shape)

    mse_x = mse_x[:,:,index1]
    mse_y = mse_y[:,:,index2]
    mse_y_ = mse_y_[:,:,index2]

    return [mse_x, mse_y, mse_y_]

def plot_recon():

    traj_index = 20
    paths0 = ['recon_traj/dim50/skel.txt', 'recon_traj/dim50/dyan_inp.txt', 'recon_traj/dim50/recon.txt', 'recon_traj/dim50/tdec_out.txt']
    recon_data0 = get_recon_data(paths0, traj_index, embed_traj=50)

    paths1 = ['recon_traj/dim100/skel.txt', 'recon_traj/dim100/dyan_inp.txt', 'recon_traj/dim100/recon.txt', 'recon_traj/dim100/tdec_out.txt']
    recon_data1 = get_recon_data(paths1, traj_index, 2 * traj_index, embed_traj=100)
    recon_data11 = get_recon_data(paths1, traj_index, 2 * traj_index + 1, embed_traj=100)

    paths2 = ['recon_traj/dim25/skel.txt', 'recon_traj/dim25/dyan_inp.txt', 'recon_traj/dim25/recon.txt', 'recon_traj/dim25/tdec_out.txt']
    recon_data2 = get_recon_data(paths2, traj_index, traj_index//2, embed_traj=25)

    paths3 = ['tenc_recon_bi_loss/skel.txt', 'tenc_recon_bi_loss/dyan_inp.txt', 'tenc_recon_bi_loss/recon.txt', 'tenc_recon_bi_loss/tdec_out.txt']
    recon_data3 = get_recon_data(paths3, traj_index, embed_traj=50)

    xs = list(range(1, 37))

    for i in range(batch):
        # draw_skeletons(inps[i])
        fig, axs = plt.subplots(2)

        axs[1].set_title('dim50 recon')
        plot(axs[2], 'blue', xs, recon_data0[0][i])
        plot(axs[2], 'red', xs, recon_data0[1][i])
        plot(axs[2], 'purple', xs, recon_data0[2][i])
        plot(axs[2], 'green', xs, recon_data0[3][i])
        axs[2].legend(['mse_x', 'mse_y', 'mse_y_', 'mse_x_'])

        axs[0].set_title('dim50 recon bi')
        plot(axs[0], 'blue', xs, recon_data3[0][i])
        plot(axs[0], 'red', xs, recon_data3[1][i])
        plot(axs[0], 'purple', xs, recon_data3[2][i])
        plot(axs[0], 'green', xs, recon_data3[3][i])
        axs[0].legend(['mse_x', 'mse_y', 'mse_y_', 'mse_x_'])
        
        # axs[0].set_title('dim100')    
        # plot(axs[0], 'blue', xs,recon_data1[0][i])
        # plot(axs[0], 'red', xs, recon_data1[1][i])
        # plot(axs[0], 'purple', xs, recon_data1[2][i])
        # plot(axs[0], 'magenta', xs, recon_data11[1][i])
        # plot(axs[0], 'black', xs, recon_data11[2][i])
        # avg_recon = 0.5 * (recon_data1[2][i] + recon_data11[2][i])
        # plot(axs[0], 'brown', xs, avg_recon)
        # plot(axs[0], 'green', xs, recon_data1[3][i])
        # axs[0].legend(['mse_x', 'mse_y0', 'mse_y_0', 'mse_y1', 'mse_y_1', 'avg_mse_y_', 'mse_x_'])

        # axs[2].set_title('dim50 cls')
        # plot(axs[2], 'blue', xs, recon_data2[0][i])
        # plot(axs[2], 'red', xs, recon_data2[1][i])
        # plot(axs[2], 'purple', xs, recon_data2[2][i])
        # plot(axs[2], 'green', xs, recon_data2[3][i])
        # axs[2].legend(['mse_x', 'mse_y', 'mse_y_', 'mse_x_'])
        
        plt.show()

def plot_recon_sc():

    traj_index = 20
    paths0 = ['recon_traj/dim50/skel.txt', 'recon_traj/dim50/dyan_inp.txt', 'recon_traj/dim50/recon.txt', 'recon_traj/dim50/tdec_out.txt']
    recon_data0 = get_recon_data(paths0, traj_index, embed_traj=50)

    paths1 = ['tenc_recon_bi_loss/skel.txt', 'tenc_recon_bi_loss/dyan_inp.txt', 'tenc_recon_bi_loss/recon.txt', 'tenc_recon_bi_loss/tdec_out.txt']
    recon_data1 = get_recon_data(paths1, traj_index, embed_traj=50)
    # paths1 = ['recon_traj/dim50/trskel.txt', 'recon_traj/dim50/trdyan_inp.txt', 'recon_traj/dim50/trrecon.txt', 'recon_traj/dim50/trtdec_out.txt']
    # recon_data1 = get_recon_data(paths1, traj_index, embed_traj=50)

    paths2 = ['exp9/skel.txt', 'exp9/dyan_inp.txt', 'exp9/recon.txt']
    recon_data2 = get_sc_data(paths2, traj_index, traj_index, embed_traj=50)

    #paths3 = ['exp9/trskel.txt', 'exp9/trdyan_inp.txt', 'exp9/trrecon.txt']
    paths3 = ['exp11/skel.txt', 'exp11/dyan_inp.txt', 'exp11/recon.txt']
    recon_data3 = get_sc_data(paths3, traj_index, traj_index, embed_traj=50)

    xs = list(range(1, 37))

    for i in range(batch):
        # draw_skeletons(inps[i])
        fig, axs = plt.subplots(2)

        axs[0].set_title('dim50 w/o bi')
        plot(axs[0], 'blue', xs, recon_data0[0][i])
        plot(axs[0], 'red', xs, recon_data0[1][i])
        plot(axs[0], 'purple', xs, recon_data0[2][i])
        plot(axs[0], 'black', xs, recon_data2[1][i])
        plot(axs[0], 'magenta', xs, recon_data2[2][i])
        axs[0].legend(['mse_x', 'recon_y', 'recon_y_', 'cls_y', 'cls_y_'])

        axs[1].set_title('dim50 w bi')
        plot(axs[1], 'blue', xs, recon_data1[0][i])
        plot(axs[1], 'red', xs, recon_data1[1][i])
        plot(axs[1], 'purple', xs, recon_data1[2][i])
        plot(axs[1], 'black', xs, recon_data3[1][i])
        plot(axs[1], 'magenta', xs, recon_data3[2][i])
        axs[1].legend(['mse_x', 'recon_y', 'recon_y_', 'cls_y', 'cls_y_'])
        
        plt.show()

if __name__ == "__main__":

    txt_path1 = 'recon_traj/y'
    txt_path2 = 'recon_traj/dyn_cls'
    txt_path3 = 'recon_traj/tenc_lam0.1'
    txt_path4 = 'recon_traj/tenc_lam0.3'
    
    T = 36
    batch = 100
    npoles = 161
    num_traj = 50

    # plot_sparseC()
    
    plot_recon_sc()

    # plot_recon()