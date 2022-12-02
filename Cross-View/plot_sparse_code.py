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

def read_recon_alltraj(txt_path, name=None):

    if name is not None:
        txt_path = txt_path + '_' + name + '.txt'

    rows = open(txt_path).read().strip().split(', ')
    rows = [float(row) for row in rows]
    rows = np.reshape(rows, (-1, T, num_traj))

    return rows 

def plot(ax, name, color, xs, ys):

    ax.set_title(name)
    #ax.scatter(xs, ys, c = color)
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
        
def get_recon_data(paths, index):


    mse_x = read_recon_alltraj(paths[0])
    mse_y = read_recon_alltraj(paths[1])
    mse_y_ = read_recon_alltraj(paths[2])
    mse_x_ = read_recon_alltraj(paths[3])
    

    mse_x = mse_x[:,:,index]
    mse_y = mse_y[:,:,index]
    mse_y_ = mse_y_[:,:,index]
    mse_x_ = mse_x_[:,:,index]
    
    return [mse_x, mse_y, mse_y_, mse_x_]

def plot_recon():

    inps1 = read_recon_alltraj(txt_path1, 'inp')
    recons1 = read_recon_alltraj(txt_path1, 'recon')

    inps2 = read_recon_alltraj(txt_path2, 'inp')
    recons2 = read_recon_alltraj(txt_path2, 'recon')

    inps3 = read_recon_alltraj(txt_path3, 'inp')
    recons3 = read_recon_alltraj(txt_path3, 'recon')

    inps4 = read_recon_alltraj(txt_path4, 'inp')
    recons4 = read_recon_alltraj(txt_path4, 'recon')

    inp1 = inps1[:,:,20]
    recon1 = recons1[:,:,20]
    inp2 = inps2[:,:,20]
    recon2 = recons2[:,:,20]

    inp3 = inps3[:,:,20]
    recon3 = recons3[:,:,20]

    inp4 = inps4[:,:,20]
    recon4 = recons4[:,:,20]

    paths0 = ['recon_traj/dim50/skel.txt', 'recon_traj/dim50/dyan_inp.txt', 'recon_traj/dim50/recon.txt', 'recon_traj/dim50/tdec_out.txt']
    recon_data0 = get_recon_data(paths0, 20)

    paths1 = ['recon_traj/mask/skel.txt', 'recon_traj/mask/dyan_inp.txt', 'recon_traj/mask/recon.txt', 'recon_traj/mask/tdec_out.txt']
    recon_data1 = get_recon_data(paths1, 20)

    xs = list(range(1, 37))

    for i in range(batch):
        # draw_skeletons(inps[i])
        fig, axs = plt.subplots(2)
        # plot(axs, 'inps1', 'blue', xs, inp1[i])
        # plot(axs, 'recons1', 'red', xs, recon1[i])
        # # # #plot(axs, 'inps2', 'green', xs, inp2[i])
        # plot(axs, 'recons2', 'green', xs, recon2[i])

        # plot(axs, 'dyan_in_cls', 'magenta', xs, inp3[i])
        # plot(axs, 'dyan_recon_cls', 'black', xs, recon3[i])

        # plot(axs, 'inps4', 'brown', xs, inp4[i])
        # plot(axs, 'recons4', 'purple', xs, recon4[i])

        plot(axs[1], 'pad loss', 'blue', xs, recon_data0[0][i])
        plot(axs[1], 'pad loss', 'red', xs, recon_data0[1][i])
        plot(axs[1], 'pad loss', 'purple', xs, recon_data0[2][i])
        plot(axs[1], 'pad loss', 'green', xs, recon_data0[3][i])
        axs[1].legend(['mse_x', 'mse_y', 'mse_y_', 'mse_x_'])

        plot(axs[0], 'no pad loss', 'blue', xs,recon_data1[0][i])
        plot(axs[0], 'no pad loss', 'red', xs, recon_data1[1][i])
        plot(axs[0], 'no pad loss', 'purple', xs, recon_data1[2][i])
        plot(axs[0], 'no pad loss', 'green', xs, recon_data1[3][i])
        

        # plt.legend(['yuexi_inps', 'yuexi_recons', 'reproduce_recons', 'tenc_out_lam0.1', 'tenc_recon_lam0.1', 'tenc_out_lam0.3', 'tenc_recon_lam0.3'])
        axs[0].legend(['mse_x', 'mse_y', 'mse_y_', 'mse_x_'])
        plt.show()

if __name__ == "__main__":

    txt_path1 = 'recon_traj/y'
    txt_path2 = 'recon_traj/dyn_cls'
    txt_path3 = 'recon_traj/tenc_lam0.1'
    txt_path4 = 'recon_traj/tenc_lam0.3'
    
    T = 36
    batch = 32
    npoles = 161
    num_traj = 50

    # plot_sparseC()
    
    plot_recon()
