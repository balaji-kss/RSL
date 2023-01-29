from torch.utils.data import DataLoader
import torch
from dataset.crossView_UCLA import *
from modelZoo.BinaryCoding import *

N = 80*2

dataType = '2D'
sampling = 'Multi' #sampling strategy
setup = 'setup1'

groupLasso = False

if groupLasso:
    fistaLam = 0.00
else:
    fistaLam = 0.1

T = 36
maskType = 'score'
gumbel_thresh = 0.505

def load_data():

    path_list = './data/CV/' + setup + '/'
    train_set = NUCLA_CrossView(root_list=path_list, dataType=dataType, sampling=sampling, phase='train', cam='2,1', T=T, maskType=maskType,
                                setup=setup)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1)

    return train_loader

def load_model(model_path):

    map_location = torch.device(gpu_id)
    state_dict = torch.load(model_path, map_location=map_location)['state_dict']

    return state_dict

def load_net(state_dict):

    Drr = state_dict['backbone.sparseCoding.rr']
    Dtheta = state_dict['backbone.sparseCoding.theta']
    print('Drr ', Drr)
    print('Dtheta ', Dtheta)
    
    net = contrastiveNet(dim_embed=128, Npole=N+1, Drr=Drr, Dtheta=Dtheta, Inference=True, gpu_id=gpu_id, dim=2, dataType='2D', fistaLam=fistaLam, fineTune=True).cuda(gpu_id)

    net.load_state_dict(state_dict)
    net.eval()

    return net

def dyn_cam(X, output):

    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    
    xgrad = X.grad.data.abs() #(1, 1, 36, 50)
    
    return xgrad

def get_key_joints(xgrad, unnorm_skeleton):

    xgrad = xgrad.reshape(25, 2)
    sumgrad = torch.sum(xgrad, dim=1).cpu().numpy()
    
    idxs = sumgrad.argsort()
    idxs = np.flip(idxs)
    #print(' sumgrad idxs ', sumgrad[idxs])

    return unnorm_skeleton[idxs]

def draw_circle(img, joints):

    for joint in joints:
        skeleton = (int(joint[0]), int(joint[1]))
        cv2.circle(img, center=skeleton, radius=4, color=(0,0,255), thickness=-1)

    return img

def draw_bar(img, num_joints=25):

    w = 255//num_joints
    colors = [(c, 0, 255-c) for c in range(0, 255, w)][:25]
    bw, bh = 10, 10
    ofx, ofy = 600, 75
    
    for (i, color) in enumerate(colors):

        # draw the class name + color on the legend
        x, y = ofx, ofy + i*bh
        x_, y_ = ofx + bw, ofy + (i + 1) * bh
        cx, cy = ofx + bw//2, ofy + i*bh + bh//2

        cv2.rectangle(img, (x, y), (x_, y_),
            tuple(color), -1)

    return img, colors

def display_res(inp_img, inp_skel):

    # draw heatmap of joints
    img = np.zeros((480, 640, 3), dtype='uint8')
    img, colors = draw_bar(img, num_joints=25)
    imp_img = np.zeros((480, 640, 3), dtype='uint8')
    num_imp_joints = 8

    for i, (skeleton, color) in enumerate(zip(inp_skel, colors)):
        
        skeleton = (int(skeleton[0]), int(skeleton[1]))
        cv2.circle(img, center=skeleton, radius=4, color=color, thickness=-1)

        if i < num_imp_joints:
            cv2.circle(imp_img, center=skeleton, radius=4, color=(255,255,255), thickness=-1)

    inp_img = inp_img.astype('uint8')
    overlay = cv2.addWeighted(inp_img, 0.3, img, 0.7, 0)

    return overlay, imp_img

def gcam_dyn(model_path):

    print('loading data')
    train_loader = load_data()

    print('loading model')
    state_dict = load_model(model_path)

    print('loading net')
    net = load_net(state_dict)

    for i, sample in enumerate(train_loader):

        print('sample:', i)
        
        skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
        unnorm_skeletons = sample['input_skeletons']['unNormSkeleton'].float()
        images = sample['input_images'].float().cuda(gpu_id)
        gt_label = sample['action'].cuda(gpu_id)

        # print('input skeleton shape ', skeletons.shape) # (1, 4, 36, 25, 2)
        # print('unnorm_skeleton shape ', unnorm_skeletons.shape) # (1, 4, 36, 25, 2)

        t = skeletons.shape[2]
        input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
        unnorm_skeletons = unnorm_skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1)
        input_images = images.reshape(images.shape[0]*images.shape[1], t, 3, 480, 640)
        
        # print('input skeleton shape ', input_skeletons.shape) # (4, 2, 36, 50)

        N, T, C = input_skeletons.shape
        gt_label = torch.tensor([gt_label]).cuda(gpu_id)

        for j in range(N):
            
            input_ = input_skeletons[j].reshape(1, 1, T, C) 
            unnorm_skeleton = unnorm_skeletons[j].reshape(T, C) 
            img_seq = input_images[j]

            input_.requires_grad_()
            actPred, _, _, _ = net(input_, bi_thresh=gumbel_thresh)        
            print('gt: ', gt_label)
            print('actPred idx: ', actPred.argmax())
            grad = dyn_cam(input_, actPred).squeeze()

            if fixed:
                grad = torch.sum(grad, axis=0)
                grad = grad.repeat(36, 1)

            for ts in range(t):     
                frame = img_seq[ts].cpu().numpy().transpose((1, 2, 0)).copy()
                
                grad_frame = grad[ts]
                unnorm_skeleton_frame = unnorm_skeleton[ts]
                unnorm_skeleton_frame = unnorm_skeleton_frame.cpu().numpy()
                unnorm_skeleton_frame = unnorm_skeleton_frame.reshape(25, 2)

                unnorm_skeleton_frame = get_key_joints(grad_frame, unnorm_skeleton_frame)
                
                #frame = np.clip(frame * 255.0, 0, 255.0)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype('uint8')
                overlay, key_img = display_res(frame, unnorm_skeleton_frame)

                img_dir = save_dir + str(i) + '/' + str(j) + '/'
                img_path = os.path.join(img_dir, str(ts) + '.jpg')
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                res_img = np.hstack((overlay, key_img))
                cv2.imwrite(img_path, res_img)
                cv2.imshow('overlay ', overlay)
                cv2.imshow('key_img ', key_img)
                cv2.waitKey(1)

if __name__ == '__main__':

    fixed = 1
    gpu_id = 2
    save_dir = '/home/balaji/Documents/code/RSL/Thesis/cam_dyn/results/'
    model_path = '/home/balaji/Documents/code/RSL/Thesis/cam_dyn/100.pth'
    gcam_dyn(model_path)
