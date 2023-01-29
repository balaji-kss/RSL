from torch.utils.data import DataLoader
import torch
from dataset.crossView_UCLA import *
from modelZoo.BinaryCoding import *

N = 80*2

dataType = '2D'
sampling = 'Multi' #sampling strategy
setup = 'setup1'
num_class = 10
groupLasso = False
fusion = False

if groupLasso:
    fistaLam = 0.00
else:
    fistaLam = 0.1

T = 36
maskType = 'score'
gumbel_thresh = 0.505

if sampling == 'Single':
    num_workers = 8
    bz = 64
else:
    num_workers = 4
    bz = 12

def load_data():

    path_list = './data/CV/' + setup + '/'
    trainSet = NUCLA_CrossView(root_list=path_list, dataType=dataType, sampling=sampling, phase='train', cam='2,1,3', T=T, maskType=maskType,
                                setup=setup)

    train_loader = DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=1)

    return train_loader

def load_model(model_path):

    map_location = torch.device(gpu_id)
    state_dict = torch.load(model_path, map_location=map_location)['state_dict']

    return state_dict

def load_net(state_dict):

    #print('state_dict keys ', state_dict.keys())
    Drr = state_dict['dynamicsClassifier.backbone.sparseCoding.rr']
    Dtheta = state_dict['dynamicsClassifier.backbone.sparseCoding.theta']
    print('Drr ', Drr)
    print('Dtheta ', Dtheta)
    
    kinetics_pretrain = './pretrained/i3d_kinetics.pth'
    net = twoStreamClassification(num_class=num_class, Npole=(N + 1), num_binary=(N + 1), Drr=Drr, Dtheta=Dtheta, dim=2,
                                  gpu_id=gpu_id, inference=True, fistaLam=0.1, dataType='2D',
                                  kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

    net.load_state_dict(state_dict)
    net.eval()

    return net

def rgb_cam(X, output):
    
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()
    
    xgrad = X.grad.data.abs() #(1, 1, 36, 50)
    
    return xgrad

def remap_img(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    
    data[:, :, 0] = data[:, :, 0] * std[0] + mean[0]
    data[:, :, 1] = data[:, :, 1] * std[1] + mean[1]
    data[:, :, 2] = data[:, :, 2] * std[2] + mean[2]

    return data

def remap_rgb(data, max_val = 255):

    min_, max_ = np.min(data), np.max(data)
    data += min_
    data *= max_val/max_
    
    return np.clip(data, 0, 255)

def saliency(grad, img_seq_org, img_num, clip_num):

    img_seq = np.copy(img_seq_org)
    remap_img_grad = np.abs(grad)

    for ts in range(36):     
        inp_frame = img_seq[0, ts].transpose((1, 2, 0)).copy()
        remap_inp_frame = remap_img(inp_frame)
        remap_inp_frame = remap_rgb(remap_inp_frame)
        remap_inp_frame = cv2.cvtColor(remap_inp_frame, cv2.COLOR_BGR2RGB).astype('uint8')

        remap_img_grad[0, ts] = remap_rgb(remap_img_grad[0, ts], max_val = 512)
        grad_frame = remap_img_grad[0, ts].transpose((1, 2, 0)).astype('uint8')
        
        combine_hmap = cv2.applyColorMap((grad_frame),cv2.COLORMAP_VIRIDIS)
        disp_img = cv2.addWeighted(remap_inp_frame, 0.3, combine_hmap , 0.7, 0)
        stack_img = np.hstack((remap_inp_frame, grad_frame))
        stack_img = np.hstack((stack_img, disp_img))

        img_dir = save_dir + str(img_num) + '/' + str(clip_num) + '/'
        img_path = os.path.join(img_dir, str(ts) + '.jpg')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        cv2.imwrite(img_path, stack_img)
        cv2.imshow('stack_img ', stack_img)
        cv2.waitKey(-1)

def get_3channel(arr):
    
    arr = arr.reshape((arr.shape[0], arr.shape[1], -1))
    arr = np.concatenate((arr, arr, arr), axis=2)

    return arr

def gcam_rgb(model_path):

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
        ROIs = sample['input_rois'].float().cuda(gpu_id)
        
        t = skeletons.shape[2]
        input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
        unnorm_skeletons = unnorm_skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1)
        input_images = images.reshape(images.shape[0]*images.shape[1], t, 3, 224, 224)
        input_rois = ROIs.reshape(ROIs.shape[0]*ROIs.shape[1], t, 3, 224, 224)

        print('gt label ', gt_label)

        for j in range(4):
            
            input_skel = input_skeletons[j].reshape(1, 1, T, 50) 
            unnorm_skeleton = unnorm_skeletons[j].reshape(T, 50) 
            img_seq = input_images[j].reshape(1, T, 3, 224, 224)
            inp_roi = input_rois[j].reshape(1, T, 3, 224, 224)
            img_seq.requires_grad_()

            actPred, binaryCode, output_skeletons, lastFeat = net(input_skel, img_seq, inp_roi, fusion, bi_thresh=gumbel_thresh)

            img_seq_grad = rgb_cam(img_seq, actPred)
            print('actPred_max ', actPred)
            
            img_seq_grad = img_seq.grad.data.abs()
            img_seq_grad = img_seq_grad.cpu().detach().numpy()
            img_seq = img_seq.cpu().detach().numpy()

            saliency(img_seq_grad, img_seq, i, j)

if __name__ == '__main__':

    fixed = 1
    gpu_id = 3
    save_dir = '/home/balaji/Documents/code/RSL/Thesis/cam_2stream/results/'
    model_path = '/home/balaji/Documents/code/RSL/Thesis/cam_2stream/multi/160.pth'
    gcam_rgb(model_path)