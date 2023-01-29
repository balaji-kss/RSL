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

def remap_rgb(data, range = (0, 255)):

    min_, max_ = np.min(data), np.max(data)
    data += min_
    data *= range[1]/max_

    return data

def saliency(grad, img_seq_org):

    img_seq = np.copy(img_seq_org)
    remap_img_grad = np.abs(grad)

    for ts in range(36):     
        inp_frame = img_seq[0, ts].transpose((1, 2, 0)).copy()
        remap_inp_frame = remap_img(inp_frame)
        remap_inp_frame = remap_rgb(remap_inp_frame).astype('uint8')

        remap_img_grad[0, ts] = remap_rgb(remap_img_grad[0, ts])
        grad_frame = remap_img_grad[0, ts].transpose((1, 2, 0)).astype('uint8')
    
        cv2.imshow('remap_inp_frame ', remap_inp_frame)
        cv2.imshow('grad_frame ', grad_frame)
        cv2.waitKey(-1)

def remap_img(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    data[:, :, 0] = data[:, :, 0] * std[0] + mean[0]
    data[:, :, 1] = data[:, :, 1] * std[1] + mean[1]
    data[:, :, 2] = data[:, :, 2] * std[2] + mean[2]

    return data

class GradCamModel_RGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        print('loading model')
        state_dict = load_model(model_path)

        print('loading net')
        self.net = load_net(state_dict)

        for name, param in self.net.named_parameters():
            param.requires_grad = True
            #print(name, param.data.shape)

        # self.layerhook.append(self.net.RGBClassifier.I3D_head.base_model[0].conv3d.register_forward_hook(self.forward_hook()))
        self.layerhook.append(self.net.RGBClassifier.cat.register_forward_hook(self.forward_hook()))
        
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        
        return self.gradients

    def forward_hook(self):

        def hook(module, inp, out):    
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
            
        return hook

    def forward(self, input_skel, img_seq, inp_roi, fusion):

        actPred, binaryCode, output_skeletons, lastFeat = self.net(input_skel, img_seq, inp_roi, fusion, bi_thresh=gumbel_thresh)

        # actPred, _ = self.net(img_seq, inp_roi)

        return actPred, self.selected_out

def get_3channel(arr):
    
    if len(arr.shape)>2 and arr.shape[2]==3:
        return arr

    arr = arr.reshape((arr.shape[0], arr.shape[1], -1))
    arr = np.concatenate((arr, arr, arr), axis=2)

    return arr

def add_overlay(img, labels):
    
    disp_img = np.copy(img)
    
    disp_img = get_3channel(disp_img)

    combine_hmap = np.copy(labels)
    
    combine_hmap = cv2.resize(combine_hmap, (disp_img.shape[1], disp_img.shape[0]))
    combine_hmap = np.clip(combine_hmap*255.0/np.max(combine_hmap), 0, 255)
    combine_hmap = get_3channel(combine_hmap)

    combine_hmap = cv2.applyColorMap((combine_hmap).astype('uint8'),cv2.COLORMAP_VIRIDIS)

    # disp_img = cv2.addWeighted(disp_img.astype('uint8'), 0.3, combine_hmap , 0.7,
    #     0)

    cv2.imshow('combine_hmap ', combine_hmap)
    #cv2.imshow('disp_img ', disp_img)
    cv2.imshow('vis_img ', img)
    cv2.waitKey(-1)

def overlay(inp_imgs, acts, grads):
    
    # act: 1, 832, 9, 14, 14
    # grads: 1, 832, 9, 14, 14
    # inp_imgs: 1, 36, 3, 224, 224

    print('0 ', acts.shape)
    print('1 ', grads.shape)
    # print('2 ', inp_imgs.shape)

    num_images = inp_imgs.shape[1]
    act = torch.mean(acts, 2, keepdim=True)
    grad = torch.mean(grads, 2, keepdim=True)

    act = torch.squeeze(act) #(256, 14, 14)
    grad = torch.squeeze(grad) #(256, 14, 14)

    pooled_grads = torch.mean(grad, dim=[1,2]).detach().cpu()

    # for i in range(act.shape[0]):
    #     act[i,:,:] += pooled_grads[i]
    
    heatmap = torch.mean(act, dim = 0).squeeze()
    # heatmap = torch.abs(heatmap)
    print('heatmap ', heatmap)

    for num in range(num_images):

        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        inp_img = inp_img.transpose((1, 2, 0))
        add_overlay(inp_img, heatmap)

def gcam_rgb(model_path):

    print('loading data')
    train_loader = load_data()

    gcmodel = GradCamModel_RGB().cuda(gpu_id)

    for i, sample in enumerate(train_loader):

        print('sample:', i)
        
        skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
        unnorm_skeletons = sample['input_skeletons']['unNormSkeleton'].float()
        images = sample['input_images'].float().cuda(gpu_id)
        gt_label = sample['action'].cuda(gpu_id)
        ROIs = sample['input_rois'].float().cuda(gpu_id)

        # print('input skeleton shape ', skeletons.shape) # (1, 4, 36, 25, 2)
        # print('unnorm_skeleton shape ', unnorm_skeletons.shape) # (1, 4, 36, 25, 2)
        
        t = skeletons.shape[2]
        input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
        unnorm_skeletons = unnorm_skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1)
        input_images = images.reshape(images.shape[0]*images.shape[1], t, 3, 224, 224)
        input_rois = ROIs.reshape(ROIs.shape[0]*ROIs.shape[1], t, 3, 224, 224)

        # print('input images shape ', input_images.shape) # (12*4, 36, 3, 224, 224)
        # print('input rois shape ', input_rois.shape) # (12*4, 36, 3, 224, 224)
        # print('gt_label shape ', gt_label.shape) # (12)
        # print('input skeleton shape ', input_skeletons.shape) # (12*4, 36, 50)
        print('gt label ', gt_label)

        for j in range(4):
            
            input_skel = input_skeletons[j].reshape(1, 1, T, 50) 
            unnorm_skeleton = unnorm_skeletons[j].reshape(T, 50) 
            img_seq = input_images[j].reshape(1, T, 3, 224, 224)
            inp_roi = input_rois[j].reshape(1, T, 3, 224, 224)
            img_seq.requires_grad_()

            actPred, act = gcmodel(input_skel, img_seq, inp_roi, fusion)

            output_idx = actPred.argmax()
            actPred_max = actPred[0, output_idx]
            actPred_max.backward()
            print('actPred_max ', output_idx)

            img_seq_grad = img_seq.grad.data.abs()
            img_seq = img_seq.cpu().detach().numpy()
            img_seq_grad = img_seq_grad.cpu().detach().numpy()

            saliency(img_seq_grad, img_seq)

            # act = act.detach().cpu()[0].unsqueeze(0) #[1, 256, 7, 14, 14]
            # grads = gcmodel.get_act_grads().detach().cpu()[0].unsqueeze(0)
            # overlay(img_seq, act, grads)     

if __name__ == '__main__':

    fixed = 1
    gpu_id = 3
    save_dir = '/home/balaji/Documents/code/RSL/Thesis/cam_2stream/results/'
    model_path = '/home/balaji/Documents/code/RSL/Thesis/cam_2stream/multi/160.pth'
    gcam_rgb(model_path)

    # bi_gt = torch.zeros_like(binaryCode).cuda(gpu_id)
    # target_skeletons = input_skel.reshape(input_skel.shape[0]* input_skel.shape[1], t, -1)
    # loss = lam1 * Criterion(actPred, gt_label) + lam2 * mseLoss(output_skeletons, target_skeletons) \
    #         + Alpha * L1loss(binaryCode, bi_gt)
    # loss.backward()
