from re import I
from cam_utils import GradCamModel_RGB, GradCamModel_DYN, load_train_data, load_model, load_net, load_kf_model
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import saliency_utils
from torchvision import transforms
np.set_printoptions(suppress=True)
from PIL import Image
import os

Alpha = 0.1
lam1 = 2
lam2 = 1
gpu_id = 0
num_class = 10
fusion = False

Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()

def stack_img(img):

    imgs = np.zeros((img.shape[0], img.shape[1], 3)).astype('float')
    imgs[:, :, 0] = img
    imgs[:, :, 1] = img
    imgs[:, :, 2] = img

    return imgs

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

    disp_img = cv2.addWeighted(disp_img.astype('uint8'), 0.3, combine_hmap , 0.7,
        0)

    cv2.imshow('disp_img ', combine_hmap)
    
    cv2.imshow('vis_img ', img)
    cv2.waitKey(-1)

def overlay(inp_imgs, acts, grads):

    # act: 1, 256, 7, 14, 14
    # grads: 1, 256, 7, 14, 14
    # inp_imgs: 1, 21, 3, 224, 224

    inp_imgs = inp_imgs.cpu().detach().numpy()
    num_images = inp_imgs.shape[1]
    act = torch.mean(acts, 2, keepdim=True)
    grad = torch.mean(grads, 2, keepdim=True)

    act = torch.squeeze(act) #(256, 14, 14)
    grad = torch.squeeze(grad) #(256, 14, 14)

    pooled_grads = torch.mean(grad, dim=[1,2]).detach().cpu()

    for i in range(act.shape[0]):
        act[i,:,:] += pooled_grads[i]

    for num in range(num_images):

        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        #print('inp_img ', inp_img.shape)
        inp_img = inp_img.transpose((1, 2, 0))
        heatmap = torch.mean(act, dim = 0).squeeze()
        #print('after inp_img ', inp_img.shape)
        #print('after heatmap ', heatmap.shape)
        add_overlay(inp_img, heatmap)

def visualize_saliency_rgb(inp_imgs, saliency):

    inp_imgs = inp_imgs.cpu().detach().numpy()
    saliency = saliency.cpu().detach().numpy()
    num_images = inp_imgs.shape[1]

    for num in range(num_images):
        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        sal = np.squeeze(saliency[:, num, :, :])
        inp_img = inp_img.transpose((1, 2, 0))
        
        #print('inp_img ', inp_img.shape)
        #print('sal ', sal.shape)
        
        sal = sal/np.max(sal)
        sal = np.clip(sal * 255.0, 0, 255.0).astype('uint8')
        cv2.imshow('inp_img ', inp_img)
        cv2.imshow('sal ', sal)
        cv2.waitKey(-1)
        

def sort_main_joints(skel_org, frame_grad_org):

    skel = np.copy(skel_org)
    frame_grad = np.copy(frame_grad_org)

    sum_max = np.sum(frame_grad, axis=1)
    #print('inp_clip ', skel)

    sum_max = sum_max.reshape((25, 1))
    skel_grads = np.concatenate((skel, sum_max), axis=1)
    #print('skel_grads shape: ', skel_grads.shape)

    idxs = skel_grads[:, -1].argsort()
    idxs = np.flip(idxs)

    #print(skel_grads[idxs])

    return idxs

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

        # cv2.putText(img, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x_, y_),
            tuple(color), -1)

    return img, colors

def get_idx_str(idxs):

    text = ""
    for idx in idxs:
        text += str(idx) + ", "
    
    return text

def display_res(inp_img, inp_skel, idxs, num_joints, resize_frac=2):

    # org joints
    img = np.zeros((480, 640, 3), dtype='uint8')
    img = cv2.resize(img, None, fx=resize_frac, fy=resize_frac)

    inp_img = inp_img.astype('uint8')
    inp_img = cv2.resize(inp_img, None, fx=resize_frac, fy=resize_frac)

    str_idxs = get_idx_str(idxs)
    cv2.putText(img, str_idxs, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    #inp_skel = reproject_skel(inp_skel)
    inp_skel[:, :] *= resize_frac

    img, colors = draw_bar(img, num_joints=25)
    for skeleton, color in zip(inp_skel, colors):
        
        skeleton = (int(skeleton[0]), int(skeleton[1]))
        #print('skel: ',skeleton)
        cv2.circle(img, center=skeleton, radius=4, color=color, thickness=-1)
        cv2.circle(inp_img, center=skeleton, radius=4, color=(0,0,255), thickness=-1)
        
    # main joints
    main_img = np.zeros((480, 640, 3), dtype='uint8')
    main_img = cv2.resize(main_img, None, fx=resize_frac, fy=resize_frac)
    inp_skels_main = inp_skel[:num_joints, :]
    # print('inp_skels_main ', inp_skels_main.shape, inp_skels_main)
    # print('inp_skel ', inp_skel.shape, inp_skel)

    for skeleton in inp_skels_main:

        skeleton = (int(skeleton[0]), int(skeleton[1]))
        cv2.circle(main_img, center=skeleton, radius=4, color=(255,255,255), thickness=-1)

    vis_img = np.hstack((img, main_img))
    
    vis_img = np.hstack((vis_img, inp_img))

    return vis_img

def visualize_saliency_joints(data, inp_imgs, inp_clips, saliency):

    inp_clips = inp_clips.cpu().detach().numpy()
    inp_imgs = inp_imgs.cpu().detach().numpy()
    saliency = saliency.cpu().detach().numpy()
    num_clips = inp_clips.shape[1]

    saliency = np.abs(saliency)
    inp_clips_unnorm = np.copy(inp_clips).reshape((inp_clips.shape[0], inp_clips.shape[1], -1, 2))    
    
    inp_clips_unnorm = data.get_unnorm(inp_clips_unnorm).unsqueeze(0).numpy()
    #print('inp_clips shape: ', inp_clips.shape) #(1, 36, 50)
    #print('saliency shape: ', saliency.shape) #(1, 36, 50)

    vis_imgs = []

    for num in range(num_clips):
        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        inp_clip_unnorm = np.squeeze(inp_clips_unnorm[:, num, :])
        frame_grad = np.squeeze(saliency[:, num, :])
        #inp_img = inp_img.transpose((1, 2, 0))

        #print('inp_img shape: ', inp_img.shape)

        frame_grad = frame_grad.reshape((-1, 2))
        inp_clip_unnorm = inp_clip_unnorm.reshape((-1, 2))
        
        idxs = sort_main_joints(inp_clip_unnorm, frame_grad)
        
        # print('sorted joint idxs: ', idxs)
        inp_clip_unnorm = inp_clip_unnorm[idxs]

        vis_img = display_res(inp_img, inp_clip_unnorm, idxs, num_joints=5, resize_frac=1)
        vis_imgs.append(vis_img)

        #print('inp_clip ', inp_clip.shape) #(50, )
        #print('sal ', sal.shape)  #(50, )
        #print('sal min max: ', np.min(sal), np.max(sal), sal)

    return vis_imgs

def vis_poles(inp_imgs, acts, grads):

    acts = acts.cpu().detach().numpy()
    grads = grads.cpu().detach().numpy()
    inp_imgs = inp_imgs.cpu().detach().numpy()
    num_images = inp_imgs.shape[1]
    
    #print('acts min max: ', np.min(acts), np.max(acts))
    #print('grads min max: ', np.min(grads), np.max(grads))

    acts = np.clip(acts*255.0, 0, 255)
    #print('acts ', acts)
    acts = acts.astype('uint8')

    for num in range(num_images):

        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        #print('inp_img ', inp_img.shape)
        inp_img = inp_img.transpose((1, 2, 0))

        cv2.imshow('inp_img ', inp_img)
        cv2.imshow('acts ', acts[0])
        cv2.waitKey(-1)

def vis_att_map_rgb():

    test_loader = load_data()
    stateDict = load_model()
    net = load_net(num_class, stateDict)

    for s, sample in enumerate(test_loader):

        'Multi'
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        input_images = sample['input_image']
        inputImage = sample['input_image'].float().cuda(gpu_id)

        t = inputSkeleton.shape[2]
        y = sample['action'].data.item()

        #print('inputSkeleton shape: ', inputSkeleton.shape)
        #print('inputImage shape: ', inputImage.shape)
        
        for i in range(0, inputSkeleton.shape[1]):
            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            inputImg_clip = inputImage[:,i, :, :, :]

            inputImg_clip.requires_grad_()
            label = net.RGBClassifier(inputImg_clip)
            y = torch.tensor([y]).cuda(gpu_id)
            saliency = saliency_utils.compute_saliency_maps_rgb(inputImg_clip, y, label)
            visualize_saliency_rgb(inputImg_clip, saliency)

def vis_att_map_dyn():

    test_loader = load_data()
    stateDict = load_model()
    net = load_net(num_class, stateDict)

    for s, sample in enumerate(test_loader):

        'Multi'
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        input_images = sample['input_image']
        inputImage = sample['input_image'].float().cuda(gpu_id)

        t = inputSkeleton.shape[2]
        y = sample['action'].data.item()

        #print('inputSkeleton shape: ', inputSkeleton.shape)
        #print('inputImage shape: ', inputImage.shape)
        
        for i in range(0, inputSkeleton.shape[1]):
            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            inputImg_clip = inputImage[:,i, :, :, :]

            input_clip.requires_grad_()
            label, _, _, _ = net(input_clip, t)
            y = torch.tensor([y]).cuda(gpu_id)
            saliency = saliency_utils.compute_saliency_maps_dyn(input_clip, y, label)
            visualize_saliency_joints(inputImg_clip, input_clip, saliency)

def get_req_frames(input_v1_skel, input_v2_skel, img1_clip, img2_clip, pad=5):

    # print('input_v1_skel shape: ', input_v1_skel.shape) #1, 26, 50
    # print('img1_clip shape: ', img1_clip.shape)    #1, 26, 480, 640, 3
    
    # print('input_v2_skel shape: ', input_v2_skel.shape) #1, 26, 50
    # print('img2_clip shape: ', img2_clip.shape)    #1, 26, 480, 640, 3

    num_frames = input_v1_skel.shape[1]
    mid = num_frames//2
    st, en = mid - pad, mid + pad
    new_num_frames = en - st
    return input_v1_skel[:, st:en, :], input_v2_skel[:, st:en, :], img1_clip[:, st:en], img2_clip[:, st:en],  new_num_frames

def vis_att_map_dyn_train_joints():

    train_dataset, train_loader = load_train_data()
    stateDict = load_model()
    net = load_net(num_class, stateDict)

    for s, sample in enumerate(train_loader):

        'Multi'
        input_v1_skels = sample['target_skeleton'].float().cuda(gpu_id)
        input_v2_skels = sample['project_skeleton'].float().cuda(gpu_id)
        input_v1_img = sample['target_image'].float().cuda(gpu_id)
        input_v2_img = sample['project_image'].float().cuda(gpu_id)

        t1 = input_v1_skels.shape[2]
        t2 = input_v2_skels.shape[2]

        y = sample['action'].data.item()
        y = torch.tensor([y]).cuda(gpu_id)

        for i in range(1):#range(0, input_v2_skels.shape[1]):

            input_v1_skel = input_v1_skels[:, i, :, :, :].reshape(1, t1, -1)
            input_v2_skel = input_v2_skels[:, i, :, :, :].reshape(1, t2, -1)
            img1_clip = input_v1_img[:,i,:,:,:]
            img2_clip = input_v2_img[:,i,:,:,:]    
            
            print('before')
            print(input_v1_skel.shape, input_v2_skel.shape)
            print(img1_clip.shape, img2_clip.shape)    

            input_v1_skel, input_v2_skel, img1_clip, img2_clip, t = get_req_frames(input_v1_skel, input_v2_skel, img1_clip, img2_clip, pad=5)
            t1 = t;  t2 = t
            
            print('after')
            print(input_v1_skel.shape, input_v2_skel.shape)
            print(img1_clip.shape, img2_clip.shape)    

            input_v1_skel.requires_grad_()
            label1, _, _, _, _ = net(input_v1_skel, t1)
            saliency1 = saliency_utils.compute_saliency_maps_dyn(input_v1_skel, y, label1)

            view1_imgs = visualize_saliency_joints(train_dataset, img1_clip, input_v1_skel, saliency1)

            input_v2_skel.requires_grad_()
            label2, _, _, _, _ = net(input_v2_skel, t2)
            saliency2 = saliency_utils.compute_saliency_maps_dyn(input_v2_skel, y, label2)
            
            view2_imgs = visualize_saliency_joints(train_dataset, img2_clip, input_v2_skel, saliency2)

            for view1_img, view2_img in zip(view1_imgs, view2_imgs):
                view_img = np.vstack((view1_img, view2_img))
                cv2.imshow('view_img ', view_img)
                cv2.waitKey(-1)
            
def get_imp_clips(saliency):

    # saliency shape (1, 26, 50)
    saliency = saliency.cpu().detach().numpy()
    saliency = np.squeeze(saliency)

    saliency = np.abs(saliency)
    
    sum_grad_frame = np.sum(saliency, axis=1)

    idxs = np.argsort(sum_grad_frame)
    idxs = np.flip(idxs)

    idxs6 = np.sort(idxs[:6])
    print('sum grad frame: ', sum_grad_frame[idxs6])
    
    return idxs

def stack_frame(inp_imgs, idxs, num, resize=0.5):
    
    h, w = inp_imgs.shape[2], inp_imgs.shape[3]

    tile_img = np.zeros((h*2, w*3, 3), dtype='uint8')
    idxs = idxs[:num]
    idxs = np.sort(idxs)
    print('idxs: ', idxs)
    str_idx = ''
    for idx in idxs:
        str_idx += str(idx) + ','

    i = 0
    for idx in idxs[:num]:
        inp_img = np.squeeze(inp_imgs[:, idx, :, :, :])
        inp_img = inp_img.astype('uint8')
        sx, sy = i%3, i//3
        tile_img[sy * h:(sy+1) * h, sx * w:(sx+1) * w] = inp_img
        i+=1

    tile_img = cv2.resize(tile_img, None, fx=resize, fy=resize)
    tile_img = cv2.putText(tile_img, str_idx, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 0, 0), 2, cv2.LINE_AA)

    return tile_img

def visualize_imp_clips(img_clips, idxs, name, num):

    inp_imgs = img_clips.cpu().detach().numpy()
    
    # for i in range(len(idxs)):
    #     inp_img = np.squeeze(inp_imgs[:, i, :, :, :])
    #     inp_img = inp_img.astype('uint8')
    #     cv2.imshow('imp img ', inp_img)
    #     cv2.waitKey(3)

    # for idx in idxs[:num]:
    #     inp_img = np.squeeze(inp_imgs[:, idx, :, :, :])
    #     inp_img = inp_img.astype('uint8')
    #     cv2.imshow('imp img ', inp_img)
    #     cv2.waitKey(-1)

    tile_img = stack_frame(inp_imgs, idxs, num)
    cv2.imshow(name, tile_img)
    cv2.waitKey(-1)
    return tile_img

def preprocess_kf_input(input_data):

    preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    input_data_np = input_data.cpu().detach().numpy().astype('uint8')
    preprocess_data = []
    #print('input_data_np shape: ', input_data_np.shape, type(input_data_np))

    nframes = input_data_np.shape[0]

    for fid in range(nframes):
        im = Image.fromarray(input_data_np[fid])
        img_tensor = preprocess(im)
        preprocess_data.append(img_tensor.unsqueeze(0))
    
    preprocess_data = torch.cat((preprocess_data), 0).cuda(gpu_id)

    return preprocess_data

def squeeze_data(input_data):

    if len(input_data.shape) == 5:
        input_data = input_data.squeeze(0)
    else:
        input_data = input_data

    return input_data

def transpose_input(input_data):

    return torch.permute(input_data, (0, 2, 3, 1))

def dup_last_frame(input_data, tn=40):

    dup_num = tn - input_data.shape[0]
    last_frame = input_data[-1]
    dup_last_frame = last_frame.repeat(dup_num, 1, 1, 1)
    return torch.cat((input_data, dup_last_frame), 0)

def vis_att_map_dyn_train_clips(model_path, transformer=False, iskf=False):

    train_dataset, train_loader = load_train_data()
    stateDict = load_model(model_path)
    net = load_net(num_class, stateDict, transformer)
    kfnet = load_kf_model()
    alpha = 3

    for s, sample in enumerate(train_loader):

        'Multi'
        input_v1_skels = sample['target_skeleton'].float().cuda(gpu_id)
        input_v2_skels = sample['project_skeleton'].float().cuda(gpu_id)
        input_v1_img = sample['target_image'].float().cuda(gpu_id)
        input_v2_img = sample['project_image'].float().cuda(gpu_id)

        t1 = input_v1_skels.shape[2]
        t2 = input_v2_skels.shape[2]

        y = sample['action'].data.item()
        y = torch.tensor([y]).cuda(gpu_id)

        for i in range(1):#range(0, input_v2_skels.shape[1]):

            input_v1_skel = input_v1_skels[:, i, :, :, :].reshape(1, t1, -1)
            input_v2_skel = input_v2_skels[:, i, :, :, :].reshape(1, t2, -1)
            img1_clip = input_v1_img[:,i,:,:,:]
            img2_clip = input_v2_img[:,i,:,:,:]    

            ## Cross View

            ## View 1  
            input_v1_skel.requires_grad_()
            label1, _, _, _, _ = net(input_v1_skel, t1)
            saliency1 = saliency_utils.compute_saliency_maps_dyn(input_v1_skel, y, label1)
            clip_idxs1 = get_imp_clips(saliency1)
            print('view1 imp clip indices: ', clip_idxs1, len(clip_idxs1))
            tile_img1 = visualize_imp_clips(img1_clip, clip_idxs1,  name='cvv1', num=6)
            # save_path1 = os.path.join(out_dir, str(s) + '_1.jpg')
            # cv2.imwrite(save_path1, tile_img1)

            ## View 2
            input_v2_skel.requires_grad_()
            label2, _, _, _, _ = net(input_v2_skel, t2)
            saliency2 = saliency_utils.compute_saliency_maps_dyn(input_v2_skel, y, label2)
            clip_idxs2 = get_imp_clips(saliency2)
            print('view2 imp clip indices: ', clip_idxs2, len(clip_idxs2))
            tile_img2 = visualize_imp_clips(img2_clip, clip_idxs2, name='cvv2', num=6)
            # save_path2 = os.path.join(out_dir, str(s) + '_2.jpg')
            # cv2.imwrite(save_path2, tile_img2)

            ## Key frame
            if iskf:
                ## View 1 
                vis_img1_clip = torch.clone(img1_clip) 
                vis_img2_clip = torch.clone(img2_clip) 

                #print('bf img1_clip shape: ', img1_clip.shape)
                img1_clip = squeeze_data(img1_clip)
                img1_clip = preprocess_kf_input(img1_clip)
                img1_clip = dup_last_frame(img1_clip)
                #print('af img1_clip shape: ', img1_clip.shape)
                sparseCode_key, Dictionary, keylist_to_pred, keylist_FRA1, key_list, imgFeature = kfnet.get_keylist(img1_clip, alpha) 
                print('view1 keyframe indices: ', keylist_FRA1, len(keylist_FRA1))
                visualize_imp_clips(vis_img1_clip, keylist_FRA1, name='kfv1', num=6)

                ## View 2
                img2_clip = squeeze_data(img2_clip)
                img2_clip = preprocess_kf_input(img2_clip)
                img2_clip = dup_last_frame(img2_clip)
                sparseCode_key, Dictionary, keylist_to_pred, keylist_FRA2, key_list, imgFeature = kfnet.get_keylist(img2_clip, alpha) 
                print('view2 keyframe indices: ', keylist_FRA2, len(keylist_FRA2))
                img2_clip = transpose_input(img2_clip)
                img2_clip = img2_clip.unsqueeze(0)
                visualize_imp_clips(vis_img2_clip, keylist_FRA2, name='kfv2', num=6)

if __name__ == '__main__':

    #vis_att_map_rgb()
    #vis_att_map_dyn()
    #vis_att_map_dyn_train_joints()

    transformer = True
    out_dir = '/home/balaji/Documents/code/RSL/NewCV/Cross-View/results/1110/'
    if transformer:
        model_path = '/home/balaji/Documents/code/RSL/NewCV/Cross-View/crossViewModel/NUCLA/freeze_tenc_full/100.pth'
    else:
        model_path = '/home/balaji/Documents/code/RSL/NewCV/Cross-View/crossViewModel/NUCLA/1110/dynamicStream_fista01_reWeighted_noBI_sqrC_T36_UCLA/100.pth'
    
    vis_att_map_dyn_train_clips(model_path, transformer)