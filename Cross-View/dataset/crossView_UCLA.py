import sys
sys.path.append('../')
sys.path.append('../data')
import os
import math
import numpy as np
import pickle
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import json

#from utils import Gaussian, DrawGaussian
#from utils import DrawGaussian
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def Gaussian(sigma):
  if sigma == 7:
    return np.array([0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.2301,  0.5205,  0.8494,  1.0000,  0.8494,  0.5205,  0.2301,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529]).reshape(7, 7)
  elif sigma == n:
    return g_inp
  else:
    raise Exception('Gaussian {} Not Implement'.format(sigma))

def DrawGaussian(img, pt, sigma):
    tmpSize = int(np.math.ceil(3 * sigma))
    # if math.isnan(float(pt[0] - tmpSize)):
    #     print('NaN')
    ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
    br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]

    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img

    size = 2 * tmpSize + 1
    g = Gaussian(size)

    g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
    g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]

    img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
    img_y = [max(0, ul[1]), min(br[1], img.shape[0])]

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

"""

   Hip_Center = 1;
   Spine = 2;
   Shoulder_Center = 3;
   Head = 4;
   Shoulder_Left = 5;
   Elbow_Left = 6;
   Wrist_Left = 7;
   Hand_Left = 8;
   Shoulder_Right = 9;
   Elbow_Right = 10;
   Wrist_Right = 11;
   Hand_Right = 12;
   Hip_Left = 13;
   Knee_Left = 14;
   Ankle_Left = 15;
   Foot_Left = 16; 
   Hip_Right = 17;
   Knee_Right = 18;
   Ankle_Right = 19;
   Foot_Right = 20;
"""
def getJsonData(fileRoot, folder):
    skeleton = []
    allFiles = os.listdir(os.path.join(fileRoot, folder))
    allFiles.sort()
    usedID = []
    for i in range(0, len(allFiles)):
        with open(os.path.join(fileRoot,folder, allFiles[i])) as f:
            data = json.load(f)
        # print(len(data['people']))
        if len(data['people']) != 0:
            # print('check')
            usedID.append(i)
            temp = data['people'][0]['pose_keypoints_2d']
            pose = np.expand_dims(np.asarray(temp).reshape(25, 3)[:,0:2], 0)
            skeleton.append(pose)
        else:
            continue

    skeleton = np.concatenate((skeleton))
    return skeleton, usedID

    # return torch.tensor(skeleton).type(torch.FloatTensor)

class NUCLA_CrossView(Dataset):
    """Northeastern-UCLA Dataset Skeleton Dataset, cross view experiment,
        Access input skeleton sequence, GT label
        When T=0, it returns the whole
    """

    def __init__(self, root_list, dataType, clip, phase, cam, T, setup):
        self.root_list = root_list
        self.data_root = '/data/N-UCLA_MA_3D/multiview_action'
        self.dataType = dataType
        self.clip = clip
        if self.dataType == '2D':
            self.root_skeleton = '/data/N-UCLA_MA_3D/openpose_est'
        else:
            self.root_skeleton = '/data/N-UCLA_MA_3D/skeletons_3d'

        # self.root_list = root_list
        self.view = []

        self.phase = phase
        if setup == 'setup1':
            self.test_view = 'view_3'
        elif setup == 'setup2':
            self.test_view = 'view_2'
        else:
            self.test_view = 'view_1'
        for name_cam in cam.split(','):
            self.view.append('view_' + name_cam)
        self.T = T
        self.ds = 2
        self.clips = 6

        self.num_samples = 0

        self.num_action = 10
        self.action_list = {'a01': 0, 'a02': 1, 'a03': 2, 'a04': 3, 'a05': 4,
                            'a06': 5, 'a08': 6, 'a09': 7, 'a11': 8, 'a12': 9}
        self.actions = {'a01': 'pick up with one hand', 'a02': "pick up with two hands", 'a03': "drop trash",
                        'a04': "walk around", 'a05': "sit down",
                        'a06': "stand up", 'a08': "donning", 'a09': "doffing", 'a11': "throw", 'a12': "carry"}
        self.actionId = list(self.action_list.keys())
        # Get the list of files according to cam and phase
        # self.list_samples = []
        self.test_list = []

        # Compute the MEAN and STD of the dataset
        allSkeleton = []
        self.samples_list = []
        for view in self.view:
            #file_list = os.path.join(self.root_list, f"{view}.list")
            file_list = self.root_list + view + '.list'
            list_samples = np.loadtxt(file_list, dtype=str)
            for name_sample in list_samples:
                self.samples_list.append((view, name_sample))
        
        print('before train len ', len(self.samples_list))
        random.seed(10)
        random.shuffle(self.samples_list)
        self.samples_list = self.samples_list[: len(self.samples_list)//2]
        print('after train len ', len(self.samples_list), self.samples_list[::10])

        self.test_list= np.loadtxt(os.path.join(self.root_list, f"{self.test_view}_test.list"), dtype=str)
        temp = []
        for item in self.test_list:
            subject = item.split('_')[1]
            if subject != 's05':
                temp.append(item)

        print('before test len ', len(temp))
        random.seed(10)
        random.shuffle(temp)
        temp = temp[: len(temp)//2]
        print('after test len ', len(temp), temp[::10])

        if self.phase == 'test':
            self.samples_list = temp


    def __len__(self):
      return len(self.samples_list)
      #   return 100

    def get_uniNorm(self, skeleton):

        'skeleton: T X 25 x 2, norm[0,1], (x-min)/(max-min)'
        # nonZeroSkeleton = []
        if self.dataType == '2D':
            dim = 2
        else:
            dim = 3
        normSkeleton = np.zeros_like(skeleton)
        visibility = np.zeros(skeleton.shape)
        for i in range(0, skeleton.shape[0]):
            nonZeros = []
            ids = []
            normPose = np.zeros_like((skeleton[i]))
            for j in range(0, skeleton.shape[1]):
                point = skeleton[i,j]

                if point[0] !=0 and point[1] !=0:

                    nonZeros.append(point)
                    ids.append(j)

            nonzeros = np.concatenate((nonZeros)).reshape(len(nonZeros), dim)
            minX, minY = np.min(nonzeros[:,0]), np.min(nonzeros[:,1])
            maxX, maxY = np.max(nonzeros[:,0]), np.max(nonzeros[:,1])
            normPose[ids,0] = (nonzeros[:,0] - minX)/(maxX-minX)
            normPose[ids,1] = (nonzeros[:,1] - minY)/(maxY-minY)
            if dim == 3:
                minZ, maxZ = np.min(nonzeros[:,2]), np.max(nonzeros[:,2])
                normPose[ids,2] = (nonzeros[:,1] - minZ)/(maxZ-minZ)
            normSkeleton[i] = normPose
            visibility[i,ids] = 1

        return normSkeleton, visibility

    def pose_to_heatmap(self, poses, image_size, outRes):
        ''' Pose to Heatmap
        Argument:
            joints: T x njoints x 2
        Return:
            heatmaps: T x 64 x 64
        '''
        GaussSigma = 1

        T = poses.shape[0]
        H = image_size[0]
        W = image_size[1]
        heatmaps = []
        for t in range(0, T):
            pts = poses[t]  # njoints x 2
            out = np.zeros((pts.shape[0], outRes, outRes))

            for i in range(0, pts.shape[0]):
                pt = pts[i]
                if pt[0] == 0 and pt[1] == 0:
                    out[i] = np.zeros((outRes, outRes))
                else:
                    newPt = np.array([outRes * (pt[0] / W), outRes * (pt[1] / H)])
                    out[i] = DrawGaussian(out[i], newPt, GaussSigma)
            # out_max = np.max(out, axis=0)
            # heatmaps.append(out_max)
            heatmaps.append(out)   # heatmaps = 20x64x64
        stacked_heatmaps = np.stack(heatmaps, axis=0)
        min_offset = -1 * np.amin(stacked_heatmaps)
        stacked_heatmaps = stacked_heatmaps + min_offset
        max_value = np.amax(stacked_heatmaps)
        if max_value == 0:
            return stacked_heatmaps
        stacked_heatmaps = stacked_heatmaps / max_value

        return stacked_heatmaps

    def get_rgb(self, view, name_sample):
        data_path = os.path.join(self.data_root, view, name_sample)
        # print(data_path)
        # fileList = np.loadtxt(os.path.join(data_path, 'fileList.txt'))
        imgId = []
        # for l in fileList:
        #     imgId.append(int(l[0]))
        # imgId.sort()

        imageList = []

        for item in os.listdir(data_path):
            if item.find('_rgb.jpg') != -1:
                id = int(item.split('_')[1])
                imgId.append(id)

        imgId.sort()

        for i in range(0, len(imgId)):
            for item in os.listdir(data_path):
                if item.find('_rgb.jpg') != -1:
                    if int(item.split('_')[1]) == imgId[i]:
                        imageList.append(item)
        # imageList.sort()
        'make sure it is sorted'

        imgSize = []
        imgSequence = []
        imgSequenceOrig = []

        for i in range(0, len(imageList)):
            img_path = os.path.join(data_path, imageList[i])
            orig_image = cv2.imread(img_path)
            imgSequenceOrig.append(np.expand_dims(orig_image,0))

            input_image = Image.open(img_path)
            imgSize.append(input_image.size)

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            img_tensor = preprocess(input_image)

            imgSequence.append(img_tensor.unsqueeze(0))

        imgSequence = torch.cat((imgSequence), 0)
        imgSequenceOrig = np.concatenate((imgSequenceOrig), 0)

        return imgSequence, imgSize, imgSequenceOrig


    def paddingSeq(self, skeleton, normSkeleton, imageSequence):
        Tadd = abs(skeleton.shape[0] - self.T)

        last = np.expand_dims(skeleton[-1, :, :], 0)
        copyLast = np.repeat(last, Tadd, 0)
        skeleton_New = np.concatenate((skeleton, copyLast), 0)  # copy last frame Tadd times

        lastNorm = np.expand_dims(normSkeleton[-1, :, :], 0)
        copyLastNorm = np.repeat(lastNorm, Tadd, 0)
        normSkeleton_New = np.concatenate((normSkeleton, copyLastNorm), 0)

        lastImg = imageSequence[-1, :, :, :].unsqueeze(0)
        copyLastImg = lastImg.repeat(Tadd, 1, 1, 1)
        imageSequence_New = torch.cat((imageSequence, copyLastImg), 0)

        return skeleton_New, normSkeleton_New, imageSequence_New

    def get_data(self, view, name_sample):

        imageSequence, _, imageSequence_orig = self.get_rgb(view, name_sample)
        if self.dataType == '2D':
            skeleton, usedID = getJsonData(os.path.join(self.root_skeleton, view), name_sample)
            imageSequence = imageSequence[usedID]
        else:
            skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)
        #
        T_sample, num_joints, dim = skeleton.shape
        normSkeleton, _ = self.get_uniNorm(skeleton)

        if self.T == 0:
            skeleton_input = skeleton
            imageSequence_input = imageSequence
            normSkeleton_input = normSkeleton
            # imgSequence = np.zeros((T_sample, 3, 224, 224))
            details = {'name_sample': name_sample, 'T_sample': T_sample, 'time_offset': range(T_sample), 'view':view}
        else:
            if T_sample <= self.T:
                skeleton_input = skeleton
                normSkeleton_input = normSkeleton
                imageSequence_input = imageSequence
            else:
                # skeleton_input = skeleton[0::self.ds, :, :]
                # imageSequence_input = imageSequence[0::self.ds]

                stride = T_sample / self.T
                ids_sample = []
                for i in range(self.T):
                    id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
                    ids_sample.append(id_sample)

                skeleton_input = skeleton[ids_sample, :, :]
                imageSequence_input = imageSequence[ids_sample]
                normSkeleton_input = normSkeleton[ids_sample,:,:]

            length = skeleton_input.shape[0] 
            if skeleton_input.shape[0] != self.T:
                skeleton_input, normSkeleton_input, imageSequence_input\
                    = self.paddingSeq(skeleton_input,normSkeleton_input, imageSequence_input)

        imgSize = (640, 480)

        # normSkeleton, _ = self.get_uniNorm(skeleton_input)
        heatmap_to_use = self.pose_to_heatmap(skeleton_input, imgSize, 64)
        skeletonData = {'normSkeleton': normSkeleton_input, 'unNormSkeleton': skeleton_input}
        # print('heatsize:', heatmap_to_use.shape[0], 'imgsize:', imageSequence_input.shape[0], 'skeleton size:', normSkeleton.shape[0])
        assert heatmap_to_use.shape[0] == self.T
        assert normSkeleton_input.shape[0] == self.T
        assert imageSequence_input.shape[0] == self.T
        return heatmap_to_use, imageSequence_input, skeletonData, length

    def get_data_multiSeq(self, view, name_sample):
        overlap_rate = 0.7
        imageSequence, _, imageSequence_orig = self.get_rgb(view, name_sample)
        if self.dataType == '2D':
            skeleton, usedID = getJsonData(os.path.join(self.root_skeleton, view), name_sample)

            imageSequence = imageSequence[usedID]
        else:
            skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)

        normSkeleton, _ = self.get_uniNorm(skeleton)
        T_sample, num_joints, dim = normSkeleton.shape


        inpSkeleton_all = []
        imageSequence_input = []
        heatmap_to_use = []
        skeleton_all = []

        if T_sample <= self.T:

            skeleton_input, normSkeleton_input, imageSequence_inp = self.paddingSeq(skeleton, normSkeleton, imageSequence)

            inpSkeleton_all.append(normSkeleton_input)
            skeleton_all.append(skeleton_input)
            imageSequence_input.append(imageSequence_inp)
            heatmap_to_use.append(self.pose_to_heatmap(skeleton_input, (640, 480), 64))


        # stride = int(T_sample/self.T)
        stride = 1
        start_frame = 0
        last_frame = self.T
        overlap = int(self.T * (1-overlap_rate))
        while last_frame <= T_sample:
            start_frame = int(start_frame + overlap)
            last_frame = int(last_frame + overlap)
            selectNorm = normSkeleton[start_frame:last_frame][0::stride]
            selectSkel = skeleton[start_frame:last_frame][0::stride]

            selectImage = imageSequence[start_frame:last_frame][0::stride]

            if selectNorm.shape[0] != self.T:
                selectSkel, selectNorm, selectImage = self.paddingSeq(selectSkel, selectNorm, selectImage)


            inpSkeleton_all.append(selectNorm)
            skeleton_all.append(selectSkel)
            heatmap_to_use.append(self.pose_to_heatmap(selectSkel, (640, 480), 64))
            imageSequence_input.append(selectImage)

        skeletonData = {'normSkeleton':inpSkeleton_all, 'unNormSkeleton': skeleton_all}
        # print('seq len:',T_sample, 'clips:', len(inpSkeleton_all))
        return heatmap_to_use, imageSequence_input, skeletonData

    def __getitem__(self, index):
        """
        Return:
            skeletons: FloatTensor, [T, num_joints, 2]
            label_action: int, label for the action
            info: dict['sample_name', 'T_sample', 'time_offset']
        """

        if self.phase == 'test':
            name_sample = self.samples_list[index]
            view = self.test_view
        else:
            view, name_sample = self.samples_list[index]

        if self.clip == 'Single':
            heat_maps, images, skeletons, lengths = self.get_data(view, name_sample)

        else:
            heat_maps, images, skeletons = self.get_data_multiSeq(view, name_sample)

        label_action = self.action_list[name_sample[:3]]
        dicts = {'heat': heat_maps, 'input_images': images, 'input_skeletons': skeletons,
                 'action': label_action, 'sample_name':name_sample, 'lengths':lengths}

        return dicts


if __name__ == "__main__":
    setup = 'setup1'  # v1,v2 train, v3 test;
    path_list = '../data/CV/' + setup + '/'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLA_CrossView(root_list=path_list, dataType='2D', clip='Single', phase='train', cam='2,1', T=36,
                               setup=setup)

    # pass

    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=1)

    for i,sample in enumerate(trainloader):
        print('sample:', i)
        heatmaps = sample['heat']
        images = sample['input_images']
        inp_skeleton = sample['input_skeletons']['normSkeleton']
        label = sample['action']
        print(len(inp_skeleton), inp_skeleton.shape, label.shape)

    print('done')