import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def load_npy(fpath):

    skeletons = np.load(fpath, allow_pickle=True, encoding='latin1').item()

    return skeletons

def load_body_model(path):

    mat = scipy.io.loadmat(path, simplify_cells=True)
    
    return mat

def rotation_matrix_from_vectors(vec1, vec2):

    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) 

def calculate_bone_lengths(skeleton, body_model):

    bone_pairs = body_model["primary_pairs"][:, 2:4]

    for bone_pair in bone_pairs:
        idx1, idx2 = bone_pair
        p1 = skeleton[:, idx1-1]
        p2 = skeleton[:, idx2-1]
        len = np.linalg.norm(p1-p2)
        print(idx1, idx2, np.round(len, 3))

def scale_invariant(skeleton, body_model):

    scale_skeleton = np.copy(skeleton).T

    scale_skeleton[[2,1], :] = scale_skeleton[[1,2], :]  ## swap y and z

    bone1_joints = body_model["primary_pairs"][:, 0:2]
    bone2_joints = body_model["primary_pairs"][:, 2:4]
    bone_lengths = body_model["bone_lengths"]

    rot_mats = compute_relative_joint_angles(scale_skeleton, bone1_joints, bone2_joints)    

    scale_skeleton = reconstruct_joint_locations(rot_mats, bone1_joints, bone2_joints, bone_lengths)

    scale_skeleton[[2,1], :] = scale_skeleton[[1,2], :]  ## swap y and z again

    return scale_skeleton.T

def compute_relative_joint_angles(norm_skeleton, bone1_joints, bone2_joints):

    num_angles = bone1_joints.shape[0]
    global_pt = np.array([1, 0, 0])
    rot_mats = []

    for i in range(num_angles):

        if bone1_joints[i, 1]:
            bone1_global = norm_skeleton[:, bone1_joints[i, 1]-1] - norm_skeleton[:, bone1_joints[i, 0]-1]
        else:
            bone1_global = global_pt.T - norm_skeleton[:, bone1_joints[i, 0]-1]

        bone2_global = norm_skeleton[:, bone2_joints[i, 1]-1] - norm_skeleton[:, bone2_joints[i, 0]-1]
        
        if np.sum(bone1_global) == 0 or np.sum(bone2_global) == 0:
            rot_mat = []
            rot_mats.append(rot_mat)

        else:
            rot_mat1 = rotation_matrix_from_vectors(bone1_global, global_pt)
            
            rot_bone1_global = rot_mat1.dot(bone1_global)
            rot_bone2_global = rot_mat1.dot(bone2_global)

            rot_mat = rotation_matrix_from_vectors(rot_bone1_global, rot_bone2_global)
            rot_mats.append(rot_mat)

    return rot_mats

def reconstruct_joint_locations(rot_mats, bone1_joints, bone2_joints, bone_lengths):

    num_angles = bone1_joints.shape[0]
    global_pt = np.array([1, 0, 0])

    for i in range(num_angles):
        if len(rot_mats[i]) == 0:
            return np.array([])

    num_joints = num_angles + 1
    joint_locations = np.zeros((3, num_joints), dtype='float')
    joint_locations[:, bone1_joints[0, 0]-1] = [0, 0, 0]
    joint_locations[:, bone2_joints[0, 1]-1] = bone_lengths[0] * rot_mats[0].dot(global_pt.T)

    for i in range(1, num_angles):
        bone1_global = joint_locations[:, bone1_joints[i, 1]-1] - joint_locations[:, bone1_joints[i, 0]-1]
        rot_mat = rotation_matrix_from_vectors(global_pt, bone1_global)
        mat = np.matmul(rot_mat, rot_mats[i])
        bone2_global = mat.dot(global_pt.T)
        joint_locations[:, bone2_joints[i,1]-1] = bone_lengths[i]*bone2_global + joint_locations[:, bone2_joints[i,0]-1]

    return joint_locations

def normalize(skeleton):

    normalize_skeleton = np.copy(skeleton)
    
    #make hip center
    hip_coord = normalize_skeleton[0, :]
    normalize_skeleton[:, :] -= normalize_skeleton[0, :]

    normalize_skeleton = scale_invariant(normalize_skeleton, body_model)

    left_hip  = normalize_skeleton[16, :]
    right_hip = normalize_skeleton[12, :]
    hip_axis = left_hip - right_hip

    vec1 = [hip_axis[0], hip_axis[1], 0]
    vec2 = [1, 0, 0]

    mat = rotation_matrix_from_vectors(vec1, vec2)
    normalize_skeleton = mat.dot(normalize_skeleton.T).T

    return normalize_skeleton

def plot_axis(skeleton, rangex, rangey, frame_num, name):

    xmin, xmax = rangex
    ymin, ymax = rangey

    plt.scatter(skeleton[:, 0], skeleton[:, 1], s=5)
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.title(name)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.text(xmax-0.5, ymax-0.1,'frame: {}'.format(frame_num))

def preprocess_skeletons(skeletons_dict, body_model):

    out_skeletons_dict = skeletons_dict.copy()
    skeletons = skeletons_dict['skel_body0']
    
    num_frames = skeletons.shape[0]
    
    norm_skeletons = np.zeros(skeletons.shape, dtype='float')

    for frame_num in range(num_frames):
        
        # plt.cla()
        skeleton = skeletons[frame_num]
        
        ##input
        plot_skeleton(skeleton, frame_num, 0, 'input skeleton')
        
        norm_skeleton = normalize(skeleton)
        norm_skeletons[frame_num] = norm_skeleton

        plot_skeleton(norm_skeleton, frame_num, 1, 'output skeleton')
        #plt.show()

    out_skeletons_dict['skel_body0'] = norm_skeletons

    return out_skeletons_dict

def preprocess_skeletons_dir(inp_dir, save_dir, body_model):

    for fname in os.listdir(inp_dir):

        npy_fpath = os.path.join(inp_dir, fname)
        save_npy_fpath = os.path.join(save_dir, fname)

        print('input path: ',npy_fpath)
        print('save path: ',save_npy_fpath)

        skeletons_dict = load_npy(npy_fpath)
        preprocess_skeletons_dict = preprocess_skeletons(skeletons_dict, body_model)
        
        np.save(save_npy_fpath, preprocess_skeletons_dict)

def check_preprocess_npys(inp_dir, save_dir):

    for inp_name, save_name in zip(os.listdir(inp_dir), os.listdir(save_dir)):

        print(inp_name, save_name)
        inp_npy_fpath = os.path.join(inp_dir, inp_name)
        inp_skeletons_dict = load_npy(inp_npy_fpath)
        inp_skeletons = inp_skeletons_dict['skel_body0']
        num_frames = inp_skeletons.shape[0]

        save_npy_fpath = os.path.join(save_dir, save_name)
        save_skeletons_dict = load_npy(save_npy_fpath)
        save_skeletons = save_skeletons_dict['skel_body0']

        for frame_num in range(0, num_frames, 10):

            inp_skeleton = inp_skeletons[frame_num]            
            plot_skeleton(inp_skeleton, frame_num, 0, 'input skeleton')

            save_skeleton = save_skeletons[frame_num]            
            plot_skeleton(save_skeleton, frame_num, 1, 'save skeleton')

            plt.show()

def plot_skeleton(skeleton, frame_num, plot_num, name):

    xmin, xmax = np.min(skeleton[:, 0]) - 0.5, np.max(skeleton[:, 0]) + 0.5  
    ymin, ymax = np.min(skeleton[:, 1]) - 0.3, np.max(skeleton[:, 1]) + 0.3

    plt.figure(plot_num)
    draw_skeleton(skeleton)
    plot_axis(skeleton, [xmin, xmax], [ymin, ymax], frame_num, name)

def draw_skeleton(skeleton_org):

    skeleton = np.copy(skeleton_org).T
    arms= np.array([24,12,11,10,9,21,5,6,7,8,22])-1 #Arms
    rightHand= np.array([12,25])-1 #one 's right hand
    leftHand= np.array([8,23])-1 #left hand
    legs= np.array([20,19,18,17,1,13,14,15,16]) - 1 #leg
    body= np.array([4,3,21,2,1]) - 1  #body
    color_bone = 'red'

    plt.plot(skeleton[0, arms], skeleton[1, arms], c=color_bone, lw=2.0) 
    plt.plot(skeleton[0, rightHand], skeleton[1, rightHand], c=color_bone, lw=2.0)
    plt.plot(skeleton[0, leftHand], skeleton[1, leftHand], c=color_bone, lw=2.0)
    plt.plot(skeleton[0, legs], skeleton[1, legs], c=color_bone, lw=2.0)
    plt.plot(skeleton[0, body], skeleton[1, body], c=color_bone, lw=2.0)

if __name__ == "__main__":

    fpath = 'S013C001P037R002A003.skeleton.npy'#'S027C001P082R001A120.skeleton.npy'
    mpath = 'body_params.mat'
    body_model = load_body_model(mpath)

    dir = '/home/ubuntu/Documents/US/NEU/RA/skeletal_action_recognition_code/data/npys/'
    save_dir = '/home/ubuntu/Documents/US/NEU/RA/skeletal_action_recognition_code/data/preprocess_npys/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preprocess_skeletons_dir(dir, save_dir, body_model)
    check_preprocess_npys(dir, save_dir)