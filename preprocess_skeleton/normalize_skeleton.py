import numpy as np
import matplotlib.pyplot as plt

def load_npy(fpath):

    skeletons = np.load(fpath, allow_pickle=True, encoding='latin1').item()

    return skeletons['skel_body0']

def rotation_matrix_from_vectors(vec1, vec2):

    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix

def normalize(skeleton):

    normalize_skeleton = np.copy(skeleton)[:, :2]

    ones = np.ones((normalize_skeleton.shape[0], 1))
    normalize_skeleton = np.concatenate((normalize_skeleton, ones), axis=1)
    
    #make hip center
    hip_coord = normalize_skeleton[0, :]
    normalize_skeleton[:, ] -= normalize_skeleton[0, :]

    left_hip  = normalize_skeleton[16, :]
    right_hip = normalize_skeleton[12, :]
    hip_axis = left_hip - right_hip

    vec1 = [hip_axis[0], hip_axis[1], 0]
    vec2 = [1, 0, 0]

    mat = rotation_matrix_from_vectors(vec1, vec2)
    normalize_skeleton = mat.dot(normalize_skeleton.T).T

    normalize_skeleton = np.concatenate((normalize_skeleton, skeleton[:,2].reshape((-1, 1))), axis=1)

    return normalize_skeleton

def plot_axis(skeleton, rangex, rangey, frame_num):

    xmin, xmax = rangex
    ymin, ymax = rangey

    plt.scatter(skeleton[:, 0], skeleton[:, 1], s=5)
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.text(xmax-0.5, ymax-0.1,'frame: {}'.format(frame_num))

def vis_skel(skeletons):

    num_frames = skeletons.shape[0]

    for frame_num in range(num_frames):

        plt.cla()
        skeleton = skeletons[frame_num]
        
        ##input
        plot_skeleton(skeleton, frame_num, 0)

        norm_skeleton = normalize(skeleton)
        plot_skeleton(norm_skeleton, frame_num, 1)

        plt.show()

def plot_skeleton(skeleton, frame_num, plot_num):

    xmin, xmax = np.min(skeleton[:, 0]) - 0.5, np.max(skeleton[:, 0]) + 0.5  
    ymin, ymax = np.min(skeleton[:, 1]) - 0.3, np.max(skeleton[:, 1]) + 0.3

    plt.figure(plot_num)
    draw_skeleton(skeleton)
    plot_axis(skeleton, [xmin, xmax], [ymin, ymax], frame_num)

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

    skeletons = load_npy(fpath)
    vis_skel(skeletons)