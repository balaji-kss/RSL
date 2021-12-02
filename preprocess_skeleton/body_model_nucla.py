import numpy as np
from scipy.io import savemat

n_joints = 20
n_bones = 19

bones = [
        [1, 2],
        [2, 3],
        [3, 4],
        [3, 9],
        [3, 5],
        [9, 10],
        [5, 6],
        [10, 11],
        [6, 7],
        [11, 12],
        [7, 8],
        [1, 17],
        [1, 13],
        [17, 18],
        [13, 14],
        [18, 19],
        [14, 15],
        [19, 20],
        [15, 16]
        ]

left_hip_index = 13
right_hip_index = 17
hip_center_index = 1
n_primary_angles = 19

# based on bone joint 2
bone_lengths = [0.071, 0.107, 0.105, 0.349, 0.176, 0.223, 0.213, 0.276, 0.255, 0.222, 0.226, 0.088, 0.097, 0.504, 0.499, 0.369, 0.371, 0.096, 0.078]

def get_relative_part_pairs(bones):

    relative_part_pairs = []
    first_segment = bones[0]

    for i in range(1, len(bones)):
        pair1 = first_segment + bones[i]
        relative_part_pairs.append(pair1)
        pair2 = bones[i] + first_segment
        relative_part_pairs.append(pair2)

    return np.array(relative_part_pairs)

def get_absolute_part_pairs(bones):

    absolute_part_pairs = []
    first_segment = [1, 0]

    for i in range(len(bones)):
        pair = first_segment + bones[i]
        absolute_part_pairs.append(pair)

    return np.array(absolute_part_pairs)

def get_primary_joint_angle_pairs(bones):

    primary_pairs = []
    joint_angle_pairs = []
    primary_pairs.append([1, 0, 1, 2])
    primary_pairs.append([1, 2, 1, 17])
    primary_pairs.append([1, 2, 1, 13])

    joint_angle_pairs.append([1, 2, 1, 0])
    joint_angle_pairs.append([1, 17, 1, 2])
    joint_angle_pairs.append([1, 13, 1, 2])

    for i in range(len(bones)):
        id1, id2 = bones[i]
        for j in range(i+1, len(bones)):
            if bones[j][0] == id2:
                pair = [id2, id1] + bones[j]
                primary_pairs.append(pair)
                pairr = bones[j] + [id2, id1]
                joint_angle_pairs.append(pairr)

    joint_angle_pairs = primary_pairs + joint_angle_pairs

    return np.array(primary_pairs), np.array(joint_angle_pairs)

relative_part_pairs = get_relative_part_pairs(bones)
absolute_part_pairs = get_absolute_part_pairs(bones)
primary_pairs, joint_angle_pairs = get_primary_joint_angle_pairs(bones)

body_model_dic = {"n_joints": n_joints, "n_bones": n_bones, "bones": bones, \
                "left_hip_index":left_hip_index, "right_hip_index":right_hip_index,
                "hip_center_index":hip_center_index,
                "n_primary_angles":n_primary_angles,
                "bone_lengths":bone_lengths,
                #"relative_body_part_pairs": relative_part_pairs,
                "absolute_body_part_pairs": absolute_part_pairs,
                "primary_pairs": primary_pairs,
                "joint_angle_pairs":joint_angle_pairs
                }

savemat("body_params_nucla.mat", body_model_dic)