import BVH
import Animation
from IK import IK
import numpy as np

'''load src'''
src_bvh_path = 'src1-distinct-jog-4388.bvh'
src_anim, src_jointnames, src_frmtime = BVH.load(src_bvh_path)

'''get world space positions'''
src_positions_ws = Animation.positions_global(src_anim)  # (F, J, 3) joint positions in world space

'''load target skeleton'''
dest_bvh_path = 'dest.bvh'
dest_anim, dest_jointnames, dest_frmtime = BVH.load(dest_bvh_path)

#TODO: dest_F should be src motion's length F, so expand dest_anim to F frames accordly here
dest_anim = dest_anim.repeat(len(src_anim) // len(dest_anim), axis=0)
src_anim.positions[:, 0, :]+= dest_anim.positions[0, 0, :] - src_anim.positions[0, 0, :]
dest_anim.positions[:,0,:] = src_anim.positions[:, 0, :]

# dest:src
joint_mapping = {
    'Hips': 'Hips',
    'LHipJoint': None,
    'LeftUpLeg': 'LeftHip',
    'LeftLeg': 'LeftKnee',
    'LeftFoot': 'LeftAnkle',
    'LeftToeBase': 'LeftToe',
    'LowerBack': None,
    'LeftShoulder': 'LeftCollar',
    'LeftArm': 'LeftShoulder',
    'LeftForeArm': 'LeftElbow',
    'LeftHand': 'LeftWrist',
    'LeftFingerBase': 'LeftFinger0',
    'LeftHandIndex1': None,
    'LThumb': None,
    'Neck': 'Neck',
    'Neck1': None,
    'Head': 'Head',
    'RHipJoint': None,
    'RightUpLeg': 'RightHip',
    'RightLeg': 'RightKnee',
    'RightFoot': 'RightAnkle',
    'RightToeBase': 'RightToe',
    'RightShoulder': 'RightCollar',
    'RightArm': 'RightShoulder',
    'RightForeArm': 'RightElbow',
    'RightHand': 'RightWrist',
    'RightFingerBase': 'RightFinger0',
    'RightHandIndex1': None,
    'RThumb': None,
    'Spine': 'Chest',
    'Spine1': 'Chest2',
}

for dest_joint, src_joint in joint_mapping.items():
    if src_joint in src_jointnames and dest_joint in dest_jointnames:
        dest_anim.rotations[:, dest_jointnames.index(dest_joint)] = src_anim.rotations[:, src_jointnames.index(src_joint)]

for dest_joint, src_joint in joint_mapping.items():
    if dest_joint in dest_jointnames and src_joint in src_jointnames:
        scale = (np.linalg.norm(dest_anim.offsets[dest_jointnames.index(dest_joint)]) + 0.0001) / (np.linalg.norm(src_anim.offsets[src_jointnames.index(src_joint)]) + 0.0001)
        src_anim.positions[:, src_jointnames.index(src_joint), :] *= scale
        src_anim.offsets[src_jointnames.index(src_joint)] *= scale


goal_positions = {}
for dest_joint, src_joint in joint_mapping.items():
    if dest_joint in dest_jointnames and src_joint in src_jointnames:
        goal_positions[dest_jointnames.index(dest_joint)] = Animation.positions_global(src_anim)[:, src_jointnames.index(src_joint)]
        print(goal_positions[dest_jointnames.index(dest_joint)].shape)

ik = IK(dest_anim, goal_positions, iterations=20, damping=2.0)
ik()

BVH.save('dest-distinct-jog-4388.bvh', dest_anim, dest_jointnames, dest_frmtime)