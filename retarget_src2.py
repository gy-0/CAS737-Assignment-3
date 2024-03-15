import BVH
import Animation
from IK import IK
import numpy as np

'''load src'''
src_bvh_path = 'src2-mixamo-walkTurn.bvh'
src_anim, src_jointnames, src_frmtime = BVH.load(src_bvh_path)

'''get world space positions'''
src_positions_ws = Animation.positions_global(src_anim)  # (F, J, 3) joint positions in world space

'''load target skeleton'''
dest_bvh_path = 'dest.bvh'
dest_anim, dest_jointnames, dest_frmtime = BVH.load(dest_bvh_path)

#TODO: dest_F should be src motion's length F, so expand dest_anim to F frames accordly here
dest_anim = dest_anim.repeat(len(src_anim) // len(dest_anim), axis=0)
dest_frmtime=src_frmtime


# Set position
src_anim.positions[:, 0, :] += dest_anim.positions[0, 0, :] - src_anim.positions[0, 0, :]
dest_anim.positions[:, 0, :] = src_anim.positions[:, 0, :]

joint_mapping = {
    'Hips': 'Hips',
    'Head': None,
    'LeftShoulder': 'LeftShoulder',
    'LeftArm': 'LeftArm',
    'LeftForeArm': 'LeftForeArm',
    'LeftHand': 'LeftHand',
    'LeftFingerBase': None,
    'LeftHandIndex1': 'LeftHandIndex1',
    'LThumb': 'LeftHandThumb1',
    'LHipJoint': None,
    'LeftUpLeg': 'LeftUpLeg',
    'LeftLeg': 'LeftLeg',
    'LeftFoot': 'LeftFoot',
    'LeftToeBase': 'LeftToeBase',
    'RightShoulder': 'RightShoulder',
    'RightArm': 'RightArm',
    'RightForeArm': 'RightForeArm',
    'RightHand': 'RightHand',
    'RightFingerBase': None,
    'RightHandIndex1': 'RightHandIndex1',
    'RThumb': 'RightHandThumb1',
    'RHipJoint': None,
    'RightUpLeg': 'RightUpLeg',
    'RightLeg': 'RightLeg',
    'RightFoot': 'RightFoot',
    'RightToeBase': 'RightToeBase',
    'LowerBack': 'Spine',
    'Spine': 'Spine1',
    'Spine1': 'Spine2',
    'Neck': 'Neck',
    'Neck1': 'Head',
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