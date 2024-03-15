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
        scale_factor = (np.linalg.norm(dest_anim.offsets[dest_jointnames.index(dest_joint)])+0.0001) / (np.linalg.norm(src_anim.offsets[src_jointnames.index(src_joint)])+0.0001)
        src_anim.offsets[src_jointnames.index(src_joint)] *= scale_factor
        if(src_joint=='Hips'):
            continue
        src_anim.positions[:, src_jointnames.index(src_joint), :] *= scale_factor


for src_joint in src_jointnames:
    if src_joint not in joint_mapping.values():
        src_anim.offsets[src_jointnames.index(src_joint)] *= 0
        src_anim.positions[:, src_jointnames.index(src_joint), :] *= 0


