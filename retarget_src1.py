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

'''set goal positions
goal_positions[j] are joint j's goal positions
goal_positions[j] is of shape (F, 3), i.e. F number of frames, each frame is (x,y,z)
'''
goal_positions = {}
for j in range(0, dest_anim.shape[1]):
    mapped_j_src = j #TODO: make your own map between src and target
    goal_positions[j] =  src_positions_ws[0:dest_F, mapped_j_src]
    print('goal_positions[k]', mapped_j_src, goal_positions[j].shape)


'''reach goals through IK'''
ik = IK(dest_anim, goal_positions, iterations=20, damping=2.0)
ik()

BVH.save(dest_bvh_path + '_retargeted_result.bvh', dest_anim, dest_jointnames, dest_frmtime)

