import math
import numpy as np

def compute_ate(gtruth_file, pred_file):
    gtruth_list = read_file_list(gtruth_file)
    pred_list = read_file_list(pred_file)
    matches = associate(gtruth_list, pred_list, 0, 0.01)
    if len(matches) < 2:
        return False

    gtruth_xyz = np.array([[float(value) for value in gtruth_list[a][0:3]] for a,b in matches])
    pred_xyz = np.array([[float(value) for value in pred_list[b][0:3]] for a,b in matches])
    
    offset = gtruth_xyz[0] - pred_xyz[0]
    pred_xyz += offset[None,:]
    scale = np.sum(gtruth_xyz * pred_xyz)/np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2))/len(matches)
    return rmse

def read_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list,offset,max_difference):
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

def rot2quat(R):
    rz, ry, rx = mat2euler(R)
    qw, qx, qy, qz = euler2quat(rz, ry, rx)
    return rz, ry, rx, qw, qx, qy, qz

def quat2mat(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def mat2euler(M, cy_thresh=None, seq='zyx'):

    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if seq=='zyx':
        if cy > cy_thresh: # cos(y) not close to zero, standard form
            z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21,  r22)
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = 0.0
    elif seq=='xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi/2
                x = atan2(r12, r13)
            else:
                y = -np.pi/2
    else:
        raise Exception('Sequence not recognized')
    return z, y, x

import functools
def euler2mat(z=0, y=0, x=0, isRadian=True):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
         Rotation matrix giving same rotation as for given angles
    '''

    if not isRadian:
        z = ((np.pi)/180.) * z
        y = ((np.pi)/180.) * y
        x = ((np.pi)/180.) * x
    assert z>=(-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
    assert y>=(-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
    assert x>=(-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x    

    Ms = []
    if z:
            cosz = math.cos(z)
            sinz = math.sin(z)
            Ms.append(np.array(
                            [[cosz, -sinz, 0],
                             [sinz, cosz, 0],
                             [0, 0, 1]]))
    if y:
            cosy = math.cos(y)
            siny = math.sin(y)
            Ms.append(np.array(
                            [[cosy, 0, siny],
                             [0, 1, 0],
                             [-siny, 0, cosy]]))
    if x:
            cosx = math.cos(x)
            sinx = math.sin(x)
            Ms.append(np.array(
                            [[1, 0, 0],
                             [0, cosx, -sinx],
                             [0, sinx, cosx]]))
    if Ms:
            return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)

def euler2quat(z=0, y=0, x=0, isRadian=True):
    ''' Return quaternion corresponding to these Euler angles
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    quat : array shape (4,)
         Quaternion in w, x, y z (real, then vector) format
    '''
  
    if not isRadian:
        z = ((np.pi)/180.) * z
        y = ((np.pi)/180.) * y
        x = ((np.pi)/180.) * x
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
                     cx*cy*cz - sx*sy*sz,
                     cx*sy*sz + cy*cz*sx,
                     cx*cz*sy - sx*cy*sz,
                     cx*cy*sz + sx*cz*sy])

def pose_vec_to_mat(vec):
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3,1))
    rot = euler2mat(vec[5], vec[4], vec[3])
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat

def pose_vec_q_to_mat(vec):
    tx = vec[1]
    ty = vec[2]
    tz = vec[3]
    trans = np.array([tx, ty, tz]).reshape((3,1))
    q = [vec[7],vec[4], vec[5], vec[6]]
    rot = quat2mat(q)
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat

# convert a 4*4 mat to vec with q
def pose_mat_to_vec_q(pose_mat):
    tx = pose_mat[0, 3]
    ty = pose_mat[1, 3]
    tz = pose_mat[2, 3]
    rot = pose_mat[:3, :3]
    qw, qx, qy, qz = rot2quat(rot)

    # set time as zero for now
    time = 0
    return [time, tx, ty, tz, qx, qy, qz, qw]

def dump_pose_seq_TUM(out_file, poses, times):
    # First frame as the origin
    first_pose = pose_vec_to_mat(poses[0])
    with open(out_file, 'w') as f:
        for p in range(len(times)):
            this_pose = pose_vec_to_mat(poses[p])
            this_pose = np.dot(first_pose, np.linalg.inv(this_pose))
            tx = this_pose[0, 3]
            ty = this_pose[1, 3]
            tz = this_pose[2, 3]
            rot = this_pose[:3, :3]
            qw, qx, qy, qz = rot2quat(rot)
            f.write('%f %f %f %f %f %f %f %f\n' % (times[p], tx, ty, tz, qx, qy, qz, qw))
