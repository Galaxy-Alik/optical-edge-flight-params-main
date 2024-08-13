import numpy as np
import cv2
import math
import json

def key_fn(item):
        return item.split('_')[2]

def get_camera_matrix(camera_data, calc_K=True):
    
    if calc_K:
        
        width = camera_data[1]
        height = camera_data[-1]
        focal = camera_data[0]
        
        # Calculate focal length in pixels
        focal_pixels = focal * max(width, height)
        
        # Calculate principal point (assuming center of image)
        cx = width / 2
        cy = height / 2
        
        # Create the camera matrix
        camera_matrix = np.array([
            [focal_pixels, 0, cx],
            [0, focal_pixels, cy],
            [0, 0, 1]
        ])
    else:
        fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
    return camera_matrix

def extract_parameters(json_file):
        
    # Load JSON data from file
    with open(json_file, 'r') as file:
        # print('------------ JSON-LOAD ------------')
        # print(json.load(file))
        # # print('------------------------')
        # exit()
        data = json.load(file)[0]

    # Extract focal, k1, k2 from cameras
    camera_params = {}
    for camera_id, camera_info in data['cameras'].items():
        camera_params[camera_id] = {
            'focal': camera_info['focal'],
            'k1': camera_info['k1'],
            'k2': camera_info['k2'],
            'w' : camera_info['width'],
            'h' : camera_info['height']
        }

    # Extract rotation (R) and translation (T) from shots
    shot_params = {}
    for shot_id, shot_info in data['shots'].items():
        shot_params[shot_id] = {
            'R': shot_info['rotation'],
            'T': shot_info['translation']
        }

    return camera_params, shot_params

def get_identity_cam(frame_ids, shot_params, K):
    # Get the first camera from the list of frame IDs
    identity_cam_id = frame_ids[0]

    # Retrieve the camera parameters (R, t, K) for the identity camera
    R = np.array(shot_params[identity_cam_id]['R']).reshape(3, 1)
    R, _ = cv2.Rodrigues(R)

    t = np.array(shot_params[identity_cam_id]['T']).reshape(3, 1)
    
    # Set the rotation matrix to identity and principal point to (0, 0)
    R = np.identity(3)
    K[0, 2] = 0  # Set ppx (principal point x) to 0
    K[1, 2] = 0  # Set ppy (principal point y) to 0

    # Create a dictionary with the identity camera parameters
    identity_cam = {
        'R': R,
        't': t,
        'K': K
    }

    return identity_cam

def undistort_image(image, K, k1, k2):

    # print('--- Undistorting Frames ---')

    h, w = image.shape[:2]
    dist_coeffs = np.array([k1, k2, 0, 0, 0])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w,h), 1, (w,h))
    dst = cv2.undistort(image, K, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst, newcameramtx
