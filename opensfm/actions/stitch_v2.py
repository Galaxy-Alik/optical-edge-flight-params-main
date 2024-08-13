import numpy as np
import cv2 as cv
import math
import json
import os
from pathlib import Path
from PIL import Image
from pprint import pprint
import matplotlib.pyplot as plt
import argparse
import tqdm

from .stitch import bl_stitch
from .Extras.Utils import *
from .stitch_helper import *
from .Extras.sourcer import *

# from stitch import bl_stitch
# from Extras.Utils import *
# from stitch_helper import *
# from Extras.sourcer import *

import tifffile
import random
import string
import logging

logger: logging.Logger = logging.getLogger(__name__)

def key_fn(item):
    return item.split('_')[2]

def get_data(src_folder, is_save = False):

    logger.info(' --------- Retreieving Vectorized Information --------- ')
    pprint(' --------- Retreieving Vectorized Information --------- ')
    
    parent_file = r"/home/datademon/Desktop/Alik/galax_spip_v2/data"
    parent_file = os.path.join(parent_file, src_folder)
    
    json_file = os.path.join(parent_file, r"reconstruction.json")
    # Extract parameters
    camera_params, shot_params = extract_parameters(json_file)

    int_params = list(list(camera_params.items())[-1])[-1]
    focal, k1, k2, w, h = int_params['focal'], int_params['k1'], int_params['k2'], int_params['w'], int_params['h']

    camera_data = [focal, w, h]
    c_matrix = get_camera_matrix(camera_data, calc_K = True)
    
    frame_ids = [frame_id for frame_id, params in shot_params.items()]

    # frame_ids = sorted(frame_ids, key = key_fn, reverse = False)

    img_data_path = os.path.join(parent_file, r"images")

    resized_frames, img_path_list = [], []

    resize_shape = (512, 512, 3)

    for i in range(len(frame_ids)):
        path = os.path.join(img_data_path, frame_ids[i])
        img = np.asarray(Image.open(path)) 
        # img = cv2.resize(img, resize_shape[:2])
        resized_frames.append(img)

    num_frames = len(resized_frames)
    print('Num-Selected-Frames: ', num_frames, ', Size: ', resized_frames[0].shape)
    
    if is_save:
        selected_frame_path = os.path.join(parent_file, 'selected_frames')
        if not os.path.isdir(selected_frame_path):
            os.makedirs(selected_frame_path)
        for i in range(len(resized_frames)):
            img_path =  os.path.join(selected_frame_path, f'frame_{i}.jpg')
            cv.imwrite(img_path, cv.cvtColor(resized_frames[i], cv.COLOR_BGR2RGB))
        
        print(' ------ Sampled-Images-Successfully-Saved ------ ')

    return resized_frames, frame_ids, c_matrix, k1, k2, shot_params

def run_through_cv2(src_folder):
    
    '''
     Stitch all camera images into a single image
    '''
    
    print(' -------- Bundle - Adjustment - Implementation -------- ')

    ## Selects ONLY subset of useful images
    resized_frames, frame_ids, K, k1, k2, shot_params = get_data(src_folder, is_save = True)

    flag = 'no_order'
    match_count = 0
    
    ### supply (image, R, t, K)
    for i in tqdm(range(len(frame_ids) - 1), desc="Stitching frames (OPT)", unit="frame"): ## Frame_ids not of the same length of images folder

        print()
        print('##########################################################')
        print(f'Frame_Ids_{i} : ', frame_ids[i], f' --->  Frame_Ids_{i + 1} : ', frame_ids[i + 1])
        print('##########################################################')
        print()

        img1_match_dict = source_matches(frame_ids[i], src_folder)
        
        if i == 0:

            result_img = resized_frames[i]
            # curr_R, curr_t = np.array(shot_params[frame_ids[i]]['R']).reshape(3, 1), np.array(shot_params[frame_ids[i]]['T']).reshape(3, 1)
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0]).astype('float64')

        else:

            img1 = result_img
            img2 = resized_frames[i]
        
            h, w = resized_frames[i].shape[:2]
            # _, new_K = undistort_image(resized_frames[i], K, k1, k2)

            R, t = np.array(shot_params[frame_ids[i]]['R']).reshape(3, 1), np.array(shot_params[frame_ids[i]]['T']).reshape(3, 1)
            R, _ = cv.Rodrigues(R)

            if i == 1:
                    curr_R = R
                    curr_t = t
                    prev_R = curr_R
                    prev_t = abs(curr_t)
            else:
                    curr_R = np.matmul(prev_R, R)   
                    curr_t = (np.matmul(prev_R, t)) + (prev_t)

            matches = source_matches_v2(frame_ids[i], frame_ids[i + 1], src_folder)
            
            # matches_dict = source_matches(frame_ids[i], src_folder)
            
            # if frame_ids[i+1] not in list(matches_dict.keys()):
            #      matches = None
            # else:
            #      matches = matches_dict[frame_ids[i+1]]

            # print('************ MATCHES **************')
            # print(matches)
            # print('**************************')

            # ## skip for no matches
            # if matches == None:
            #      continue 
            
            ## Use Cv2 matcher
            if type(matches) is type(None):

                flag = 'order2'
                feat = FeatureLocalization(img1, img2)
                
                baseImage_kp, baseImage_des, secImage_kp, secImage_des = feat.feature_detection()
                
                # baseImage_kp, baseImage_des = source_kp(frame_ids[i], src_folder)
                # secImage_kp, secImage_des = source_kp(frame_ids[i + 1], src_folder)
                # baseImage_kp, secImage_kp = baseImage_kp[:, :2], secImage_kp[:, :2]
                
                matches = feat.feature_match(baseImage_des, secImage_des)

            else:

                match_count += 1
                flag = 'no_order'
                baseImage_kp, baseImage_des = source_kp(frame_ids[i], src_folder)
                secImage_kp, secImage_des = source_kp(frame_ids[i + 1], src_folder)
                baseImage_kp, secImage_kp = baseImage_kp[:, :2], secImage_kp[:, :2]

            # homographyMatrix = findHomography_v3(prev_R, prev_t, curr_R, curr_t, camera_matrix)
            homographyMatrix, _ = findHomography_v2(matches, baseImage_kp, secImage_kp, curr_R, curr_t, camera_matrix, flag = flag)
            # homographyMatrix, _ = findHomography_v2(matches_cv2, baseImage_kp_cv2, secImage_kp_cv2, curr_R, curr_t, camera_matrix, flag = 'order2')
            # homographyMatrix, _ = findHomography_v2(matches_intr, baseImage_kp_intr, secImage_kp_intr, curr_R, curr_t, camera_matrix, flag = 'order1')

            print()
            print('************************************')
            print(' --- Homography-Matrix ---')
            print(homographyMatrix, type(homographyMatrix))
            print()
            print('************************************')
            
            # exit()

            newFrameSize, correction, homographyMatrix = getNewFrameSizeAndMatrix(homographyMatrix, img2.shape[:2], img1.shape[:2])
            result_img = cv2.warpPerspective(img2, homographyMatrix, (newFrameSize[1], newFrameSize[0]))
            result_img[correction[1]:correction[1] + img1.shape[0], correction[0]:correction[0] + img1.shape[1]] = img1

            prev_R = curr_R
            prev_t = abs(curr_t)

        pass
    
    print(' ---------------------------- ')
    print('Matches-Found: ', match_count)
    print()
    print('Matches-NOT-Found', len(resized_frames) - match_count)
    print(' ---------------------------- ')
    
    return result_img

def run_through_both(src_folder):
    
    print(' -------- Bundle - Adjustment - Implementation -------- ')

    ## Selects ONLY subset of useful images
    resized_frames, frame_ids, K, k1, k2, shot_params = get_data(src_folder)
    
    ### supply (image, R, t, K)
    for i in tqdm(range(len(frame_ids) - 1), desc="Stitching frames (OPT)", unit="frame"): ## Frame_ids not of the same length of images folder

        print()
        print('##########################################################')
        print(f'Frame_Ids_{i} : ', frame_ids[i], f' --->  Frame_Ids_{i + 1} : ', frame_ids[i + 1])
        print('##########################################################')
        print()

        img1_match_dict = source_matches(frame_ids[i], src_folder)
        is_present = frame_ids[i + 1] in list(img1_match_dict.keys())

        if i == 0:

            result_img = resized_frames[i]
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0]).astype('float64')

        else:

            img1 = result_img
            img2 = resized_frames[i]
        
            h, w = resized_frames[i].shape[:2]
            # _, new_K = undistort_image(resized_frames[i], K, k1, k2)

            R, t = np.array(shot_params[frame_ids[i]]['R']).reshape(3, 1), np.array(shot_params[frame_ids[i]]['T']).reshape(3, 1)
            R, _ = cv.Rodrigues(R)

            if i == 1:
                    curr_R = R
                    curr_t = t
            else:
                    curr_R = np.matmul(prev_R, R)   
                    curr_t = (np.matmul(prev_R, t)) + (prev_t)
                
            if not is_present:
                
                logger.info(f' ------ Feature-Matches-Regeneration for {frame_ids[i]} --> {frame_ids[i + 1]} ------ ')
                print(f' ------ Feature-Matches-Regeneration for {frame_ids[i]} --> {frame_ids[i + 1]} ------ ')
                
                ''' CV2 based feature-matching '''
                
                flag = 'order2'
                feat = FeatureLocalization(img1, img2)
                baseImage_kp, baseImage_des, secImage_kp, secImage_des = feat.feature_detection()
                matches = feat.feature_match(baseImage_des, secImage_des)

            else:
                
                logger.info(f' ------ Extracted-Feature-Matches for {frame_ids[i]} --> {frame_ids[i + 1]} ------ ')
                print(f' ------ Extracted-Feature-Matches for {frame_ids[i]} --> {frame_ids[i + 1]} ------ ')
                
                ''' Sourcer-based f-matching'''
                
                flag = 'order1'
                # flag = 'no_order'
                feat = FeatureLocalization(img1, img2)
                baseImage_kp, baseImage_des = source_kp(frame_ids[i], src_folder)
                secImage_kp, secImage_des = source_kp(frame_ids[i + 1], src_folder)
                baseImage_kp, secImage_kp = baseImage_kp[:, :2], secImage_kp[:, :2]

                matches = feat.feature_match(baseImage_des, secImage_des)
                # matches = img1_match_dict[frame_ids[i + 1]]
                
            homographyMatrix, _ = findHomography_v2(matches, baseImage_kp, secImage_kp, curr_R, curr_t, camera_matrix, flag = flag)
        
            print()
            print('************************************')
            print('Homography-Matrix:')
            print(homographyMatrix, type(homographyMatrix))
            print()
            print('************************************')
            
            # exit()

            newFrameSize, correction, homographyMatrix = getNewFrameSizeAndMatrix(homographyMatrix, img2.shape[:2], img1.shape[:2])
            result_img = cv2.warpPerspective(img2, homographyMatrix, (newFrameSize[1], newFrameSize[0]))
            result_img[correction[1]:correction[1] + img1.shape[0], correction[0]:correction[0] + img1.shape[1]] = img1

            prev_R = curr_R
            prev_t = abs(curr_t)

        pass

    return result_img


def run_through_cv2_no_order(src_folder):
    
    '''
     Stitch all camera images into a single image - without ordering
    '''
    
    parent_file = r"/home/datademon/Desktop/Alik/galax_spip_v2/data"
    parent_file = os.path.join(parent_file, src_folder)
    img_data_path = os.path.join(parent_file, r"images")

    print(' -------- Bundle - Adjustment - Implementation -------- ')

    ## Selects ONLY subset of useful images
    resized_frames, frame_ids, K, k1, k2, shot_params = get_data(src_folder) ## not ordered
    
    ### supply (image, R, t, K)
    for i in tqdm(range(len(frame_ids) - 1), desc="Stitching frames (OPT)", unit="frame"): ## Frame_ids not of the same length of images folder

        # print('CAM: ', i)
        print()
        print('##########################################################')
        print(f'Frame_Ids_{i} : ', frame_ids[i], f' --->  Frame_Ids_{i + 1} : ', frame_ids[i + 1])
        print('##########################################################')
        print()

        img1_match_dict = source_matches(frame_ids[i], src_folder)

        print()
        print(' ------------ Key-Present ------------ ')
        print(img1_match_dict)
        print()
        print(f'{i}. --> {frame_ids[i + 1] in list(img1_match_dict.keys())}')
        print()


        if i == 0:

            result_img = resized_frames[i]
            # curr_R, curr_t = np.array(shot_params[frame_ids[i]]['R']).reshape(3, 1), np.array(shot_params[frame_ids[i]]['T']).reshape(3, 1)
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0]).astype('float64')

        else: 

            h, w = resized_frames[i].shape[:2]
            # _, new_K = undistort_image(resized_frames[i], K, k1, k2)

            R, t = np.array(shot_params[frame_ids[i]]['R']).reshape(3, 1), np.array(shot_params[frame_ids[i]]['T']).reshape(3, 1)
            R, _ = cv.Rodrigues(R)

            img1 = resized_frames[i]
    
            baseImage_kp, _ = source_kp(frame_ids[i], src_folder)
            baseImage_kp = baseImage_kp[:, :2]

            print("******************************************")

            continue


            secImage_kps = []
            match_indices = []
            img2_list = []

            if i == 1:
                    curr_R = R
                    curr_t = t
            else:
                    curr_R = np.matmul(prev_R, R)   
                    curr_t = (np.matmul(prev_R, t)) + (prev_t)
                
            for id in list(img1_match_dict.keys()):
                print('Matched-ID: ', id)
                match_idx = img1_match_dict[id]
                path = os.path.join(img_data_path, id)
                img2_list.append(np.asarray(Image.open(path))) 
                secImage_kps.append(source_kp(id, src_folder))
                match_indices.append(match_idx)        

            print(f'---------- num(Matches) for {frame_ids[i]} is {len(match_indices)} ----------')
            print()

            # exit()

            ''' Won't work because no (R, t) information about intermediate /matches dict frames is present '''

            for i in range(len(match_indices)):

                img2 = img2_list[i]
                secImage_kp = secImage_kps[i][0][:, :2]
                match = match_indices[i]

                # print(secImage_kp, match)

                homographyMatrix, _ = findHomography_v2(match, baseImage_kp, secImage_kp, curr_R, curr_R, camera_matrix, flag = 'no_order')
                # homographyMatrix, _ = findHomography_v2(matches, baseImage_kp, secImage_kp, R, t, camera_matrix)
                
                # print()
                # print('************************************')
                # print(' --- Homography-Matrix ---')
                # print(homographyMatrix, type(homographyMatrix))
                # print()
                # print('************************************')
                
                # exit()

                newFrameSize, correction, homographyMatrix = getNewFrameSizeAndMatrix(homographyMatrix, img2.shape[:2], img1.shape[:2])
                result_img = cv2.warpPerspective(img2, homographyMatrix, (newFrameSize[1], newFrameSize[0]))
                result_img[correction[1]:correction[1] + img1.shape[0], correction[0]:correction[0] + img1.shape[1]] = img1

            prev_R = curr_R
            prev_t = abs(curr_t)

        pass

    return result_img

def run_basic(src_folder):
    
    '''
    Stitch all camera images into a single image
    '''
    resized_frames, frame_ids, K, k1, k2, shot_params = get_data(src_folder)

    # Get identity image (used as ref frame)
    identity_cam = get_identity_cam(frame_ids, shot_params, K)
    
    # For each non-identity image, calculate transform
    offsets = {}

    x_min_best = 9_000_000_000
    y_min_best = 9_000_000_000
    x_max_best = -9_000_000_000
    y_max_best = -9_000_000_000
    
    ### supply (image, R, t, K)
    for i, id in enumerate(frame_ids):

        print('CAM: ', i)
        h, w = resized_frames[i].shape[:2]
        # _, new_K = undistort_image(resized_frames[i], K, k1, k2)
        
        R, t = np.array(shot_params[frame_ids[i]]['R']).reshape(3, 1), np.array(shot_params[frame_ids[i]]['T']).reshape(3, 1)
        R, _ = cv.Rodrigues(R)

        if i == 0:
                curr_R = R
                curr_t = t
        else:
                curr_R = np.matmul(prev_R, R)   
                curr_t = (np.matmul(prev_R, t)) + (prev_t)
                
        pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1, 1, 2)
        H = identity_cam['K'] @ identity_cam['R'] @ R.T @ np.linalg.pinv(K)
        transformed_corners = cv.perspectiveTransform(pts, H)

        prev_R = curr_R
        prev_t = abs(curr_t)

        [x_min, y_min] = np.int32(transformed_corners.min(axis=0).ravel()) # x,y
        [x_max, y_max] = np.int32(transformed_corners.max(axis=0).ravel()) # x,y

        if (x_min < x_min_best):
            x_min_best = x_min

        if (y_min < y_min_best):
            y_min_best = y_min

        if (x_max > x_max_best):
            x_max_best = x_max
        
        if (y_max > y_max_best):
            y_max_best = y_max

        offsets[i] = H

    im_x_0 = x_min_best
    im_y_0 = y_min_best
    results = []

    for idx in range(len(frame_ids) - 1):

        H = offsets[idx]
        im_x_shift = -im_x_0
        im_y_shift = -im_y_0

        Ht = np.array([
        [1,0,im_x_shift],
        [0,1,im_y_shift],
        [0,0,1]])
        
        # ap-1
        result = cv.warpPerspective(resized_frames[i],  Ht @ H, (x_max_best - x_min_best, y_max_best - y_min_best))
        results.append(result)

        # ap-2
        # img1, img2 = resized_frames[idx], resized_frames[idx+1]
        # newFrameSize, correction, homographyMatrix = getNewFrameSizeAndMatrix(Ht @ H, img2.shape[:2], img1.shape[:2], is_direct=True)
        # stitchedFrame = cv2.warpPerspective(img2, homographyMatrix, (newFrameSize[1], newFrameSize[0]))
        # print(' ---- Stitched -- Frame ---- ')
        # print(stitchedFrame.shape)
        # print()
        # stitchedFrame[correction[1]:correction[1] + img1.shape[0], correction[0]:correction[0] + img1.shape[1]] = img1
        # results.append(stitchedFrame)

    final_img = results[0]


    print('IMG: ', final_img.shape)
    print()
    print()
    print('IMG-List: ', [f.shape for f in results])
    print()

    for res in results[1:]:
        rows, cols, _ = res.shape
        res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(res_gray, 0, 255, cv.THRESH_BINARY)
        
        mask_inv = cv.bitwise_not(mask)

        final_img_bg = cv.bitwise_and(final_img, final_img, mask=mask_inv)
        res_img_fg = cv.bitwise_and(res, res, mask=mask)

        dst = cv.add(final_img_bg, res_img_fg)
        final_img[0:rows,0:cols] = dst

    final_img[np.where((final_img==[0,0,0]).all(axis=2))] = [0, 0, 0]
    
    return final_img

def main_plot_save(src_folder, main = False):
        
    b_less_result_img = bl_stitch(src_folder=src_folder).run()
    
    # b_result_img = run_basic(src_folder=src_folder)
    b_result_img = run_through_cv2(src_folder=src_folder)
    # b_result_img = run_through_both(src_folder=src_folder)
    # b_result_img = run_through_cv2_no_order(src_folder=src_folder)
    # b_result_img = run_simple(src_folder)

    save_path = r"/home/datademon/Desktop/Alik/galax_spip_v2/data"
    save_path = os.path.join(save_path, src_folder)
    save_path = os.path.join(save_path, 'saves')
    
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    b_less_path = os.path.join(save_path, f'{random_string}_b_less.tiff')
    b_path = os.path.join(save_path, f'{random_string}_b.tiff')

    print('Save-Paths:')
    print(b_less_path)
    print(b_path)
    
    if not main:
        tifffile.imwrite(b_less_path, b_less_result_img)
        tifffile.imwrite(b_path, b_result_img)

    print(f'--------------- Successfully Saved TIFF Images | Plot={main} ---------------')

    if main:
        
        fig, axs = plt.subplots(1, 2, figsize = (7, 7))
        
        axs[0].imshow(b_less_result_img)
        axs[0].set_title('B-Less-Stitching')

        axs[1].imshow(b_result_img)
        axs[1].set_title('B-Stitching')

        plt.show()



if __name__ == '__main__':

    main = True
    ### OpenSfM Extraction ->
    # Load JSON data
    parser = argparse.ArgumentParser(description='Copy every nth image from a source folder to a destination folder.')
    parser.add_argument('--src', type=str, required=True, help='Path to the source folder containing images')
    args = parser.parse_args()
    src_folder = Path(args.src)
    print(src_folder)

    main_plot_save(src_folder, main)

