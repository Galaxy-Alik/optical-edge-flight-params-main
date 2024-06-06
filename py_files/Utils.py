import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import random
import numpy as np
from math import atan2, asin
import pandas as pd
from IPython.display import display
from pprint import pprint
from PIL import Image
import time
import datetime
import time
import sys
from pymap3d import geodetic2enu, enu2geodetic
import warnings
warnings.filterwarnings('ignore')
from vo import FeatureLocalization

def jetson_timestamp_to_utc(jetson_timestamp):

    seconds, nanoseconds = jetson_timestamp.split('_')
    seconds = int(seconds)
    nanoseconds = int(nanoseconds)
    microseconds = nanoseconds / 1000

    utc_time = datetime.datetime.fromtimestamp(seconds)
    utc_time += datetime.timedelta(microseconds=microseconds)
    utc_time_str = utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')

    return utc_time_str

def closest_ea(target_timestamp, rev_flight_obj, df_data, want_timestamp = False, want_prev = False):

    # Calculate the absolute difference between the target timestamp and each timestamp in the list
    utc_target_timestamp = rev_flight_obj[target_timestamp]
    dt = datetime.datetime.strptime(utc_target_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    target_unix_timestamp = dt.timestamp()

    df_ins_data_time, df_ins_data_all = df_data[0], df_data[-1] 
    j_col = df_ins_data_time.columns[-1]
    p_col = df_ins_data_time.columns[0]
    jetson_timestamps = list(df_ins_data_time[' Jetson Time'])
    diffs = [(abs(target_unix_timestamp - ts), ts) for ts in jetson_timestamps]
    diffs.sort(key=lambda x: x[0])
    closest_timestamp = diffs[0][1]
    packet_count = list(df_ins_data_time[p_col].loc[df_ins_data_time[j_col]  == closest_timestamp])[0]

    if want_prev:
      sampled_row = df_ins_data_all.loc[df_ins_data_all[p_col] == (packet_count - 1)]

    else:
      sampled_row = df_ins_data_all.loc[df_ins_data_all[p_col] == packet_count]

    closest_euler_angle = [list(sampled_row[' Roll_E'])[0], list(sampled_row[' Pitch_N'])[0], list(sampled_row[' Yaw_U'])[0]]
    
    if want_timestamp:
      return closest_euler_angle, closest_timestamp 
    
    else:
      return closest_euler_angle


def closest_t(target_timestamp, rev_flight_obj, df_data, want_timestamp = False, want_prev = False):

    utc_target_timestamp = rev_flight_obj[target_timestamp]
    dt = datetime.datetime.strptime(utc_target_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    target_unix_timestamp = dt.timestamp()

    df_ins_data_time, df_ins_data_all = df_data[0], df_data[-1]
    j_col = df_ins_data_time.columns[-1]
    p_col = df_ins_data_time.columns[0]
    jetson_timestamps = list(df_ins_data_time[' Jetson Time'])
    diffs = [(abs(target_unix_timestamp - ts), ts) for ts in jetson_timestamps]
    diffs.sort(key=lambda x: x[0])
    closest_timestamp = diffs[0][1]
    packet_count = list(df_ins_data_time[p_col].loc[df_ins_data_time[j_col]  == closest_timestamp])[0]

    if want_prev:
      sampled_row = df_ins_data_all.loc[df_ins_data_all[p_col] == (packet_count - 1)]

    else:
      sampled_row = df_ins_data_all.loc[df_ins_data_all[p_col] == packet_count]

    closest_dir = [list(sampled_row[' Latitude'])[0], list(sampled_row[' Longitude'])[0], list(sampled_row[' AltEllipsoid'])[0]]

    if want_timestamp:
      return closest_dir, closest_timestamp 
    
    else:
      return closest_dir

def get_unix_timestamps(jetson_timestamp):

    seconds, nanoseconds = jetson_timestamp.split('_')
    seconds = int(seconds)
    nanoseconds = int(nanoseconds)
    microseconds = nanoseconds / 1000

    utc_time = datetime.datetime.fromtimestamp(seconds)
    utc_time += datetime.timedelta(microseconds=microseconds)
    utc_time_str = utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')

    dt = datetime.datetime.strptime(utc_time_str, "%Y-%m-%d %H:%M:%S.%f")
    unix_timestamp = dt.timestamp()

    return unix_timestamp


def trans_vec_to_lla(trans_vec, R, scale_factor, initial_ea, initial_coordinates):

    lat0, lon0, alt0 = initial_coordinates
    yaw0, pitch0, roll0 = initial_ea

    e0, n0, u0 = geodetic2enu(lat0, lon0, alt0, lat0, lon0, 0)

    R_yaw = np.array([[np.cos(yaw0), -np.sin(yaw0), 0],
                      [np.sin(yaw0), np.cos(yaw0), 0],
                      [0, 0, 1]])

    R_pitch = np.array([[np.cos(pitch0), 0, np.sin(pitch0)],
                        [0, 1, 0],
                        [-np.sin(pitch0), 0, np.cos(pitch0)]])

    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll0), -np.sin(roll0)],
                       [0, np.sin(roll0), np.cos(roll0)]])

    R0 = R_yaw @ R_pitch @ R_roll

    trans_vec_scaled = trans_vec * scale_factor
    trans_vec_rotated = R @ R0.T @ trans_vec_scaled

    e1 = e0 + trans_vec_rotated[0]
    n1 = n0 + trans_vec_rotated[1]
    u1 = u0 + trans_vec_rotated[2]

    lat, lon, alt = enu2geodetic(e1, n1, u1, lat0, lon0, 0)

    return lat[0], lon[0], alt[0]

class EAE():

  def __init__(self, second_frame, first_frame):
    self.img1 = second_frame
    self.img2 = first_frame

  def calc_euler_angle(self, rot_matrix):

    r11, r12, r13 = rot_matrix[0]
    r21, r22, r23 = rot_matrix[1]
    r31, r32, r33 = rot_matrix[2]

    # Calculate pitch angle
    pitch = -asin(r31)

    # Calculate roll angle
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) > 1e-8:  # Prevent division by zero
        roll = atan2(r32, r33)
    else:
        roll = atan2(r21, r11)

    # Calculate yaw angle
    sec_pitch = 1.0 / cos_pitch
    yaw = atan2(r21 * sec_pitch, r11 * sec_pitch)

    return roll, pitch, yaw

  def run(self, is_store = False):
    feat = FeatureLocalization(self.img1, self.img2)
    baseImage_kp, baseImage_des, secImage_kp, secImage_des = feat.feature_detection()
    matches = feat.feature_match(baseImage_des, secImage_des)
    rot_matrix, t_vec = feat.estimatePose(matches, baseImage_kp, secImage_kp)

    # print(rot_matrix)

    roll, pitch, yaw = self.calc_euler_angle(rot_matrix)

    roll_deg = np.rad2deg(roll)
    pitch_deg = np.rad2deg(pitch)
    yaw_deg = np.rad2deg(yaw)

    euler_angle_rad = {
        'roll' : roll,
        'pitch' : pitch,
        'yaw' : yaw
    }

    euler_angle_deg = {
        'roll' : roll_deg,
        'pitch' : pitch_deg,
        'yaw' : yaw_deg
    }
    if is_store:
      return rot_matrix, t_vec, euler_angle_rad, euler_angle_deg

    else:
      return euler_angle_rad, euler_angle_deg


def findHomography(matches, kp1, kp2):
    
    if len(matches) < 4:
        print("Not enough matches found between the images to compute homography")
        exit(0)

    baseImage_pts = []
    secImage_pts = []
    for match in matches:
        baseImage_pts.append(kp1[match[0].queryIdx].pt)
        secImage_pts.append(kp2[match[0].trainIdx].pt)

    baseImage_pts = np.float32(baseImage_pts)
    secImage_pts = np.float32(secImage_pts)

    (homographyMatrix, status) = cv2.findHomography(secImage_pts, baseImage_pts, cv2.RANSAC, 4.0)
    return homographyMatrix, status

def getNewFrameSizeAndMatrix(homographymatrix, shape_img2, shape_img1):

    (h, w) = shape_img2

    initialmatrix = np.array([[0, w - 1, w - 1, 0],
                          [0, 0, h - 1, h - 1],
                          [1, 1, 1, 1]])

    finalmatrix = np.dot(homographymatrix, initialmatrix)
    [x, y, c] = finalmatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    modW, modH = max_x, max_y
    correction = [0,0]
    if min_x < 0:
        modW -= min_x
        correction[0] = abs(min_x)

    if min_y < 0:
        modH -= min_y
        correction[1] = abs(min_y)

    if modW < shape_img1[1] + correction[0]:
        modW = shape_img1[1] + correction[0]
    if modH < shape_img1[0] + correction[1]:
        modH = shape_img1[0] + correction[1]

    x = np.add(x, correction[0])
    y = np.add(y, correction[1])
    
    initialPts= np.float32([[0, 0],
                                [w - 1, 0],
                                [w - 1, h - 1],
                                [0, h - 1]])
    finalPts = np.float32(np.array([x, y]).transpose())
    homographyMatrix = cv2.getPerspectiveTransform(initialPts, finalPts)  
    return [modH, modW], correction, homographyMatrix

def compute_homography(img1, img2):

    feat = FeatureLocalization(img1, img2)
    baseImage_kp, baseImage_des, secImage_kp, secImage_des = feat.feature_detection()
    matches = feat.feature_match(baseImage_des, secImage_des)
    homography_matrix, _ = findHomography(matches, baseImage_kp, secImage_kp)

    return homography_matrix

def stitch_frames(img1, img2, curr_R, curr_t):

    homographyMatrix = compute_homography(img1, img2)
    newFrameSize, correction, homographyMatrix = getNewFrameSizeAndMatrix(homographyMatrix, img2.shape[:2], img1.shape[:2])
    stitchedFrame = cv2.warpPerspective(img2, homographyMatrix, (newFrameSize[1], newFrameSize[0]))
    stitchedFrame[correction[1]:correction[1] + img1.shape[0], correction[0]:correction[0] + img1.shape[1]] = img1

    return stitchedFrame

def euler_angles_to_rotation_matrix(ea):

    roll, pitch, yaw = ea

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    rotation_matrix = Rz @ Ry @ Rx

    return rotation_matrix
