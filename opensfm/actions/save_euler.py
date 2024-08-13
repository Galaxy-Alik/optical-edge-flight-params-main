from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import pandas as pd
from math import atan2, asin
import argparse
import datetime
import numpy as np

from .Extras.Utils import *
from .stitch_helper import *
# from stitch_helper import *


sensor_width = 29.3  # mm
sensor_height = 29  # mm

image_width = 1920  # in pixels
image_height = 1200  # in pixels

pixel_size_x = sensor_width / image_width
pixel_size_y = sensor_height / image_height

scale_factor_x = pixel_size_x / fx  # m/pixel
scale_factor_y = pixel_size_y / fy  # m/pixel

scale_factor = (scale_factor_x + scale_factor_y) / 2


''' All the Rotations and Translations obtained are relative, so a reference has to be taken from INS data Source '''

def key_fn(item):
    return item.split('_')[2]

def get_rt(src_folder):

    parent_file = r"/home/datademon/Desktop/Alik/galax_spip_v2/data"
    parent_file = os.path.join(parent_file, src_folder)
    
    json_file = os.path.join(parent_file, r"reconstruction.json")
    print('JSON-File: ', json_file)
    
    _, shot_params = extract_parameters(json_file)
    
    print("Camera Parameters:")
    for shot_id, params in shot_params.items():
        print(f"Shot ID: {shot_id}")
        print(f"  Rotation (R): {params['R']}")
        print(f"  Translation (T): {params['T']}")

    frame_ids = [frame_id for frame_id, _ in shot_params.items()]

    frame_ids = sorted(frame_ids, key = key_fn, reverse = False)

    return frame_ids, shot_params

def calc_euler_angle(rot_matrix):

    r11, r12, r13 = rot_matrix[0]
    r21, r22, r23 = rot_matrix[1]
    r31, r32, r33 = rot_matrix[2]

    pitch = -asin(r31)

    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) > 1e-8:  # Prevent division by zero
        roll = atan2(r32, r33)
    else:
        roll = atan2(r21, r11)

    sec_pitch = 1.0 / cos_pitch
    yaw = atan2(r21 * sec_pitch, r11 * sec_pitch)

    return roll, pitch, yaw

def convert_to_utc(jetson_timestamp):

    seconds, nanoseconds = jetson_timestamp.split('_')[3:]
    nanoseconds = nanoseconds[:-4]
    
    print(seconds, nanoseconds)
    
    seconds = int(seconds)
    nanoseconds = int(nanoseconds)
    microseconds = nanoseconds / 1000

    utc_time = datetime.datetime.fromtimestamp(seconds)
    utc_time += datetime.timedelta(microseconds=microseconds)
    utc_time_str = utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')

    return utc_time_str


def get_closest_timestamp(target_timestamp, df_data):

    utc_target_timestamp = convert_to_utc(target_timestamp)
    dt = datetime.datetime.strptime(utc_target_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    target_unix_timestamp = dt.timestamp()
    df_ins_data_time, df_ins_data_all = df_data[0], df_data[-1] 
    j_col = df_ins_data_time.columns[-1]
    p_col = df_ins_data_time.columns[0]
    jetson_timestamps = list(df_ins_data_time[' Jetson Time'])
    diffs = [(abs(target_unix_timestamp - ts), ts) for ts in jetson_timestamps]
    diffs.sort(key=lambda x: x[0])
    closest_timestamp = diffs[0][1]

    return closest_timestamp

def get_timestamp(target_timestamp):

    utc_target_timestamp = convert_to_utc(target_timestamp)
    dt = datetime.datetime.strptime(utc_target_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    target_unix_timestamp = dt.timestamp()
    
    return target_unix_timestamp

def get_ins_ea_t(target_timestamp, df_data):

    df_ins_data_time, df_ins_data_all = df_data[0], df_data[-1] 
    j_col = df_ins_data_time.columns[-1]
    p_col = df_ins_data_time.columns[0]
    ins_timestamp = get_closest_timestamp(target_timestamp, df_data)
    print('Closest-INS-Timestamp: ', ins_timestamp)
    packet_count = list(df_ins_data_time[p_col].loc[df_ins_data_time[j_col]  == ins_timestamp])[0]
    sampled_row = df_ins_data_all.loc[df_ins_data_all[p_col] == packet_count]
    closest_euler_angle = [list(sampled_row[' Roll_E'])[0], list(sampled_row[' Pitch_N'])[0], list(sampled_row[' Yaw_U'])[0]]
    closest_dir = [list(sampled_row[' Latitude'])[0], list(sampled_row[' Longitude'])[0], list(sampled_row[' AltEllipsoid'])[0]]

    return closest_euler_angle, closest_dir


def run_main(src_folder, df_data):

    frame_ids, shot_params = get_rt(src_folder)
    
    timestamp_list, roll_list, pitch_list, yaw_list, lat_list, lon_list, alt_list = [], [], [], [], [], [], []
    del_roll_list, del_pitch_list, del_yaw_list = [], [], []

    ## Are the Selected Frames Chronologically Orgainsed? NO
    initial_timestamp = frame_ids[0]
    init_ea, init_t = get_ins_ea_t(initial_timestamp, df_data)
    init_roll, init_pitch, init_yaw = init_ea
    init_lat, init_lon, init_alt = init_t
    
    print(' ------------ Initial Values ------------ ')

    print(f' Initial Frame: {frame_ids[0]}')
    print(f' Initial Roll: {init_roll}, Initial Pitch: {init_pitch}, Initial Yaw: {init_yaw}')
    print(f' Initial Lat: {init_lat}, Initial Lon: {init_lon}, Initial Alt: {init_alt}')
    
    print(' ------------------------------------ ')

    for i in tqdm(range(len(frame_ids) - 1), desc="Saving eulers (OPT)", unit="frame"):

        print('---------------------')
        print('Frames_ids: ', frame_ids[i])

        timestamp = get_timestamp(frame_ids[i])

        timestamp_list.append(timestamp)
        roll_list.append(init_roll)
        pitch_list.append(init_pitch)
        yaw_list.append(init_yaw)
        lat_list.append(init_lat)
        lon_list.append(init_lon)
        alt_list.append(init_alt)

        R, t = np.array(shot_params[frame_ids[i]]['R']).reshape(3, 1), np.array(shot_params[frame_ids[i]]['T']).reshape(3, 1)
        R, _ = cv2.Rodrigues(R)

        del_roll, del_pitch, del_yaw = calc_euler_angle(R)

        del_roll_list.append(del_roll)
        del_pitch_list.append(del_pitch)
        del_yaw_list.append(del_yaw)

        init_roll += del_roll
        init_pitch += del_pitch
        init_yaw += del_yaw
        
        print(f'roll: {init_roll}, pitch: {init_pitch}, yaw: {init_yaw}')
        print('---------------------')

        init_lat, init_lon, init_alt = trans_vec_to_lla(t, R, scale_factor, [init_roll, init_pitch, init_yaw], [init_lat, init_lon, init_alt])

    df_OF_cols = ['Timestamps', 'Roll_N', 'Pitch_E', 'Yaw_U', 'Lat', 'Lon', 'Alt']
    df_ins = pd.DataFrame(columns = df_OF_cols)

    data = {
        'Timestamps': timestamp_list,
        'Roll_N': roll_list,
        'Pitch_E': pitch_list,
        'Yaw_U': yaw_list,
        'Lat' : lat_list,
        'Lon' : lon_list,
        'Alt' : alt_list
        }

    del_data = {
        'Timestamps': timestamp_list,
        'del_Roll_N': del_roll_list,
        'del_Pitch_E': del_pitch_list,
        'del_Yaw_U': del_yaw_list
    }

    df_ins = pd.DataFrame(data)
    df_del_ins = pd.DataFrame(del_data)

    return df_ins, df_del_ins

def run_and_save(src_folder):

    print('Scale-Factor: ', scale_factor)

    parent_path = '/home/datademon/Desktop/Alik/galax_spip_v2/data'

    ''' INS - matching '''
    ins_path = os.path.join(parent_path, src_folder)
    ins_path = os.path.join(ins_path, 'ins')
    
    comb_ins_path, time_ins_path = [nm for nm in os.listdir(ins_path) if nm[:3] == 'com'][0], [nm for nm in os.listdir(ins_path) if not nm[:3] == 'com'][0]
    comb_ins_path, time_ins_path = os.path.join(ins_path, comb_ins_path), os.path.join(ins_path, time_ins_path)
    print(comb_ins_path, time_ins_path)
    df_data = [pd.read_csv(time_ins_path), pd.read_csv(comb_ins_path)]

    df_ins, df_del_ins = run_main(src_folder, df_data)
    
    parent_path = os.path.join(parent_path, src_folder)
    parent_path = os.path.join(parent_path, 'saves')
    save_path = os.path.join(parent_path, 'EA')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    folder_number = str(src_folder).split("-")[1]
    n_frames = str(src_folder).split('-')[-1]
    # save_path = os.path.join(save_path, f'optical_frame_parameters_strip_{folder_number}.csv')
    # save_path = os.path.join(save_path, f'optical_frame_parameters_strip_{str(src_folder).split('-')[1]}.csv')

    df_ins.to_csv(os.path.join(save_path, f'optical_frame_parameters_strip_{folder_number}_n{n_frames}.csv'), index=False)
    df_del_ins.to_csv(os.path.join(save_path, f'DEL_optical_frame_parameters_strip_{folder_number}_n{n_frames}.csv'), index=False)

    fig, axs = plt.subplots(1, 3, figsize = (30, 12))

    axs[0].plot(df_ins['Timestamps'].tolist(), df_ins['Roll_N'].tolist())
    axs[0].set_xlabel('time (UNIX)')
    axs[0].set_ylabel('degree')
    axs[0].set_title('Roll_N')
    axs[1].plot(df_ins['Timestamps'].tolist(), df_ins['Pitch_E'].tolist())
    axs[1].set_title('Pitch_E')
    axs[2].plot(df_ins['Timestamps'].tolist(), df_ins['Yaw_U'].tolist())
    axs[2].set_title('Yaw_U')

    plt.savefig(os.path.join(save_path, f'optical_frame_parameters_strip_{folder_number}_n{n_frames}.png'))

    ''' DEL-Eulers '''

    fig, axs = plt.subplots(1, 3, figsize = (30, 12))

    axs[0].plot(df_del_ins['Timestamps'].tolist(), df_del_ins['del_Roll_N'].tolist())
    axs[0].set_xlabel('time (UNIX)')
    axs[0].set_ylabel('degree')
    axs[0].set_title('del_Roll_N')
    axs[1].plot(df_del_ins['Timestamps'].tolist(), df_del_ins['del_Pitch_E'].tolist())
    axs[1].set_title('del_Pitch_E')
    axs[2].plot(df_del_ins['Timestamps'].tolist(), df_del_ins['del_Yaw_U'].tolist())
    axs[2].set_title('del_Yaw_U')

    plt.savefig(os.path.join(save_path, f'DEL_optical_frame_parameters_strip_{folder_number}_n{n_frames}.png'))

    print(' --------- Sucessfully Saved Euler Angles --------- ')

if __name__ == '__main__':

    main = True
    parser = argparse.ArgumentParser(description='Copy every nth image from a source folder to a destination folder.')
    parser.add_argument('--src', type=str, required=True, help='Path to the source folder containing images')
    args = parser.parse_args()
    src_folder = Path(args.src)
    print(' ----- Src-Folder ----- ')
    print(src_folder)

    run_and_save(src_folder)