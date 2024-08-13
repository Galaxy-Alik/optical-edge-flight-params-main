import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import io
import sys
from pathlib import Path
import zipfile
import gzip
import argparse
from pprint import pprint

def source_kp(frame_id, src_folder):

    # print(f' --- Sourcing - Key - Points - for - {frame_id} ---')

    source_kp_path = '/home/datademon/Desktop/Alik/galax_spip_v2/data'
    source_kp_path = os.path.join(source_kp_path, src_folder)
    source_kp_path = os.path.join(source_kp_path, 'features')
    kp_path = [p for p in os.listdir(source_kp_path) if p.split('_')[2] == frame_id.split('_')[2]][0]
    kp_path = os.path.join(source_kp_path, kp_path)

    # print(f'kp_path - {kp_path}')

    with zipfile.ZipFile(kp_path, 'r') as zip_file:
    # Read points.npy
        with zip_file.open('points.npy') as file:
            points = np.load(io.BytesIO(file.read()))
        
        # Read descriptors.npy
        with zip_file.open('descriptors.npy') as file:
            descriptors = np.load(io.BytesIO(file.read()))

    return points, descriptors

def source_matches(frame_id, src_folder):

    # print(f' --- Sourcing - Matches - for - {frame_id} ---')

    source_m_path = '/home/datademon/Desktop/Alik/galax_spip_v2/data'
    source_m_path = os.path.join(source_m_path, src_folder)
    source_m_path = os.path.join(source_m_path, 'matches')
    m_path = [p for p in os.listdir(source_m_path) if p.split('_')[2] == frame_id.split('_')[2]][0]
    m_path = os.path.join(source_m_path, m_path)

    # print(f'm_path - {m_path}')

    with gzip.open(m_path, 'rb') as f:
        matches = pickle.load(f)

    return matches
    
def source_matches_v2(frame_id_0, frame_id_1, src_folder):

    source_m_path = '/home/datademon/Desktop/Alik/galax_spip_v2/data'
    source_m_path = os.path.join(source_m_path, src_folder)
    source_m_path = os.path.join(source_m_path, 'all_matches')
    source_m_path = os.path.join(source_m_path, os.listdir(source_m_path)[0])

    match_dict = np.load(source_m_path, allow_pickle = True).tolist()
    target_match = (frame_id_0, frame_id_1)
    
    is_match = target_match in list(match_dict.keys())
    
    if not is_match:
        return None
    else:
        target_match_idx = match_dict[target_match]
        return target_match_idx
    
    
# if __name__ == '__main__':

#     frame_id = r"img_E333_299087788230_1691740725_20397897.jpg"
#     parser = argparse.ArgumentParser(description='Copy every nth image from a source folder to a destination folder.')
#     parser.add_argument('--src', type=str, required=True, help='Path to the source folder containing images')
#     args = parser.parse_args()
#     src_folder = Path(args.src)
#     print(' ----- Src-Folder ----- ')
#     points, descriptors = source_kp(frame_id, src_folder)
#     print()
#     matches = source_matches(frame_id, src_folder)
