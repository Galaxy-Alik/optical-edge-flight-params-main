import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import tqdm
import time
import argparse

from .Extras.pose_est import *
from .Extras.vo import * 
from .Extras.Utils import *

# from Extras.pose_est import *
# from Extras.vo import * 
# from Extras.Utils import *


class bl_stitch():
  
  def __init__(self, src_folder):

    print(' ----------- Bundle - Less - Implementation ----------- ')

    parent_file = r"/home/datademon/Desktop/Alik/galax_spip_v2/data"

    ''' Load the Image-Files '''
    parent_file = os.path.join(parent_file, src_folder)
    img_data_path = os.path.join(parent_file, r"images")
    img_path_list = [os.path.join(img_data_path, fn) for fn in os.listdir(img_data_path) if (fn.endswith('.jpg') or fn.endswith('.tiff'))]
    self.resized_frames = [np.asarray(Image.open(img)) for img in img_path_list]

    self.num_frames = len(self.resized_frames)
    print('Num-Frames: ', self.num_frames)
  
  def run(self):
    
    # euler_angle_list = []
    # translation_list = []

    ft = time.time()

    for i in tqdm((range(self.num_frames)), desc="Stitching frames", unit="frame"):

        if i == 0:
            result_img = self.resized_frames[i]
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0]).astype('float64')

        else:
            Image1 = result_img
            Image2 = self.resized_frames[i]

            # Checking if images read
            if Image1 is None or Image2 is None:
                print("\nImages not read properly or does not exist.\n")
                exit(0)

            eae = EAE(Image2, Image1)
            _, t, euler_angle_rad, _ = eae.run(is_store = True)
            euler_angle = [euler_angle_rad['roll'], euler_angle_rad['pitch'], euler_angle_rad['yaw']]
            R = euler_angles_to_rotation_matrix(euler_angle)
            # euler_angle_list.append(euler_angle)

            # Calling function for stitching images.prev_t
            # stitcher = Stitcher(Image2, Image1, detector, matcher, threshold)
            # _, R_temp, _ = stitcher.stitch_frames(curr_R, curr_t, fx, cx, cy)

            if i == 1:
              prev_R = curr_R
              prev_t = curr_t

              curr_R = R
              curr_t = t

            else:
              curr_R = np.matmul(prev_R, R)
              curr_t = (np.matmul(prev_R, t)) + (prev_t)

            result_img = manual_stitch_frames(Image2, Image1, prev_R, np.squeeze(prev_t), curr_R, np.squeeze(curr_t))

            prev_R = curr_R
            prev_t = abs(curr_t)

        pass

    lt = time.time()

    print('Execution-time (UNIX): ', (lt - ft))
    return result_img
  
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Copy every nth image from a source folder to a destination folder.')
  parser.add_argument('--src', type=str, required=True, help='Path to the source folder containing images')
  args = parser.parse_args()
  src_folder = Path(args.src)

  result_img = bl_stitch(src_folder=src_folder).run()

  fig, axs = plt.subplots(1, 1, figsize = (12, 12))

  # result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
  # result_img = cv2.resize(result_img, image_size)

  axs.imshow(result_img)
  axs.set_title('Optical-Frame-Stitched-Image')

  plt.axis('off')
  plt.show()