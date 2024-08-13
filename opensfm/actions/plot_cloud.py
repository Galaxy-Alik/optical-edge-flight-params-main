import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
from pathlib import Path
import os
import sys
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import io
from PIL import Image
# import open3d as o3d

def pc_to_ply(points, colors, output_path):

    """
    Write a point cloud to a .ply file.
    
    :param filename: Output .ply filename
    :param points: Nx3 numpy array of point coordinates
    :param colors: Nx3 numpy array of RGB colors (values 0-255)
    """
    assert points.shape == colors.shape
    assert points.shape[1] == 3
    
    num_points = points.shape[0]
    
    filename = os.path.join(output_path, 'sparse_pc.ply')

    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write point data
        for point, color in zip(points, colors):
            x, y, z = point
            r, g, b = color
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def plot_3d_point_cloud(json_path, is_main = False):

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    points = data[0]['points']

    x = []
    y = []
    z = []
    colors = []

    for point in points.values():
        coord = point['coordinates']
        x.append(coord[0])
        y.append(coord[1])
        z.append(coord[2])
        colors.append(np.array(point['color']) / 255.0)  # Normalize color values to [0, 1]

    point_new = np.array([x, y, z]).reshape(-1, 3)
    colors = np.array(colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=colors, s=20, marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    src_folder = json_path.split('/')[-2]
    output_path = os.path.join('/'.join(json_path.split('/')[:-1]), 'saves')
    output_path = os.path.join(output_path, 'PC')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if is_main:
        plt.show()

    def update_v(frame):
        ax.view_init(elev=10., azim=frame)
        return scatter,

    def update_h(frame):
        ax.view_init(elev=frame, azim=0)  # Changed this line
        return scatter,

    anim_v = FuncAnimation(fig, update_v, frames=np.linspace(0, 360, 180), interval=50, blit=True)
    anim_h = FuncAnimation(fig, update_h, frames=np.linspace(0, 360, 180), interval=50, blit=True)
    
    frames = []
    for i in range(180):
        update_v(i*2)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))
    
    fn_name_v = f'point_cloud_V_{src_folder}.gif'
    output_path_v = os.path.join(output_path, fn_name_v) 

    frames[0].save(output_path_v, save_all=True, append_images=frames[1:], duration=50, loop=0)
    
    frames = []
    for i in range(180):
        update_h(i*2)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))
    
    fn_name_h = f'point_cloud_H_{src_folder}.gif'
    output_path_h = os.path.join(output_path, fn_name_h) 

    frames[0].save(output_path_h, save_all=True, append_images=frames[1:], duration=50, loop=0)

    plt.close(fig)

    pc_to_ply(point_new, colors, output_path)

    
if __name__ == '__main__':

    is_main = True

    parser = argparse.ArgumentParser(description='Copy every nth image from a source folder to a destination folder.')
    parser.add_argument('--src', type=str, required=True, help='Path to the source folder containing images')
    args = parser.parse_args()
    src_folder = Path(args.src)
    print(src_folder)

    json_path = os.path.join('/home/datademon/Desktop/Alik/galax_spip_v2/data', src_folder)
    json_path = os.path.join(json_path, 'reconstruction_sampled.json')
    print('JSON_PATH: ', json_path)

    plot_3d_point_cloud(json_path, is_main)
