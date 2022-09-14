# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import norm
from pytorch3d.structures import Meshes

from mesh3d_utils import (compile_all_steps, convert_to_mesh, get_mesh,
                          get_renderer, remove_duplicate_vertices,
                          render_image, save_mesh, my_export_mesh,
                          upsample_mesh_based_on_edge_length,
                          upsample_mesh_based_on_triangular_face_area,
                          upsample, upsample_v2)

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

#%%

# obj_file_path = '/home/ali/text2mesh/astro_utils/collected_shapes/sink/421eca272e20457f174cc62fa41808fc.obj'
# # obj_file_path = '/home/ali/text2mesh/data/source_meshes/person.obj'
# image_size = 1024
# camera_dist = 4
# elevation = np.array([0, 90])
# azim_angle = np.array([0, 90])

# fig, axs = plt.subplots(len(azim_angle), len(elevation))
# for xi, x in  enumerate(elevation):
#     for yi, y in enumerate(azim_angle):
#         compile_all_steps(image_size, camera_dist, device,
#                             x, y, obj_file_path, axs[yi][xi])

# fig.suptitle(f'azim_elev \n {obj_file_path}', fontsize=16)
# fig.set_size_inches(8, 8)
# fig.tight_layout()
# plt.show()

# %%

# The mesh might have duplicate vertices. We need to remove them.
# mesh = get_mesh(obj_file_path, device, normalize=True)

# n_area, area_threshold = 3, 0.05
# n_edge, len_threshold = 5, 0.05


# mesh = upsample(mesh, n_area, area_threshold, n_edge, len_threshold)
# save_mesh(mesh, f'AAupsampled_mesh{n_area}_{area_threshold}_{n_edge}_{len_threshold}.obj')


#%%
import os

root_dir = '/home/ali/text2mesh/astro_utils/collected_shapes'


n_area, area_threshold = 3, 0.05
n_edge, len_threshold = 5, 0.05


# get all the .obj files in the root_dir or any sub-directory
obj_files = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.obj'):
            if 'upsampled' not in root:
                if 'sink' in root:
                    obj_files.append(
                        [root, file]
                        )

# %%
for dir_path, file_name in obj_files:
    obj_file_path = os.path.join(dir_path, file_name)
    mesh = get_mesh(obj_file_path, device, normalize=True)
    mesh = upsample(mesh, n_area, area_threshold, n_edge, len_threshold)

    dir_name = os.path.basename(dir_path)
    save_dir = f'{root_dir}/upsampled/{dir_name}/{n_area}_{area_threshold}_{n_edge}_{len_threshold}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    # save_mesh(mesh, save_path)
    my_export_mesh(mesh, save_path)
    print(f'saved to {save_path}.')
    # break
print('Done!!!')

# %%

n_upsample = 4
area_threshold = 0.05
len_threshold = 0.03
for dir_path, file_name in obj_files:
    obj_file_path = os.path.join(dir_path, file_name)
    mesh = get_mesh(obj_file_path, device, normalize=True)
    mesh = upsample_v2(mesh, n_upsample, area_threshold, len_threshold)

    dir_name = os.path.basename(dir_path)
    save_dir = f'{root_dir}/upsampled/{dir_name}/{n_upsample}_{area_threshold}_{len_threshold}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    # save_mesh(mesh, save_path)
    my_export_mesh(mesh, save_path)
    print(f'saved to {save_path}. \n\n')

print('Done!!!')

# %%
