#!/usr/bin/env python
# coding: utf-8
# need conda env pytorch3d on astro2
# Run as jupyter notebook
#%%
import os
import shutil

import matplotlib.pyplot as plt
%matplotlib inline # type: ignore

#%%
import numpy as np
import pandas as pd
import torch

from mesh3d_utils import compile_all_steps

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


#%%

models_root_dir = '/data/ShapeNetSem/models/'
shapenet_metadata_file_path = '/data/ShapeNetSem/metadata.csv'

def read_shapnet_date(file_path):
    """
    Reads the metadata file and returns a pandas dataframe.

    Args:
        file_path: str, path to the metadata file
    """
    df = pd.read_csv(file_path)
    df.fillna('NAN', inplace=True)
    return df


# %%
MAX_NUM_FILES_TO_RENDER = 25
image_size = 1024
camera_dist = 4
elevation = np.array([0, 90])
azim_angle = np.array([0, 90])

df = read_shapnet_date(shapenet_metadata_file_path)

# target_category = 'sink' #[xxx, 11523, 2720, 7022]
# target_category = 'plant' #[223, 541, 1070]
# target_category = 'rose' #[5898, 6388]
# target_category = 'dollar'
# target_category = 'chair'
# target_category = 'pot'
# target_category = 'vase'
# target_category = 'lamp'
# target_category = 'cap' 
# target_category = 'shoe'
target_category = 'house'

# %%

df_subset = df[df.name.str.contains(target_category)]

print(f'number of target_category={target_category}', 
        df_subset.shape)

df_subset = df_subset[['fullId','wnlemmas','name','tags']]

if len(df_subset) > MAX_NUM_FILES_TO_RENDER:
    df_subset = df_subset.iloc[:MAX_NUM_FILES_TO_RENDER]

# df_subset = df_subset[MAX_NUM_FILES_TO_RENDER:MAX_NUM_FILES_TO_RENDER+20]

for row_idx, row in df_subset.iterrows():
    try:
        obj_file_name = row.fullId
        wnlemmas = row.wnlemmas
        name = row['name']
        obj_file_path = os.path.join(models_root_dir, obj_file_name[len('wss.'):]) + '.obj'
        print(f'obj file path: {obj_file_path}')
        print(f'wnlemmas: {wnlemmas}, name: {name}')

        fig, axs = plt.subplots(len(azim_angle), len(elevation))
        for xi, x in  enumerate(elevation):
            for yi, y in enumerate(azim_angle):
                compile_all_steps(image_size, camera_dist, device,
                                    x, y, obj_file_path, axs[yi][xi])

        fig.suptitle(f'{row_idx}_{target_category} azim_elev \n {obj_file_path}', fontsize=16)
        fig.set_size_inches(8, 8)
        fig.tight_layout()
        plt.show()
    except FileNotFoundError:
        print(f'{obj_file_path} not found')
        continue



# %%

collected_data_root = f'/home/ali/text2mesh/astro_utils/collected_shapes/{target_category}'
# make directory to save the data
os.makedirs(collected_data_root, exist_ok=True)

# save the .obj files that look good
# looks_good_list = [12, 234, 1348, 2968, 4126, 5534, 5898, 6388] # rose
# looks_good_list = [502, 865, 1381, 1707, 2720, 3999, 5071, 5129, 5472, 7022, 8042] # sink
# looks_good_list = [591, 548, 525, 510, 293, 110, 61, 9] # chair
# looks_good_list = [3907, 2450, 2365, 1648, 663, 201, 200] # tea pot
# looks_good_list = [2170, 1877, 1845, 1720, 1671, 1564, 1103, 937, 895, 872, 847, 730, 423, 397] # vase
# looks_good_list = [489, 462, 459, 456, 443, 426, 421, 381, 324, 306, 208, 148, 139, 134] # lamp
# looks_good_list = [8989, 6606, 6279, 4308, 4205, 263] # cap
# looks_good_list = [9654, 9463, 9160, 8926, 6620, 6058, 5608, 2083, 2008, 1550, 706] # shoe
looks_good_list = [6878, 1273] # house

for i in looks_good_list:
    obj_file_name = df.loc[i].fullId
    obj_file_path = os.path.join(models_root_dir, obj_file_name[len('wss.'):]) + '.obj'
    
    # copy the .obj file to the collected data directory
    shutil.copy(obj_file_path, collected_data_root)

# %%

