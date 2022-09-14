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


import trimesh                         

#%%

./build/quadriflow -sharp -sat -mcf \
-i /home/ali/text2mesh/astro_utils/collected_shapes/sink/421eca272e20457f174cc62fa41808fc.obj \
-o temp_output.obj \
-f 100000

./build/quadriflow -sharp -sat \
-i /home/ali/text2mesh/astro_utils/collected_shapes/amir_cleaned.obj \
-o temp_output.obj \
-f 10000

# %%


from pyvista import examples
import pyacvd

# download cow mesh
cow = examples.download_cow()

# plot original mesh
# cow.plot(show_edges=True, color='w')




#%%

obj_file_path = '/home/ali/text2mesh/astro_utils/collected_shapes/sink/421eca272e20457f174cc62fa41808fc.obj'
# obj_file_path = '/home/ali/text2mesh/data/source_meshes/person.obj'

mesh = trimesh.load(obj_file_path, process=True, force='mesh')
print(len(mesh.vertices), len(mesh.faces))

# pt3dmesh = get_mesh(obj_file_path, device, normalize=False)
mesh.show(viewer="notebook")

mesh.vertices, mesh.faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
print(len(mesh.vertices), len(mesh.faces))

# save mesh to file
mesh.export('./421eca272e20457f174cc62fa41808fc_subdivided.obj')

