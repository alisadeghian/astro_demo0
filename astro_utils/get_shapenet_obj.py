# %%
# DEPRECATED in favor of read_shapnetsem_obj_files.py
# 
from tabnanny import verbose
import pandas as pd
import open3d as o3d
import numpy as np

import os

model_root_dir = '/data/ShapeNetSem/models/'
file_path = '/data/ShapeNetSem/metadata.csv'
# categories.synset.csv'
# 
# synset_id = 'n4230655' # sink
# synset_id = 'n17402' # plant

df = pd.read_csv(file_path)
# fill all nan values with 'NAN'
df.fillna('NAN', inplace=True)

rotation_ctr = 0
def rotate_view(vis):
    global rotation_ctr
    rotation_x = 10.0

    # set background to balck
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.995, 0.995, 0.995])
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = True
    # opt.point_size = 5.0
    # opt.show_coordinate_frame = True
    # opt.line_width = 1.0

    # rotate the view
    ctr = vis.get_view_control()
    ctr.rotate(rotation_x, 0.0) # see https://stackoverflow.com/questions/62065410/what-is-the-argument-type-for-get-view-control-rotate-in-open3d
    rotation_ctr += 1
    if rotation_ctr * rotation_x > 2090:
        rotation_ctr = 0
        vis.close()


# %%
# df[df.wnsynset == synset_id].head()
# df_subset = df[df.wnsynset == synset_id]

df_subset = df.copy()
target_category = 'sink' #[xxx, 11523, 2720, 7022]
# target_category = 'plant' #[223, 541, 1070]
target_category = 'rose' #[5898, 6388]
# target_category = 'dollar'

print(df_subset[df_subset.name.str.contains(target_category)].shape)

df_subset[df_subset.name.str.contains(target_category)][['fullId','wnlemmas','name','tags']].iloc[:20]

# %%
obj_file_path = df_subset.loc[4003].fullId
obj_file_path = os.path.join(model_root_dir, obj_file_path[len('wss.'):]) + '.obj'
print(f'obj file path: {obj_file_path}')
# open the 3d object file

mesh = o3d.io.read_triangle_mesh(obj_file_path) 
mesh.paint_uniform_color([1, 0.706, 0])

# %%
# visualize the mesh
o3d.visualization.draw_geometries([mesh])

#%%

# o3d.io.write_triangle_mesh('./lowres'+target_category+'_2.obj', mesh, write_vertex_colors=False)

# %%
# updample and save the mesh
from obj2gif import up_sample_mesh

print(f"num of vertices: {np.asanyarray(mesh.vertices).shape}")
print(f"num of faces/triangles: {np.asanyarray(mesh.triangles).shape}\n\n")
# print(f"has_triangle_normals {mesh.has_triangle_normals()}")
# print(f"has_vertex_normals {mesh.has_vertex_normals()}")

mesh_hq = up_sample_mesh(mesh, n_iter=5, verbose=False, clean_mesh=False)

for i in range(1):
    mesh_hq.remove_non_manifold_edges()
    mesh_hq.remove_duplicated_triangles()
    mesh_hq.remove_duplicated_vertices()
    mesh_hq.remove_degenerate_triangles()
    mesh_hq.remove_unreferenced_vertices()
o3d.visualization.draw_geometries([mesh_hq])

print(f"num of vertices: {np.asanyarray(mesh_hq.vertices).shape}")
print(f"num of faces/triangles: {np.asanyarray(mesh_hq.triangles).shape}\n\n")
# print(f"has_triangle_normals {mesh_hq.has_triangle_normals()}")
# print(f"has_vertex_normals {mesh_hq.has_vertex_normals()}")

# %%
# Save the upsamples mesh
up_sampled_path = './'+target_category+'_3.obj'
print(f'Saving the upsamples mesh to {up_sampled_path}')
o3d.io.write_triangle_mesh(up_sampled_path, mesh_hq, write_vertex_colors=False)
