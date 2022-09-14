#%%
import pyacvd
import pyvista as pv


# download cow mesh
# mesh = examples.download_cow()
obj_file_path = '/Users/amir2/Desktop/reza-txt2mesh/text2mesh/astro_utils/collected_shapes/sink/421eca272e20457f174cc62fa41808fc.obj'
out_file_path = './temp.obj'
mesh = pv.read(obj_file_path)

# write mesh to obj file
def save_mesh_to_obj(mesh, out_file_path):
    pl = pv.Plotter()
    _ = pl.add_mesh(mesh)
    pl.export_obj(out_file_path)

# %%
# plot original mesh
# mesh.plot(show_edges=True, color='w')

mesh_clustered = pyacvd.Clustering(mesh)
# mesh is not dense enough for uniform remeshing
mesh_clustered.subdivide(3)
mesh_clustered.cluster(1000)

# plot clustered cow mesh
# mesh_clustered.plot()

# remesh
remeshed = mesh_clustered.create_mesh()

# plot uniformly remeshed cow
remeshed.plot(color='w', show_edges=True)

# %%

mesh_clustered = pyacvd.Clustering(remeshed)
# mesh is not dense enough for uniform remeshing
mesh_clustered.subdivide(10)
mesh_clustered.cluster(1000)

# remesh
remeshed = mesh_clustered.create_mesh()

# plot uniformly remeshed cow
remeshed.plot(color='w', show_edges=True)

# %%

# write mesh to obj file
def save_mesh_to_obj(mesh, out_file_path):
    pl = pv.Plotter()
    _ = pl.add_mesh(mesh)
    pl.export_obj(out_file_path)

save_mesh_to_obj(remeshed, out_file_path)

print('done')
# %%
