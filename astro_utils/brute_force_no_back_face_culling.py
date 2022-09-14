#%%
# Read the mesh from obj file and brute force add faces in both directions to avoid back face culling.

import trimesh  
import numpy as np
import os 

# the folder containing the .obj files you want to process
root_dir = '/Users/amir2/Desktop/reza-txt2mesh/text2mesh/astro_utils/collected_shapes/meshlab_remeshed'

# a folder to store the output meshes
out_root_dir = '/Users/amir2/Desktop/reza-txt2mesh/text2mesh/astro_utils/collected_shapes/meshlab_remeshed/brute_force_no_culling'

# %%
# The meshse are first upsampled-remeshed using meshlab.
# SIMPLY WRITE BOTH FACE DIRECTIONS and save the mesh
def brute_force_no_back_face_culling(mesh, out_file_path):
    faces_reversed = mesh.faces[:, [0, 2, 1]]
    # append reversed faces to the end of the mesh.faces
    mesh.faces = np.append(mesh.faces, faces_reversed, axis=0)
    # print(f'{len(mesh.vertices)=}, {len(mesh.faces)=}')

    # save the mesh to obj file
    _ = mesh.export(out_file_path)
    print(f'{out_file_path} saved')


for file_name in os.listdir(root_dir):
    print(file_name)
    obj_file_path = os.path.join(root_dir, file_name)
    out_file_path = os.path.join(out_root_dir, file_name)
    os.makedirs(out_root_dir, exist_ok=True)

    mesh = trimesh.load(obj_file_path, process=True, force='mesh')
    brute_force_no_back_face_culling(mesh, out_file_path)


# %%

exit()
# Below are other attempts to do the same thing.

obj_file_path = '/Users/amir2/Desktop/reza-txt2mesh/text2mesh/astro_utils/collected_shapes/meshlab_remeshed/cap2_14df58bdedfbb41141edb96d309c9a23.obj'
# obj_file_path = '/Users/amir2/Desktop/reza-txt2mesh/text2mesh/astro_utils/collected_shapes/cap/14df58bdedfbb41141edb96d309c9a23.obj'
out_file_path = './temp.obj'

mesh = trimesh.load(obj_file_path, process=True, force='mesh')
print(f'{len(mesh.vertices)=}, {len(mesh.faces)=}')

mesh.show()

mesh_centroid = mesh.centroid
print(f'{mesh_centroid=}')

# Since strict=True is not available in python 3.9, lest manually check if they have same size
assert len(mesh.faces) == len(mesh.face_normals), 'faces and face_normals have different size'
num_faces = len(mesh.faces)

# for each face if it is facing center of mass, then re-orient it
# if it is not, then keep it
for face, facen, fid in zip(mesh.faces, mesh.face_normals, range(num_faces)):
    # print(face, facen)
    # https://stackoverflow.com/a/40465409
    if facen.dot(mesh_centroid - mesh.vertices[face[0]]) > 0:
        # print('flipping face')
        # print(mesh.faces[fid])
        mesh.faces[fid] = face[::-1]
        # print(mesh.faces[fid])

mesh.show()


# %%
trimesh.repair.fix_inversion(mesh, multibody=True)
trimesh.repair.fix_normals(mesh, multibody=True)
trimesh.repair.fix_winding(mesh)

mesh.show()


# %%

# The above works for convex meshes, but not for non-convex meshes
# for example, the tip of a cap should not always be facing the center of mass.
# so for each edge we check if all tirangles that share it have same orientation.
flipped_faces = set()
for fid1, fid2 in mesh.face_adjacency:
    # print(fid1, '#', fid2)
    # print(mesh.faces[fid1], '#', mesh.faces[fid2])
    # print(mesh.face_normals[fid1], '#', mesh.face_normals[fid2])
    # check if they have same orientation
    if fid2 in flipped_faces:
        continue

    if mesh.face_normals[fid1].dot(mesh.face_normals[fid2]) < 0:
        # flip the face if two adjacent faces have opposite orientation
        print(mesh.face_normals[fid1], '\n', mesh.face_normals[fid2])
        mesh.faces[fid2] = mesh.faces[fid2][::-1]
        print(mesh.face_normals[fid1], '\n', mesh.face_normals[fid2])
        flipped_faces.add(fid2)
        break

print(f'{len(flipped_faces)=}')

mesh.show(show_edges=True)

# %%
visited_faces = set()
for fid in range(num_faces):
    visited_faces.add(fid1)
    # get all the neighbors of this face
    face_rows, face_cols = np.where(mesh.face_adjacency == fid)
    adj_face_rows = face_rows
    adj_face_cols = (face_cols + 1) % 2
    adjacent_faces = mesh.face_adjacency[adj_face_rows, adj_face_cols]

    # check if they have same orientation
    for fid2 in adjacent_faces:
        if fid2 in visited_faces:
            continue
        if mesh.face_normals[fid].dot(mesh.face_normals[fid2]) < 0:
            # flip the face if two adjacent faces have opposite orientation
            # print(mesh.face_normals[fid], '\n', mesh.face_normals[fid2])
            mesh.faces[fid2] = mesh.faces[fid2][::-1]
            # print(mesh.face_normals[fid], '\n', mesh.face_normals[fid2])
            # break
    # break

mesh.show(show_edges=True)

# %%

# save the mesh to obj file
_ = mesh.export(out_file_path)
print(f'{out_file_path} saved')