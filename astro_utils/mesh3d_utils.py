import os

import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import numpy as np
from pytorch3d.io import load_obj, save_ply
from pytorch3d.renderer import (FoVOrthographicCameras, FoVPerspectiveCameras,
                                Materials, MeshRasterizer, MeshRenderer,
                                PointsRasterizationSettings, PointsRasterizer,
                                PointsRenderer, RasterizationSettings,
                                SoftPhongShader, Textures, TexturesAtlas,
                                TexturesVertex, look_at_view_transform)
from pytorch3d.structures import Meshes
from pytorch3d.io import IO

def convert_to_mesh(verts, faces, normals=None, textures=None, device="cpu"):
    # convert faces and faces to torch.tensor if not already
    if not isinstance(verts, torch.Tensor):
        verts = torch.tensor(verts, dtype=torch.float32, device=device)
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, dtype=torch.long, device=device)
    if textures is None:
        # Create a texture for the mesh
        verts_rgb = torch.ones_like(verts)[None]
        textures = Textures(verts_rgb=verts_rgb.to(device))
    if normals is not None:
        if not isinstance(normals, torch.Tensor):
            normals = torch.tensor(normals, dtype=torch.float32).to(device)
        mesh = Meshes(verts=[verts], faces=[faces], verts_normals=[normals], textures=textures)
    else:
        mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return mesh

def save_mesh(mesh, filename):
    IO().save_mesh(mesh, filename)
    print(f'Saved mesh to {filename}')

def my_export_mesh(mesh, filename, sort_faces=True, avoid_culling_hack=True):
    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    normals = mesh.verts_normals_list()[0] 

    assert normals.shape == verts.shape, "normals and verts must have the same shape"

    with open(filename, "w+") as f:
        for vi, v in enumerate(verts):
            f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            f.write("vn %f %f %f\n" % (normals[vi, 0], normals[vi, 1], normals[vi, 2]))
        for face in faces:
            if sort_faces:
                face = sorted(face)
            f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))
            if avoid_culling_hack:
                # save the other rotation of the face to avoid culling
                f.write("f %d %d %d\n" % (face[0] + 1, face[2] + 1, face[1] + 1))
    print(f'Saved mesh to {filename}')

def get_mesh(obj_file_path, device, normalize=True):
    """
    Generates Meshes object and initializes the mesh with vertices, faces,
    and textures.

    Note: the mesh vertices are normalized to [-1, 1].

    Args:
        obj_filename: str, path to the 3D obj filename
        device: str, the torch device containing a device type ('cpu' or
        'cuda')

    Returns:
        mesh: Meshes object
    """
    # Get vertices, faces, and auxiliary information
    verts, faces, aux = load_obj(
        obj_file_path, device=device, load_textures=False, create_texture_atlas=False
    )

    if normalize:
        # normalize and center the vertices
        verts = verts - verts.mean(0)  # [N, 3]
        verts = verts / torch.max(torch.abs(verts))

    mesh = convert_to_mesh(verts, faces.verts_idx, aux.normals, None, device)
    return mesh


def get_renderer(image_size, dist, device, elev, azim):
    """
    Generates a mesh renderer by combining a rasterizer and a shader.

    Args:
        image_size: int, the size of the rendered .png image
        dist: int, distance between the camera and 3D object
        device: str, the torch device containing a device type ('cpu' or
        'cuda')
        elev: list, contains elevation values
        azim: list, contains azimuth angle values

    Returns:
        renderer: MeshRenderer class
    """
    # Initialize the camera with camera distance, elevation, azimuth angle,
    # and image size
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        bin_size=0,
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    # Initialize rasterizer by using a MeshRasterizer class
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    # The textured phong shader interpolates the texture uv coordinates for
    # each vertex, and samples from a texture image.
    shader = SoftPhongShader(device=device, cameras=cameras)
    # Create a mesh renderer by composing a rasterizer and a shader
    renderer = MeshRenderer(rasterizer, shader)
    return renderer


def render_image(renderer, mesh, obj_file_path, azim, elev, ax, save_image=False):
    """
    Renders an image using MeshRenderer class and Meshes object. Saves the
    rendered image as a .png file.

    Args:
        image_size: int, the size of the rendered .png image
        dist: int, distance between the camera and 3D object
        device: str, the torch device containing a device type ('cpu' or
        'cuda')
        elev: list, contains elevation values
        azim: list, contains azimuth angle values

    Returns:
        renderer: MeshRenderer class
    """
    image = renderer(mesh)
    ax.imshow(image[0, ..., :3].cpu().numpy())
    ax.set_title(f"{azim}_{elev}")

    if save_image:
        dir_to_save = "./output_meshes"
        os.makedirs(dir_to_save, exist_ok=True)
        out = os.path.normpath(obj_file_path).split(os.path.sep)
        mesh_filename = out[-1].split(".")[0]
        file_to_save = f"{mesh_filename}_elev{int(elev)}_azim{int(azim)}.png"
        filename = os.path.join(dir_to_save, file_to_save)
        plt.imsave(filename, image[0, ..., :3].cpu().numpy())
        print("Saved image as " + str(filename))


def compile_all_steps(image_size, dist, device, elev, azim, obj_file_path, ax):
    """
    Combines the above steps to read a mesh and render its image.

    Args:
        image_size: int, the size of the rendered .png image
        dist: int, distance between the camera and 3D object
        device: str, the torch device containing a device type ('cpu' or
        'cuda')
        elev: list, contains elevation values
        azim: list, contains azimuth angle values
        obj_filename: str, path to the 3D obj filename

    Returns:
        None
    """
    renderer = get_renderer(image_size, dist, device, elev, azim)
    mesh = get_mesh(obj_file_path, device)
    render_image(renderer, mesh, obj_file_path, azim, elev, ax)
    return None



def remove_duplicate_vertices(mesh):
    verts = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()
    normals = mesh.verts_normals_list()[0].cpu().numpy()
    print(f'Removing duplicate vertices...')
    print(f'\tBefore: {len(verts)=}, {len(faces)=}')

    # find duplicate vertices - brute force
    unique_verts_dict = defaultdict(list)
    for idx, vert in enumerate(verts):
        unique_verts_dict[tuple(vert)].append(idx)

    unique_vert_ids = [verts[0] for verts in unique_verts_dict.values()]
    old2new_vinds = {oldv: newv for newv, oldv in enumerate(unique_vert_ids)}

    for vert_coord, vert_ids in unique_verts_dict.items():
        for vert_id in vert_ids:
            old2new_vinds[vert_id] = old2new_vinds[vert_ids[0]]


    cleaned_faces = []
    # if the coordinates are the same, use the same vert_inds for all the faces
    for face in faces:
        new_face = [old2new_vinds[vertid] for vertid in face]
        cleaned_faces.append(new_face)
    # some of the old faces might now be duplicates.
    cleaned_faces = list({tuple(sorted(face)) for face in cleaned_faces})

    # remove faces that have duplicate vertices
    cleaned_faces = [face for face in cleaned_faces if len(set(face)) == 3]

    cleaned_verts = verts[unique_vert_ids]
    cleaned_normals = normals[unique_vert_ids]

    cleaned_mesh = convert_to_mesh(cleaned_verts, cleaned_faces, cleaned_normals, None, device=mesh.device)
    print(f'\tAfter: {len(cleaned_verts)=}, {len(cleaned_faces)=}')
    return cleaned_mesh


def upsample_mesh_based_on_edge_length(mesh, len_threshold=0.2):
    """
    If an edge is long, break it into two edges in the middle and break each of its faces into two faces.
    """
    verts = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()
    normals = mesh.verts_normals_list()[0].cpu().numpy()
    print('Upsampling mesh based on edge length')
    print(f'\tBefore: {len(verts)=}, {len(normals)=}, {len(faces)=}')

    edges2faceids = defaultdict(set) # edge (vi, vj) -> {face_inds}
    for i, face in enumerate(faces):
        for j in range(3):
            edges2faceids[
                tuple(sorted([face[j], face[(j + 1) % 3]]))
                ].add(i)

    # If an edge in the mesh is too big, break the two triangles sharing this edge into 2 triangles each.
    # This is done by adding a new vertex to the middle of the edge.
    # It is complicated to break a single triangle into two triangles more than once even if it
    # has two large edges. So we keep track of the faces that have already been broken.
    verts_to_add = []
    normals_to_add = []
    faces_to_add = []
    already_broken_faces = set() # these are also the faces to remove from the faces list
    for edge, face_ids in edges2faceids.items():
        # if euclidean lengh if the edge is too big, break it into two triangles
        edge_len = np.linalg.norm(verts[edge[0]] - verts[edge[1]], ord=2)
        if edge_len > len_threshold:
            # Otherwise this edge can't be broken because it shares an already 
            # broken face so we skip it. We can break it in a later iteration.
            if len(face_ids & already_broken_faces) == 0:
                for faceid in face_ids:
                    # find the other vert of the face
                    other_vert = set(faces[faceid]) - set(edge)
                    assert len(other_vert) == 1, f'this function only works for triangles. {other_vert}, {edge}, {faceid}, {faces[faceid]}'
                    other_vert = other_vert.pop()
                    faces_to_add.append([edge[0], other_vert, len(verts) + len(verts_to_add)])
                    faces_to_add.append([edge[1], other_vert, len(verts) + len(verts_to_add)])

                # find the middle point of the edge and add it to the verts list
                mid_point = (verts[edge[0]] + verts[edge[1]]) / 2
                verts_to_add.append(mid_point)
                # find the normal of the vert and add it to the normals list
                mid_point_normal = (normals[edge[0]] + normals[edge[1]]) / 2
                normals_to_add.append(mid_point_normal)
                # finally mark the faces that share this edge as broken
                already_broken_faces |= face_ids
                
    # remove the old faces
    faces = np.delete(faces, list(already_broken_faces), axis=0)
    # add the new vertices and faces to the mesh
    if verts_to_add:
        verts = np.append(verts, verts_to_add, axis=0)
        normals = np.append(normals, normals_to_add, axis=0)
    if faces_to_add:
        faces = np.append(faces, faces_to_add, axis=0)
    if not verts_to_add and not faces_to_add:
        print('\tWARNING: No faces were added....')

    print(f'\tAfter: {len(verts)=}, {len(normals)=}, {len(faces)=}')

    mesh_upsampled = convert_to_mesh(verts, faces, normals=normals, textures=None, device=mesh.device)
    return mesh_upsampled



def upsample_mesh_based_on_triangular_face_area(mesh, area_threshold=0.1):
    """
    If the area of a triangles is too big, break the face into 3 smaller triangles.
    """
    verts = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()
    normals = mesh.verts_normals_list()[0].cpu().numpy()

    print('Upsampling mesh based on triangular face area')
    print(f'\tBefore: {len(verts)=}, {len(normals)=}, {len(faces)=}')

    # get the area of each face
    face_areas = np.zeros(len(faces))
    for i, face in enumerate(faces):
        face_areas[i] = np.linalg.norm(np.cross(verts[face[1]] - verts[face[0]], verts[face[2]] - verts[face[0]])) / 2

    # if the face area is larger than a threshold, we break the face into 3 triangles
    # by adding a new vertex to the middle of the face
    verts_to_add = []
    normals_to_add = []
    faces_to_add = []
    face_inds_to_remove = []
    for i, face in enumerate(faces):
        if face_areas[i] > area_threshold:
            # get the middle of the face
            center = np.mean(verts[face], axis=0)
            # add the new triangles
            faces_to_add.append([face[0], face[1], len(verts) + len(verts_to_add)])
            faces_to_add.append([face[1], face[2], len(verts) + len(verts_to_add)])
            faces_to_add.append([face[2], face[0], len(verts) + len(verts_to_add)])
            # add the center point to the vertices and normals
            verts_to_add.append(center)
            center_normal = np.mean(normals[face], axis=0)
            normals_to_add.append(center_normal)
            # remove the old face
            face_inds_to_remove.append(i)

    # remove the old faces
    faces = np.delete(faces, face_inds_to_remove, axis=0)
    # add the new vertices and faces to the mesh
    if verts_to_add:
        verts = np.append(verts, verts_to_add, axis=0)
        normals = np.append(normals, normals_to_add, axis=0)
    if faces_to_add:
        faces = np.append(faces, faces_to_add, axis=0)
    if not verts_to_add and not faces_to_add:
        print('\tWARNING: No faces were added....')

    print(f'\tAfter: {len(verts)=}, {len(normals)=}, {len(faces)=}')

    mesh_upsampled = convert_to_mesh(verts, faces, normals=normals, textures=None, device=mesh.device)
    return mesh_upsampled

def upsample(mesh, n_area, area_threshold, n_edge, len_threshold):
    mesh = remove_duplicate_vertices(mesh)
    for i in range(n_area): 
        print(i)
        mesh = upsample_mesh_based_on_triangular_face_area(mesh, area_threshold)
    for i in range(n_edge):
        print(i)
        mesh = upsample_mesh_based_on_edge_length(mesh, len_threshold)
    return mesh

def upsample_v2(mesh, n, area_threshold, len_threshold):
    mesh = remove_duplicate_vertices(mesh)
    for i in range(n):
        print(i)
        mesh = upsample_mesh_based_on_triangular_face_area(mesh, area_threshold)
        mesh = upsample_mesh_based_on_edge_length(mesh, len_threshold)
    return mesh