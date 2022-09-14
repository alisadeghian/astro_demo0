import os
import copy

import numpy as np
from PIL import Image
import open3d as o3d

# http://www.open3d.org/docs/0.9.0/tutorial/Advanced/pointcloud_outlier_removal.html


# root_dir = '/Users/amir2/Downloads/roman_soldier_best_send_to_amir 2/new additions/'
# root_dir = '/Users/amir2/Downloads/results 14/demo/roman_resnet_normratio'
# root_dir = '/Users/amir2/Downloads/coca/demo/coca'
# root_dir = "/Users/amir2/Downloads/results 15/demo/sink"
# root_dir = "/Users/amir2/Downloads/results2 4/demo/shoe"
# root_dir = "/Users/amir2/Downloads/results 16/demo/rose1"
# root_dir = "/Users/amir2/Downloads/results 16/demo/sink_nocolor_sample"
# root_dir = "/Users/amir2/Downloads/results 16/demo/sink_nocolor1"
# root_dir = "/Users/amir2/Downloads/results 17/demo/dollar_2"
# root_dir = '/Users/amir2/Downloads/results 17/demo/colab_dollar_2'
root_dir = '/Users/amir2/Downloads/snoop_dogg_examples'

screen_captures = []
rotation_ctr = 0


def rotate_view(vis):
    rotation_x = 10.0
    global screen_captures
    global rotation_ctr

    # set background to balck
    opt = vis.get_render_option()
    # opt.background_color = np.asarray([0.995, 0.995, 0.995]) - 0.995
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    # opt.mesh_show_back_face = False
    # opt.mesh_show_wireframe = False
    # opt.point_size = 5.0
    # opt.show_coordinate_frame = True
    # opt.line_width = 1.0

    import time
    time.sleep(0.010)

    # rotate the view
    view_control = vis.get_view_control()
    if rotation_ctr > 0:
        view_control.camera_local_translate(forward=0.45, right=0, up=0)
    if rotation_ctr == 0:
        pass
        view_control.rotate(-800, 0)
        # view_control.translate(20, 0)
        # view_control.rotate(-120, -80.0)
        # view_control.rotate(-150, -520.0) # sink?
        # view_control.rotate(0, -360.0) # sink!?
        # view_control.rotate(-300, 0) # sink!?
        # view_control.rotate(-120, -510.0) # sink!!!
        # view_control.rotate(+140, 0) # sink!!!
        # view_control.rotate(-120, -240.0) # rose
        # view_control.rotate(150, 200.0) # money
        # view_control.rotate(100.0, 0) # money
    view_control.rotate(rotation_x, 0.0)

    # save the visualization to file
    view_control.camera_local_translate(forward=-0.45, right=0, up=0)

    screen_float_buffer = vis.capture_screen_float_buffer()
    screen_captures.append(screen_float_buffer)
    rotation_ctr += 1

    # see https://stackoverflow.com/questions/62065410/what-is-the-argument-type-for-get-view-control-rotate-in-open3d
    if rotation_ctr * rotation_x > 2094/4.0:
        rotation_ctr = 0
        vis.close()


def voxelize_mesh(input_mesh, voxel_size=0.05):
    """converts a mesh to a voxel grid and siplays it"""
    mesh = copy.deepcopy(input_mesh)
    # Fit to unit cube.
    mesh.scale(
        1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
        center=mesh.get_center(),
    )
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
    # mesh, voxel_size=0.03) # doesn't include color
    pcl = o3d.geometry.PointCloud(mesh.vertices)
    pcl.colors = mesh.vertex_colors
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcl, voxel_size)
    o3d.visualization.draw([voxel_grid])


def up_sample_mesh(
    input_mesh, method="subdivide_midpoint", clean_mesh=True, verbose=False, n_iter=4
):
    """Upsample the mesh. Note that the returned mesh is normalized."""
    mesh = copy.deepcopy(input_mesh)

    if verbose:
        print(f"Up sampling mesh using {method}")
        o3d.visualization.draw_geometries([mesh])

    ## Fit to unit cube.
    mesh.scale(
        1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
        center=mesh.get_center(),
    )

    if method == "uniform":
        # Upsample by uniform sampling the mesh
        pcl = mesh.sample_points_uniformly(
            number_of_points=500000, use_triangle_normal=True, seed=-1
        )
        if verbose:
            o3d.visualization.draw([pcl], point_size=1)
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=12)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcl, alpha=0.01)

        # estimate radius for rolling ball
        distances = pcl.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = o3d.utility.DoubleVector([1.5 * avg_dist, 3 * avg_dist])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcl, radii
        )
        if verbose:
            o3d.visualization.draw_geometries([mesh])

    if method == "subdivide_loop":
        mesh = mesh.subdivide_loop(n_iter)
        if verbose:
            o3d.visualization.draw_geometries([mesh])

    if method == "subdivide_midpoint":
        mesh = mesh.subdivide_midpoint(n_iter)
        if verbose:
            o3d.visualization.draw_geometries([mesh])

    if method == "poisson":
        ## convert to point cloud and upsample using Poisson surface reconstruction method.
        pcl = o3d.geometry.PointCloud(mesh.vertices)
        if verbose:
            o3d.visualization.draw([pcl])
        pcl.colors = mesh.vertex_colors
        # pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # print(np.asarray(pcl.normals).shape)
        pcl.normals = mesh.vertex_normals
        if verbose:
            print(f"normals shape {np.asarray(pcl.normals).shape}")

        # Upsampled mesh...
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcl, depth=9
        )
        # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #         pcd, o3d.utility.DoubleVector(radii))
        pcl = o3d.geometry.PointCloud(mesh.vertices)

        if verbose:
            o3d.visualization.draw([pcl])
            o3d.visualization.draw([mesh])

    if verbose:
        print(f" num of vertices: {np.asanyarray(mesh.vertices).shape}")
        print(f" num of faces/triangles: {np.asanyarray(mesh.triangles).shape} ")

    if clean_mesh:
        # simplify the mesh a bit
        if verbose:
            print("Cleaning...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        # mesh.simplify_quadric_decimation(target_number_of_triangles=40_000, maximum_error=1000.0, boundary_weight=2.0)
        if verbose:
            print(f" num of vertices: {np.asanyarray(mesh.vertices).shape}")
            print(f" num of faces/triangles: {np.asanyarray(mesh.triangles).shape} ")
            o3d.visualization.draw_geometries([mesh])

    return mesh

if __name__ == '__main__':
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".obj"):
                # get the file path
                obj_file_path = os.path.join(root, file)
                print(f"Reading file path: {obj_file_path}")

                # if not (('sink_3' in obj_file_path) ):
                    # continue

                ## Read the file
                mesh = o3d.io.read_triangle_mesh(
                    obj_file_path, enable_post_processing=True, print_progress=True
                )

                print(f"num of vertices: {np.asanyarray(mesh.vertices).shape}")
                print(f"num of faces/triangles: {np.asanyarray(mesh.triangles).shape} ")
                print(f"has_triangle_normals {mesh.has_triangle_normals()}")
                print(f"has_vertex_normals {mesh.has_vertex_normals()}")

                ## Up-sample the mesh
                # mesh = up_sample_mesh(mesh, verbose=False)

                if False:
                    ## Re-color the mesh
                    # increase the color intensity of the mesh (here we make it darker)
                    mesh_colors = np.asarray(mesh.vertex_colors)
                    # print(mesh_colors)

                    # swap the R and B channels
                    # mesh_colors[:, 0] = mesh_colors[:, 0]*1.05
                    # mesh_colors[:, 1] = mesh_colors[:, 1]*1.0
                    # mesh_colors[:, 2] = mesh_colors[:, 2]*1.05
                    # mesh_colors[mesh_colors >= 1.0] = 1.0
                    # mesh_colors[mesh_colors <= 0.0] = 0.0
                    # print(f'WARNING')

                    # make it darker
                    mesh_colors = mesh_colors * 0.95975
                    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

                ## Voxelize the mesh
                # voxelize_mesh(mesh, 0.02)

                ## Visualize the mesh
                # o3d.visualization.draw_geometries([mesh])
                o3d.visualization.draw_geometries_with_animation_callback(
                    [mesh], rotate_view
                )
                # o3d.visualization.draw_geometries_with_editing([mesh])

                ## Save the screen captures to a gif
                parent_folder = os.path.dirname(obj_file_path).split("/")[-2:]
                parent_folder = "_".join(parent_folder)
                gif_file_path = (
                    "./gifs/" + parent_folder + "_" + file.replace(".obj", ".gif")
                )
                print(f"Saving to {gif_file_path}")
                # Removing the first frame, because sometimes it has white bakground
                screen_captures = screen_captures[1:]
                screen_captures = [np.asarray(img) for img in screen_captures]
                screen_captures = [
                    Image.fromarray(np.uint8(img * 255)) for img in screen_captures
                ]
                screen_captures[0].save(
                    gif_file_path,
                    save_all=True,
                    append_images=screen_captures[1:],
                    duration=50,
                    loop=0,
                )

                # empty for next object
                screen_captures = []
                # exit()