import torch
import numpy as np
from mesh import Mesh
from Normalization import MeshNormalizer
from render import Renderer
import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument('--obj_path', type=str, default='data/source_meshes/person.obj')
parser.add_argument('--background', nargs=3, type=float, default=(1.0, 1.0, 1.0))
# ['green', 'white', 'black', 'red', 'blue', 'random_noise', 'checkerboard'], fixed_image
parser.add_argument('--background_image_mode', type=str, default='fixed_image', 
                    help='None to turn off, "change_per_iter" to randomly changing per iter.') 
parser.add_argument('--mesh_normalizer_func', type=str, default='min_max',
                    help='bounding_sphere or min_max')
parser.add_argument('--frontview_center', nargs=2, type=float, default=[0.0, 0.0]) # horse [3., 0.15], cap/sink [1.57, 0.1]
parser.add_argument('--camera_r', type=float, default=1.25)
parser.add_argument('--renderer_shape', nargs=2, type=int, default=[768, 432]) #[768, 432]
parser.add_argument('--n_frames', type=int, default=70)
parser.add_argument('--prior_color_fill', type=float, default=0.8)
args = parser.parse_args()


obj_name = args.obj_path.split('/')[-1].split('.')[0]
root_dir = './videos360/'
video_frames_path = root_dir + f'person2/{obj_name}_{args.camera_r}_{args.mesh_normalizer_func}_G{args.background_image_mode}1_' + \
    f'{args.renderer_shape[0]}x{args.renderer_shape[1]}_{args.n_frames}_{args.prior_color_fill}' + '/'
# create the video frames directory
os.makedirs(video_frames_path, exist_ok=True)

device = 'cuda:0' 

mesh = Mesh(args.obj_path)
MeshNormalizer(mesh, normalizer=args.mesh_normalizer_func)()

prior_color = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=args.prior_color_fill, device=device)
mesh.set_mesh_color(prior_color)

background = None
# if args.background is not None:
#     assert len(args.background) == 3
#     background = torch.tensor(args.background).to(device)

render = Renderer(dim=(args.renderer_shape[0], args.renderer_shape[1]))

def save_rendered_image(rendered_image, frame_idx):
    rendered_image = rendered_image[0].cpu().numpy()
    rendered_image = np.moveaxis(rendered_image, 0, -1)
    rendered_image = np.clip(rendered_image, 0, 1)
    rendered_image = rendered_image * 255
    rendered_image = rendered_image.astype(np.uint8)
    rendered_image = Image.fromarray(rendered_image)
    # set file name to a 5 digit number (e.g. 00000.jpg)
    frame_path = video_frames_path + f'{frame_idx:05d}.jpg'
    rendered_image.save(frame_path)
    print(f'save image to {frame_path}')

with torch.no_grad():
    frame_idx = 0
    # for center_azim in np.linspace(-np.pi, np.pi, args.n_frames+1)[:-1]:
    for center_azim in np.linspace(-np.pi/4, np.pi/4, args.n_frames):
        for center_elev in np.linspace(0, 0, 1):
            center_elev = torch.tensor([center_elev], dtype=torch.float32)
            center_azim = torch.tensor([center_azim], dtype=torch.float32)
            print(f'center_evel: {center_elev}, center_azim: {center_azim}')
            save_camera_poses_path = video_frames_path + f'camera_poses/'
            os.makedirs(save_camera_poses_path, exist_ok=True)
            save_camera_poses_path += f'{frame_idx:05d}_'
            rendered_image, elev, azim = render.render_front_views(mesh, num_views=1, # TODO: add as parameter
                                                                    show=False,
                                                                    center_azim=args.frontview_center[0] + center_azim,
                                                                    center_elev=args.frontview_center[1] + center_elev,
                                                                    camera_r=args.camera_r,
                                                                    std=1000, #TODO: add as parameter
                                                                    return_views=True,
                                                                    background=background,
                                                                    frontview_elev_std=1,
                                                                    background_image_mode=args.background_image_mode,
                                                                    save_camera_poses_path=save_camera_poses_path)
            save_rendered_image(rendered_image, frame_idx)
            frame_idx += 1

# save args to a text file
with open(video_frames_path + 'args.txt', 'w') as f:
    f.write(str(args))
    