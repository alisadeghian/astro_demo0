# %% 
# find a mesh's front view
import argparse

import matplotlib.pyplot as plt
# %matplotlib notebook
import numpy as np
import torch

from mesh import Mesh
from Normalization import MeshNormalizer
from render import Renderer
import sys
# sys.argv = ['--frontview_center', '0', '1']
sys.argv = ['']

parser = argparse.ArgumentParser()
parser.add_argument('--obj_path', type=str, 
    default='data/brute_force_no_culling_rotated/sink3_99ffc34e3e5019c319620b61f6587b3e.obj')
# parser.add_argument('--obj_path', type=str, 
    # default='data/brute_force_no_culling_rotated/cap2_14df58bdedfbb41141edb96d309c9a23.obj')
parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
parser.add_argument('--mesh_normalizer_func', type=str, default='bounding_sphere')

args = parser.parse_args()

print(args.frontview_center)

mesh = Mesh(args.obj_path)
# rotate the mesh 90 degrees 
MeshNormalizer(mesh, normalizer=args.mesh_normalizer_func)()

render = Renderer(dim=(512, 512))

with torch.no_grad():
    for center_elev in np.linspace(-np.pi, np.pi, 10):
        for center_azim in np.linspace(-np.pi, np.pi, 10):
            # center_azim, center_elev = 1.57, 0.1 
            # convert to torch float tensor
            center_elev = torch.tensor([center_elev], dtype=torch.float32)
            center_azim = torch.tensor([center_azim], dtype=torch.float32)
            print(f'center_evel: {center_elev}, center_azim: {center_azim}')
            rendered_images, elev, azim = render.render_front_views(mesh, num_views=1, # TODO: add as parameter
                                                                    show=False,
                                                                    center_azim=args.frontview_center[0] + center_azim,
                                                                    center_elev=args.frontview_center[1] + center_elev,
                                                                    camera_r=2.0,
                                                                    std=100, 
                                                                    return_views=True,
                                                                    background=None,
                                                                    frontview_elev_std=1,
                                                                    background_image_mode=None)

            # plot rendered_images
            rendered_images = rendered_images.cpu().numpy().squeeze()
            plt.imshow(np.transpose(rendered_images, (1, 2, 0)))
            plt.show()
            break
        break

plt.close('all')

# %%
