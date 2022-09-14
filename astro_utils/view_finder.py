# W.I.P
from Normalization import MeshNormalizer
from render import Renderer
from mesh import Mesh
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--obj_path', type=str, default='/Users/amir2/Desktop/reza-txt2mesh/text2mesh/data/source_meshes/person.obj')
args = parser.parse_args()

render = Renderer()
mesh = Mesh(args.obj_path)
MeshNormalizer(mesh)()

for frontview_center in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]:
    renders, elev, azim = render.render_front_views(mesh, num_views=5,
                                                    show=True,
                                                    center_azim=frontview_center[0],
                                                    center_elev=frontview_center[1],
                                                    std=100,
                                                    return_views=True,
                                                    background=torch.tensor([1, 1, 1]))
    # view the rendered views
