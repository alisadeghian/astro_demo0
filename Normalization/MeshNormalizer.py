import copy
from . import Normalizer


class MeshNormalizer:
    def __init__(self, mesh, normalizer='bounding_sphere'):
        self._mesh = mesh  # original copy of the mesh
        if normalizer == 'bounding_sphere':
            self.normalizer = Normalizer.get_bounding_sphere_normalizer(self._mesh.vertices)
        elif normalizer == 'min_max':
            self.normalizer = Normalizer.get_min_max_normalizer(self._mesh.vertices)
        else:
            raise ValueError(f'Unknown normalizer: {normalizer}')
            
    def __call__(self):
        self._mesh.vertices = self.normalizer(self._mesh.vertices)
        return self._mesh

