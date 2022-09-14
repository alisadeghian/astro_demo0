import torch


class Normalizer:
    @classmethod
    def get_bounding_box_normalizer(cls, x):
        shift = torch.mean(x, dim=0)
        scale = torch.max(torch.norm(x-shift, p=1, dim=1))
        return Normalizer(scale=scale, shift=shift)

    @classmethod
    def get_bounding_sphere_normalizer(cls, x):
        shift = torch.mean(x, dim=0)
        scale = torch.max(torch.norm(x-shift, p=2, dim=1))
        # print(f'Bounding sphere scale: {scale}, shift: {shift}')
        return Normalizer(scale=scale, shift=shift)
    
    @classmethod
    def get_min_max_normalizer(cls, points):
        """ Normalize points to [-0.5, 0.5]^3 keeping the original aspect ratio. """
        min_, _ = torch.min(points, dim=0)
        max_, _ = torch.max(points, dim=0)
        scale = torch.max(max_ - min_) / 0.95
        # shift = min_ + scale / 2.0
        shift = (min_ + max_) / 2.0
        # print(f'Min max scale: {scale}, shift: {shift}')
        return Normalizer(scale=scale, shift=shift)

    def __init__(self, scale, shift):
        self._scale = scale
        self._shift = shift

    def __call__(self, x):
        return (x-self._shift) / self._scale

    def get_de_normalizer(self):
        inv_scale = 1 / self._scale
        inv_shift = -self._shift / self._scale
        return Normalizer(scale=inv_scale, shift=inv_shift)