from mesh import Mesh
import kaolin as kal
from utils import get_camera_from_view2
import matplotlib.pyplot as plt
from utils import device
import torch
import numpy as np


class Renderer():

    def __init__(self, mesh='sample.obj',
                 lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 camera=kal.render.camera.generate_perspective_projection(np.pi / 3).to(device),
                 dim=(224, 224)):

        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim

        self.background_tensor_cache = dict()

    def render_y_views(self, mesh, num_views=8, show=False, lighting=True, background=None, mask=False):

        faces = mesh.faces
        n_faces = faces.shape[0]

        azim = torch.linspace(0, 2 * np.pi, num_views + 1)[:-1]  # since 0 =360 dont include last element
        # elev = torch.cat((torch.linspace(0, np.pi/2, int((num_views+1)/2)), torch.linspace(0, -np.pi/2, int((num_views)/2))))
        elev = torch.zeros(len(azim))
        images = []
        masks = []
        rgb_mask = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)

            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    # ax.imshow(images[i].permute(1,2,0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        return images

    def render_single_view(self, mesh, elev=0, azim=0, show=False, lighting=True, background=None, radius=2,
                           return_mask=False):
        # if mesh is None:
        #     mesh = self._current_mesh
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        camera_transform = get_camera_from_view2(torch.tensor(elev), torch.tensor(azim), r=radius).to(device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection, camera_transform=camera_transform)

        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # Debugging: color where soft mask is 1
        # tmp_rgb = torch.ones((224,224,3))
        # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1,0,0]).float()
        # rgb_mask.append(tmp_rgb)

        if background is not None:
            image_features, mask = image_features

        image = torch.clamp(image_features, 0.0, 1.0)

        if lighting:
            image_normals = face_normals[:, face_idx].squeeze(0)
            image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
            image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
            image = torch.clamp(image, 0.0, 1.0)

        if background is not None:
            background_mask = torch.zeros(image.shape).to(device)
            mask = mask.squeeze(-1)
            assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
            background_mask[torch.where(mask == 0)] = background
            image = torch.clamp(image + background_mask, 0., 1.)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(figsize=(89.6, 22.4))
                axs.imshow(image[0].cpu().numpy())
                # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        if return_mask == True:
            return image.permute(0, 3, 1, 2), mask
        return image.permute(0, 3, 1, 2)

    def render_uniform_views(self, mesh, num_views=8, show=False, lighting=True, background=None, mask=False,
                             center=[0, 0], radius=2.0):

        # if mesh is None:
        #     mesh = self._current_mesh

        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[
               :-1]  # since 0 =360 dont include last element
        elev = torch.cat((torch.linspace(center[1], np.pi / 2 + center[1], int((num_views + 1) / 2)),
                          torch.linspace(center[1], -np.pi / 2 + center[1], int((num_views) / 2))))
        images = []
        masks = []
        background_masks = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=radius).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            # Debugging: color where soft mask is 1
            # tmp_rgb = torch.ones((224,224,3))
            # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1,0,0]).float()
            # rgb_mask.append(tmp_rgb)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                background_masks.append(background_mask)
                image = torch.clamp(image + background_mask, 0., 1.)

            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        if background is not None:
            background_masks = torch.cat(background_masks, dim=0).permute(0, 3, 1, 2)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    # ax.imshow(background_masks[i].permute(1,2,0).cpu().numpy())
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        return images

    def generate_checkerboard_background(self, k):
        """
            shape: dimensions of output tensor
            k: edge size of square, self.dim must be divisible by k
        """
        def checkerboard_indices_helper(h, w):
            return torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)))
        w, h = self.dim
        assert h % k == 0 and w % k == 0, f"image dim={self.dim} must be divisible by k={k}"
        base = checkerboard_indices_helper(h//k, w//k).sum(dim=0) % 2
        x = base.repeat_interleave(k, 0).repeat_interleave(k, 1)
        return x
    
    def get_image_background(self, mode):
        bkg_dim = (1,) + (self.dim[1],self.dim[0]) + (3,)
        if mode == 'red':
            # Return an image with shape bkg_dim with all pixels red
            return torch.ones(bkg_dim).to(device) * torch.tensor([1, 0, 0]).to(device)
        elif mode == 'green':
            # Return an image with shape bkg_dim with all pixels green
            return torch.ones(bkg_dim).to(device) * torch.tensor([0, 1, 0]).to(device)
        elif mode == 'blue':
            # Return an image with shape bkg_dim with all pixels blue
            return torch.ones(bkg_dim).to(device) * torch.tensor([0, 0, 1]).to(device)
        elif mode == 'white':
            # Return an image with shape bkg_dim with all pixels white
            return torch.ones(bkg_dim).to(device) * torch.tensor([1, 1, 1]).to(device)
        elif mode == 'black':
            # Return an image with shape bkg_dim with all pixels black
            return torch.zeros(bkg_dim).to(device)
        elif mode == 'random_noise':
            # Return an image with shape bkg_dim with random pixels
            return torch.rand(bkg_dim).to(device)
        elif mode == 'checkerboard':
            # Return an image with shape bkg_dim with a checkerboard pattern
            return torch.ones(bkg_dim).to(device) * self.generate_checkerboard_background(8).to(device).unsqueeze(0).unsqueeze(-1)
        elif mode == 'fixed_image':
            # Return an image with shape bkg_dim with a fixed image
            from PIL import Image
            # image_path = './data/background_images/city2.jpeg'
            # image_path = './data/background_images/road.jpeg'
            # image_path = './data/background_images/gotham2.jpeg'
            image_path = './data/background_images/green1.jpeg'
            image = Image.open(image_path)
            # image = image.crop((0, 0,
            #                     bkg_dim[2], bkg_dim[1]))
            
            # crop image in the center to bkg_dim
            w, h = image.size
            image = image.crop((w//2 - bkg_dim[2]//2, h//2 - bkg_dim[1]//2,
                                w//2 + bkg_dim[2]//2, h//2 + bkg_dim[1]//2))
            # convert to float tensor
            image = torch.from_numpy(np.array(image)).float().to(device).unsqueeze(0)
            image = image / 255.
            return image
        else:
            raise ValueError('Unknown background mode: {}'.format(mode))
    
    def get_image_background_cached(self, mode):
        # cache background image tensors for faster rendering
        if mode == 'random_noise':
            # If random_noise background is requested, we have to create a new random background each time
            return self.get_image_background('random_noise')
        # If already cached, return the cached background
        if mode in self.background_tensor_cache:
            return self.background_tensor_cache[mode]
        # Otherwise, cache the background and return it
        self.background_tensor_cache[mode] = self.get_image_background(mode)
        return self.background_tensor_cache[mode]

    def render_front_views(self, mesh, num_views=8, std=8, center_elev=0, center_azim=0, camera_r=2, show=False, lighting=True,
                           background=None, mask=False, return_views=False, frontview_elev_std=1.0, background_image_mode=None, save_camera_poses_path=None):
        # background_image_mode is used to be backward compatible with the old version of this function, we can remove background later.
        # Front view with small perturbations in viewing angle
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        elev = torch.cat((torch.tensor([center_elev]), torch.randn(num_views - 1) * np.pi / (std*frontview_elev_std) + center_elev))
        azim = torch.cat((torch.tensor([center_azim]), torch.randn(num_views - 1) * 2 * np.pi / std + center_azim))
        images = []
        masks = []
        # rgb_mask = []

        if (background is not None) or (background_image_mode is not None):
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=camera_r).to(device)
            if save_camera_poses_path:
                np.save(save_camera_poses_path + 'num_view' + str(i) + '.txt', camera_transform.cpu().numpy())
                # print(f'{camera_transform = }')
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            # Debugging: color where soft mask is 1
            # tmp_rgb = torch.ones((self.dim[0], self.dim[1], 3))
            # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1, 0, 0]).float()
            # rgb_mask.append(tmp_rgb)

            if (background is not None) or (background_image_mode is not None):
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if (background is not None) and (background_image_mode is None):
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)
            if background_image_mode is not None:
                background_image = self.get_image_background_cached(background_image_mode)
                mask = mask.squeeze(-1)
                image[torch.where(mask == 0)] = background_image[torch.where(mask == 0)]
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        # masks = torch.cat(masks, dim=0)
        # rgb_mask = torch.cat(rgb_mask, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views == True:
            return images, elev, azim
        else:
            return images
        
    def render_front_views_uniform(self, mesh, num_views=8, azim_dev_degree=30, center_elev=0, center_azim=0, camera_r=2, show=False, lighting=True,
                           background=None, mask=False, return_views=False, frontview_elev_std=1.0, background_image_mode=None):
        # background_image_mode is used to be backward compatible with the old version of this function, we can remove background later.
        # Front view with small perturbations in viewing angle
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]
        
        azim = torch.linspace(center_azim -  azim_dev_degree * np.pi/360, azim_dev_degree * np.pi/360 + center_azim, num_views + 1)[
               :-1]

        # repeat the tensor torch.tensor([center_elev] multiple times to match the number of views
        elev = torch.tensor([center_elev])
        elev = elev.repeat(num_views, 1).squeeze()
        assert elev.shape == azim.shape
        
        # elev = torch.cat((torch.tensor([center_elev]), torch.randn(num_views - 1) * np.pi / (std*frontview_elev_std) + center_elev))
        # azim = torch.cat((torch.tensor([center_azim]), torch.randn(num_views - 1) * 2 * np.pi / std + center_azim))
        images = []
        masks = []
        rgb_mask = []

        if (background is not None) or (background_image_mode is not None):
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=camera_r).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            # Debugging: color where soft mask is 1
            tmp_rgb = torch.ones((self.dim[0], self.dim[1], 3))
            tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1, 0, 0]).float()
            rgb_mask.append(tmp_rgb)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if (background is not None) and (background_image_mode is None):
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)
            if background_image_mode is not None:
                background_image = self.get_image_background_cached(background_image_mode)
                mask = mask.squeeze(-1)
                image[torch.where(mask == 0)] = background_image[torch.where(mask == 0)]
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        # masks = torch.cat(masks, dim=0)
        # rgb_mask = torch.cat(rgb_mask, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views == True:
            return images, elev, azim
        else:
            return images
        
    def render_prompt_views(self, mesh, prompt_views, center=[0, 0], background=None, show=False, lighting=True,
                            mask=False):

        # if mesh is None:
        #     mesh = self._current_mesh

        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]
        num_views = len(prompt_views)

        images = []
        masks = []
        rgb_mask = []
        face_attributes = mesh.face_attributes

        for i in range(num_views):
            view = prompt_views[i]
            if view == "front":
                elev = 0 + center[1]
                azim = 0 + center[0]
            if view == "right":
                elev = 0 + center[1]
                azim = np.pi / 2 + center[0]
            if view == "back":
                elev = 0 + center[1]
                azim = np.pi + center[0]
            if view == "left":
                elev = 0 + center[1]
                azim = 3 * np.pi / 2 + center[0]
            if view == "top":
                elev = np.pi / 2 + center[1]
                azim = 0 + center[0]
            if view == "bottom":
                elev = -np.pi / 2 + center[1]
                azim = 0 + center[0]

            if background is not None:
                face_attributes = [
                    mesh.face_attributes,
                    torch.ones((1, n_faces, 3, 1), device=device)
                ]
            else:
                face_attributes = mesh.face_attributes

            camera_transform = get_camera_from_view2(torch.tensor(elev), torch.tensor(azim), r=2).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        if not mask:
            return images
        else:
            return images, masks


if __name__ == '__main__':
    mesh = Mesh('sample.obj')
    mesh.set_image_texture('sample_texture.png')
    renderer = Renderer()
    # renderer.render_uniform_views(mesh,show=True,texture=True)
    mesh = mesh.divide()
    renderer.render_uniform_views(mesh, show=True, texture=True)
