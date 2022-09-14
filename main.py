import argparse
import copy
import os
import random
from pathlib import Path
from uuid import uuid4

import clip
import kaolin as kal
import kaolin.ops.mesh
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import wandb
from mesh import Mesh
from neural_style_field import NeuralStyleField
from bacon_modules import BACON, MultiscaleBACON
from Normalization import MeshNormalizer
from render import Renderer
from utils import add_vertices, device, sample_bary, prompt_encoder, diversify_batch_trans, add_color_to_prompt, parse_clip_model_names, count_parameters, add_background_to_prompt, make_gif


def run_branched(args):
    wandb.init(project=args.wandb_proj_name, mode="online") # mode = "online", "offline" or "disabled"
    # wandb.run.log_code(".") # to save the all the code in the current directory
    wandb.config.update(args)

    # Access all hyperparameter values through wandb.config to enable hyperparameter tuning
    args = wandb.config

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Constrain all sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
    # Check that isn't already done
    if (not args.overwrite) and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        print(f"Already done with {args.output_dir}")
        exit()
    elif args.overwrite and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        import shutil
        for filename in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    model_name_list, coeff_models, model_device_list = parse_clip_model_names(args.model_name, args.coeff_models_deviate)

    print(f"\nmodel_name_list: {model_name_list}")
    print(f"coeff_models: {coeff_models}")
    print(f"model_device_list: {model_device_list}\n")

    mesh = Mesh(args.obj_path)
    MeshNormalizer(mesh, normalizer=args.mesh_normalizer_func)()

    prior_color = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)

    background = None
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background).to(device)

    losses = []

    n_augs = args.n_augs
    dir = args.output_dir
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    normweight = 1.0

    # MLP Settings
    input_dim = 6 if args.input_normals else 3
    if args.only_z:
        input_dim = 1
    print(f'Using {input_dim} input dimensions')
    if args.neural_style_field == 'NeuralStyleField':
        mlp = NeuralStyleField(args.sigma, args.depth, args.width, 'gaussian', args.colordepth, args.normdepth,
                                    args.normratio, args.clamp, args.normclamp, niter=args.n_iter,
                                    progressive_encoding=args.pe, input_dim=input_dim, exclude=args.exclude).to(device)
        mlp.shrink_weights(shrink_factor=0.0)
    elif args.neural_style_field == 'BACON':
        print('Using BACON')
        mlp = BACON(in_size=input_dim, hidden_size=args.width, out_size=4, # 3 for RGB and 1 for displacement
                    hidden_layers=args.depth,
                    bias = True,
                    frequency=(args.sigma,)*input_dim,
                    quantization_interval=2*np.pi, # data on range [-0.5, 0.5]
                    reuse_filters=False,
                    normratio=args.normratio, clamp=args.clamp, normclamp=args.normclamp,
                    colordepth=args.colordepth, normdepth=args.normdepth)
        mlp.init_weights()
        mlp.to(device)
    elif args.neural_style_field == 'MultiscaleBACON':
        print('Using MultiscaleBACON')
        mlp = MultiscaleBACON(in_size=input_dim, hidden_size=args.width,
                    hidden_layers=args.depth,
                    bias = True, frequency=(args.sigma,)*input_dim,
                    quantization_interval=2*np.pi, # data on range [-0.5, 0.5]
                    reuse_filters=False,
                    normratio=args.normratio, clamp=args.clamp, normclamp=args.normclamp,
                    colordepth=args.colordepth, normdepth=args.normdepth)
        mlp.init_weights()
        mlp.to(device)
    else:
        raise ValueError(f'Unknown neural style field {args.neural_style_field}')

    if args.neural_style_field == 'NeuralStyleField':
        optim = torch.optim.Adam(mlp.parameters(), args.learning_rate, weight_decay=args.decay)
    if args.neural_style_field == 'BACON':
        optim = torch.optim.Adam(mlp.parameters(), args.learning_rate, weight_decay=args.decay, amsgrad=True)
    if args.neural_style_field == 'MultiscaleBACON':
        optim = torch.optim.Adam([{f'params': mlp.get_params_for_layer(i)} for i in range(args.depth)],
                                 args.learning_rate, weight_decay=args.decay, amsgrad=True)
    activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    if activate_scheduler:
        # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.01*args.learning_rate, max_lr=args.learning_rate, step_size_up=args.decay_step, mode="triangular", cycle_momentum=False)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.decay_step, gamma=args.lr_decay)
    
    clip_model_list, preprocess_list, render_list, INPUT_IMAGE_SHAPE_LIST, norm_encoded_list, ensemble_prompt_encoded_list,\
        encoded_text_list, encoded_image_list, ensemble_norm_encoded_list = [[] for _ in range(9)]
    colored_encoded_text_dict = {}
    transform_dict = {}
    for id, model_name in enumerate(model_name_list):
        device_id = model_device_list[id]
        clip_model, preprocess = clip.load(model_name, device_id, jit=False)
        INPUT_IMAGE_SHAPE = clip_model.visual.input_resolution
        print(f'Loaded clip model {model_name}')
        print(f'Number of parameters: {count_parameters(clip_model)}, visual model: {count_parameters(clip_model.visual)}')
        print(f'Model input shape is {INPUT_IMAGE_SHAPE}')
        
        if args.renderer_shape: # 
            render = Renderer(dim=(args.renderer_shape, args.renderer_shape))
        else:
            render = Renderer(dim=(INPUT_IMAGE_SHAPE, INPUT_IMAGE_SHAPE))
    
        # CLIP Transform
        clip_transform = transforms.Compose([
            transforms.Resize((INPUT_IMAGE_SHAPE, INPUT_IMAGE_SHAPE)),
            clip_normalizer
        ])
        transform_dict[('clip_transform', INPUT_IMAGE_SHAPE)] = clip_transform

        # background_choices = ['red', 'green', 'blue', 'white', 'black', 'random_noise', 'checkerboard']
        background_choices = ['green', 'white', 'black', 'random_noise', 'checkerboard']
        fill_values = [0.0, 1.0] if args.random_background else [1.0]
        for fill_value in fill_values:
            # Augmentation settings
            augment_transform = transforms.Compose([
                transforms.RandomResizedCrop(INPUT_IMAGE_SHAPE, scale=(1, 1)), # scale=(0.1, 1)
                transforms.RandomPerspective(fill=fill_value, p=0.8, distortion_scale=0.5),
                clip_normalizer
            ])
            # The above applies the same transform to all the images in a batch. Apply a different transform to each sample
            if args.divers_batch_aug:
                augment_transform = diversify_batch_trans(augment_transform)
            transform_dict[('augment_transform', INPUT_IMAGE_SHAPE, fill_value)] = augment_transform

            # Augmentations for normal network
            if args.cropforward :
                curcrop = args.normmincrop
            else: 
                curcrop = args.normmaxcrop
            normaugment_transform = transforms.Compose([
                transforms.RandomResizedCrop(INPUT_IMAGE_SHAPE, scale=(curcrop, curcrop)),
                transforms.RandomPerspective(fill=fill_value, p=0.8, distortion_scale=0.5),
                clip_normalizer
            ])
            if args.divers_batch_aug:
                normaugment_transform = diversify_batch_trans(normaugment_transform)
            transform_dict[('normaugment_transform', INPUT_IMAGE_SHAPE, fill_value)] = normaugment_transform

            cropiter = 0
            cropupdate = 0
            if args.normmincrop < args.normmaxcrop and args.cropsteps > 0:
                cropiter = round(args.n_iter / (args.cropsteps + 1))
                cropupdate = (args.maxcrop - args.mincrop) / cropiter

                if not args.cropforward:
                    cropupdate *= -1

            # Displacement-only augmentations
            displaugment_transform = transforms.Compose([
                transforms.RandomResizedCrop(INPUT_IMAGE_SHAPE, scale=(args.normmincrop, args.normmincrop)),
                transforms.RandomPerspective(fill=fill_value, p=0.8, distortion_scale=0.5),
                clip_normalizer
            ])
            if args.divers_batch_aug:
                displaugment_transform = diversify_batch_trans(displaugment_transform)
            transform_dict[('displaugment_transform', INPUT_IMAGE_SHAPE, fill_value)] = displaugment_transform

        if not args.no_prompt:
            if args.prompt and (not args.template_prompt):
                prompt = args.prompt
                print(f'\n Prompt: {prompt} \n')
                with torch.no_grad():
                    prompt_token = clip.tokenize([prompt]).to(device_id)
                    encoded_text = clip_model.encode_text(prompt_token)

                if args.prompt_based_background:
                    for fill_value in fill_values:
                        prompt_colored = add_color_to_prompt(prompt, fill_value)
                        print(f'\n Prompt: {prompt_colored} \n')
                        prompt_token = clip.tokenize([prompt_colored]).to(device_id)
                        colored_encoded_text_dict[(model_name, fill_value)] = clip_model.encode_text(prompt_token)

                if args.background_based_prompt:
                    for background_choice in background_choices:
                        prompt_colored = add_background_to_prompt(prompt, background_choice)
                        print(f'\n Prompt: {prompt_colored} \n')
                        prompt_token = clip.tokenize([prompt_colored]).to(device_id)
                        colored_encoded_text_dict[(model_name, background_choice)] = clip_model.encode_text(prompt_token)

                # Save prompt
                with open(os.path.join(dir, prompt), "w") as f:
                    f.write("")
                wandb.log({'prompt_string': wandb.Html(prompt)}, step=0)

                # Same with normprompt
                norm_encoded = encoded_text
        if args.normprompt is not None:
            prompt = ' '.join(args.normprompt)
            print(f'\n normprompt: {prompt} \n')
            with torch.no_grad():
                prompt_token = clip.tokenize([prompt]).to(device_id)
                norm_encoded = clip_model.encode_text(prompt_token)

            # Save prompt
            with open(os.path.join(dir, f"NORM {prompt}"), "w") as f:
                f.write("")
            wandb.log({'normprompt_string': wandb.Html(prompt)}, step=0)

        ensemble_prompt_encoded = []
        if args.ensemble_prompt is not None:
            for prompt_ in args.ensemble_prompt:
                print(f' prompt_ = {prompt_}')
                with torch.no_grad():
                    prompt_tokens = clip.tokenize([prompt_]).to(device_id)
                    prompt_encoded = clip_model.encode_text(prompt_tokens)
                    print(f' prompt_encoded.shape = {prompt_encoded.shape}')
                    ensemble_prompt_encoded.append(prompt_encoded)
            ensemble_prompt_encoded = torch.cat(ensemble_prompt_encoded)

            assert args.no_prompt, "Ensemble prompt is not supported without no_prompt."
            assert args.prompt is None, "Ensemble prompt is not supported with prompt."

            print(f'ensemble_prompt_encoded.shape = {ensemble_prompt_encoded.shape}')

            # Save prompt
            with open(os.path.join(dir, f"ENSEMBLE {'**'.join(args.ensemble_prompt)}"), "w") as f:
                f.write("")
            wandb.log({'prompt_string': wandb.Html(f"ENSEMBLE {'**'.join(args.ensemble_prompt)}")}, step=0)

            # ensemble_norm_encoded = ensemble_prompt_encoded
            # encoded_text = prompt_encoded # set it to the last prompt, for compatibility with eval_clip logging
            # with torch.no_grad():
            #     prompt_ = prompt_.replace('.', '').strip() + ' in grayscale.'
            #     prompt_tokens = clip.tokenize([prompt_]).to(device_id)
            #     prompt_encoded = clip_model.encode_text(prompt_tokens)
            # norm_encoded = prompt_encoded
            # print('Note: eval_clip_loss and geo_loss will be computed using the last prompt in the prompt-ensemble.')
            # print(f'\t norm_encoded prompt_ = {prompt_} \n')

        if args.image:
            img = Image.open(args.image)
            wandb.log({"Input image": wandb.Image(img)}, step=0)
            img = preprocess(img).to(device_id)
            wandb.log({"Preprocess input image": wandb.Image(img)}, step=0)
            with torch.no_grad():
                encoded_image = clip_model.encode_image(img.unsqueeze(0))
            if args.no_prompt:
                norm_encoded = encoded_image
            encoded_image_list.append(encoded_image)

        clip_model_list.append(clip_model), preprocess_list.append(preprocess), render_list.append(render),\
            INPUT_IMAGE_SHAPE_LIST.append(INPUT_IMAGE_SHAPE), norm_encoded_list.append(norm_encoded),\
            ensemble_prompt_encoded_list.append(ensemble_prompt_encoded), encoded_text_list.append(encoded_text),\
            # ensemble_norm_encoded_list.append(ensemble_norm_encoded)

    # Log gradients and model parameters
    wandb.watch((*clip_model_list, mlp), log='all', log_freq=10*args.log_interval, idx=1)

    loss_check = None
    vertices = copy.deepcopy(mesh.vertices)
    network_input = copy.deepcopy(vertices)
    if args.symmetry == True:
        network_input[:,2] = torch.abs(network_input[:,2])

    if args.standardize == True:
        # Each channel into z-score
        network_input = (network_input - torch.mean(network_input, dim=0))/torch.std(network_input, dim=0)

    if 'BACON' in args.neural_style_field:
        assert network_input.min() >= -0.5 and network_input.max() <= 0.5, \
            f'Network input out of range: {network_input.min()} {network_input.max()}' \
            'Use args.mesh_normalizer_func = min_max with BACON and disable standardization'

    if args.neural_style_field == 'MultiscaleBACON':
        global bacon_layer_idx
        bacon_layer_idx = -1
        increase_layer_idx = args.n_iter // (args.depth/2+1) # +1 to train the last layer longer
    
    for i in tqdm(range(args.n_iter)):
        
        if args.background_image_mode == 'change_per_iter':
            background_image_mode = random.choice(background_choices)
        else:
            background_image_mode = args.background_image_mode

        all_losses = 0.0
        optim.zero_grad()

        sampled_mesh = mesh

        if args.neural_style_field == 'MultiscaleBACON':
            if i % increase_layer_idx == 0 and (bacon_layer_idx + 2 < args.depth):
                bacon_layer_idx += 2
                print(f'\n Using MultiscaleBACON layer {bacon_layer_idx} \n')
                # reduce the learning rate for the layers lower than bacon_layer_idx-2 by a factor of 10
                if bacon_layer_idx >= 3:
                    for param_group in optim.param_groups[bacon_layer_idx-3:bacon_layer_idx-1]:
                        print('\n\t changing learning rate.\n')
                        param_group['lr'] = param_group['lr'] / 10.0

        update_mesh(mlp, network_input, prior_color, sampled_mesh, vertices, args)

        clip_loss_aggregated = 0.0
        for idx, model_name in enumerate(model_name_list):
            clip_model = clip_model_list[idx]
            preprocess = preprocess_list[idx]
            render = render_list[idx]
            INPUT_IMAGE_SHAPE = INPUT_IMAGE_SHAPE_LIST[idx]
            norm_encoded = norm_encoded_list[idx]
            ensemble_prompt_encoded = ensemble_prompt_encoded_list[idx]
            encoded_text = encoded_text_list[idx]
            clip_transform = transform_dict[('clip_transform', INPUT_IMAGE_SHAPE)]
            
            # encoded_image = encoded_image_list[idx]
            coeff_model = coeff_models[idx]
            if args.image:
                encoded_image = encoded_image_list[idx]
            else:
                encoded_image = None
            device_id = model_device_list[idx]

            # a reset to background is added here for the sake of consistency of evaluation
            background = torch.tensor(args.background).to(device)
            # compute a clip loss for logging/QA purposes
            if i % 50 == 0:
                with torch.no_grad():
                    rendered_images, elev, azim = render.render_front_views(sampled_mesh, num_views=args.n_views, # TODO: add as parameter
                                                                            show=args.show,
                                                                            center_azim=args.frontview_center[0],
                                                                            center_elev=args.frontview_center[1],
                                                                            camera_r=args.camera_r,
                                                                            std=20, #TODO: add as parameter
                                                                            return_views=True,
                                                                            background=background,
                                                                            frontview_elev_std=1,
                                                                            background_image_mode=None) # always use white background for evaluation

                    wandb.log({f"{model_name}_eval_rendered_images": wandb.Image(rendered_images)}, step=i)
                    clip_images = clip_transform(rendered_images)
                    encoded_renders = clip_model.encode_image(clip_images.to(device_id))
                    clip_loss_basic = -torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                    clip_loss2_view = -torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True), encoded_text)
                    encoded_renders_normless_mean = encoded_renders / encoded_renders.norm(dim=-1, keepdim=True)
                    clip_loss_normless_view = -torch.cosine_similarity(torch.mean(encoded_renders_normless_mean, dim=0, keepdim=True), encoded_text)
                    wandb.log({f"{model_name}_eval_clip_loss_basic": clip_loss_basic}, step=i)
                    wandb.log({f"{model_name}_eval_clip_loss2_view": clip_loss2_view}, step=i)
                    wandb.log({f"{model_name}_eval_clip_loss_normless_view": clip_loss_normless_view}, step=i)
                    clip_loss_aggregated += clip_loss_basic.detach().cpu().numpy()
                    wandb.log({"agg_eval_clip_loss": clip_loss_aggregated}, step=i)

            if args.random_background:
                back_value = float(random.randint(0,1))
            else:
                back_value = 1.0
            background = torch.tensor([back_value]).repeat(3).to(device)
            
            augment_transform = transform_dict[('augment_transform', INPUT_IMAGE_SHAPE, back_value)]
            normaugment_transform = transform_dict[('normaugment_transform', INPUT_IMAGE_SHAPE, back_value)]
            displaugment_transform = transform_dict[('displaugment_transform', INPUT_IMAGE_SHAPE, back_value)]

            if args.prompt_based_background:
                encoded_text = colored_encoded_text_dict[(model_name, back_value)]
            if args.background_based_prompt:
                encoded_text = colored_encoded_text_dict[(model_name, background_image_mode)]
            
            rendered_images, elev, azim = render.render_front_views(sampled_mesh, num_views=args.n_views,
                                                                    show=args.show,
                                                                    center_azim=args.frontview_center[0],
                                                                    center_elev=args.frontview_center[1],
                                                                    camera_r=args.camera_r,
                                                                    std=args.frontview_std,
                                                                    return_views=True,
                                                                    background=background, 
                                                                    frontview_elev_std=args.frontview_elev_std,
                                                                    background_image_mode=background_image_mode)
            if i % args.log_interval == 0:
                # Note : When logging a `torch.Tensor` as a `wandb.Image`, images are normalized. 
                # If you do not want to normalize your images, please convert your tensors to a PIL Image.
                wandb.log({f"{model_name}_rendered_images1": wandb.Image(rendered_images)}, step=i)

            if n_augs == 0:
                clip_image = clip_transform(rendered_images)
                if i % args.log_interval == 0:
                    wandb.log({f"{model_name}_rendered_images1_clip_transform": wandb.Image(clip_image)}, step=i)
                encoded_renders = clip_model.encode_image(clip_image.to(device_id))
                if not args.no_prompt:
                    loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                    all_losses += loss.detach().cpu().numpy()
                    wandb.log({f"{model_name}_loss0": loss}, step=i)

            # Check augmentation steps
            if args.cropsteps != 0 and cropupdate != 0 and i != 0 and i % args.cropsteps == 0:
                curcrop += cropupdate
                # print(curcrop)
                normaugment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(INPUT_IMAGE_SHAPE, scale=(curcrop, curcrop)),
                    transforms.RandomPerspective(fill=back_value, p=0.8, distortion_scale=0.5),
                    clip_normalizer
                ])
                wandb.log({f"{model_name}_curcrop": curcrop}, step=i)
                # Apply a different transform to each sample
                if args.divers_batch_aug:
                    normaugment_transform = diversify_batch_trans(normaugment_transform)

            if n_augs > 0:
                loss = 0.0
                for _ in range(n_augs):
                    augmented_image = augment_transform(rendered_images)
                    
                    if i % args.log_interval == 0:
                        wandb.log({f"{model_name}_rendered_images1_augment_transform":
                                    wandb.Image(augmented_image)}, step=i)
                    encoded_renders = clip_model.encode_image(augmented_image.to(device_id))
                    if not args.no_prompt:
                        if args.prompt:
                            if args.clipavg == "view":
                                if encoded_text.shape[0] > 1:
                                    temp_loss = torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                    torch.mean(encoded_text, dim=0), dim=0)
                                    loss -= temp_loss
                                    wandb.log({f"{model_name}_loss1.1": -temp_loss}, step=i)
                                else:
                                    temp_loss = torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                    encoded_text)
                                    loss -= temp_loss
                                    wandb.log({f"{model_name}_loss1.2": -temp_loss}, step=i)
                            else:
                                temp_loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                                loss -= temp_loss
                                wandb.log({f"{model_name}_loss1.3": -temp_loss}, step=i)
                    if args.ensemble_prompt is not None:
                        temp_loss = -torch.logsumexp(-2*torch.cosine_similarity(
                                                                torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                ensemble_prompt_encoded),
                                                    dim=0)
                        loss -= temp_loss
                        wandb.log({f"{model_name}_loss2": -temp_loss}, step=i)
                    if args.image:
                        if encoded_image.shape[0] > 1:
                            temp_loss = torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_image, dim=0), dim=0)
                            loss -= temp_loss
                            wandb.log({f"{model_name}_loss3.1": -temp_loss}, step=i)
                        else:
                            temp_loss = torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_image)
                            loss -= temp_loss
                            wandb.log({f"{model_name}_loss3.1": -temp_loss}, step=i)
                        # if args.image:
                        #     loss -= torch.mean(torch.cosine_similarity(encoded_renders,encoded_image))
            if args.splitnormloss:
                if args.neural_style_field == 'NeuralStyleField':
                    for param in mlp.mlp_normal.parameters():
                        param.requires_grad = False
            wandb.log({f"{model_name}_loss_n_augs": loss}, step=i)
            all_losses += loss.detach().cpu().numpy()
            # loss.backward(retain_graph=True)
            weighted_loss = loss * coeff_model
            weighted_loss.backward(retain_graph=True)

            # optim.step()

            # with torch.no_grad():
            #     losses.append(loss.item())

            # Normal augment transform
            # loss = 0.0
            if args.n_normaugs > 0:
                normloss = 0.0
                for _ in range(args.n_normaugs):
                    augmented_image = normaugment_transform(rendered_images)
                    if i % args.log_interval == 0:
                        wandb.log({"rendered_images1_normaugment_transform": wandb.Image(augmented_image)}, step=i)
                    encoded_renders = clip_model.encode_image(augmented_image.to(device_id))
                    if not args.no_prompt:
                        if args.prompt:
                            if args.clipavg == "view":
                                if norm_encoded.shape[0] > 1:
                                    temp_normloss = normweight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                                    torch.mean(norm_encoded, dim=0),
                                                                                    dim=0)
                                    normloss -= temp_normloss
                                    wandb.log({f"{model_name}_normloss1.1": -temp_normloss}, step=i)
                                else:
                                    temp_normloss = normweight * torch.cosine_similarity(
                                        torch.mean(encoded_renders, dim=0, keepdim=True),
                                        norm_encoded)
                                    normloss -= temp_normloss
                                    wandb.log({f"{model_name}_normloss1.2": -temp_normloss}, step=i)
                            else:
                                temp_normloss = normweight * torch.mean(
                                    torch.cosine_similarity(encoded_renders, norm_encoded))
                                normloss -= temp_normloss
                                wandb.log({f"{model_name}_normloss1.3": -temp_normloss}, step=i)
                    # TODO: This needs to be added for ensemble prompt
                    # if args.ensemble_prompt is not None:
                    #     temp_normloss = -torch.logsumexp(-2*torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                    #                                             ensemble_norm_encoded), dim=0)
                    #     normloss -= temp_normloss
                    #     wandb.log({"normloss2": -temp_normloss}, step=i)
                    if args.image:
                        # NOTE: This loss will never backwards if args.no_prompt is True
                        if not args.no_prompt:
                            print('WARNING: Why are you calculating the normloss')
                        if encoded_image.shape[0] > 1:
                            temp_normloss = torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_image, dim=0), dim=0)
                            normloss -= temp_normloss
                            wandb.log({f"{model_name}_normloss2.1": -temp_normloss}, step=i)
                        else:
                            temp_normloss = torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_image)
                            normloss -= temp_normloss
                            wandb.log({f"{model_name}_normloss2.2": -temp_normloss}, step=i)
                        # if args.image:
                        #     loss -= torch.mean(torch.cosine_similarity(encoded_renders,encoded_image))
                if args.splitnormloss:
                    if args.neural_style_field == 'NeuralStyleField':
                        for param in mlp.mlp_normal.parameters():
                            param.requires_grad = True
                    else: 
                        raise Exception('splitnormloss is not implemented for BACON')
                if args.splitcolorloss:
                    if args.neural_style_field == 'NeuralStyleField':
                        for param in mlp.mlp_rgb.parameters():
                            param.requires_grad = False
                    else: 
                        raise Exception('splitcolorloss is not implemented for BACON')
                if not args.no_prompt:
                    wandb.log({f"{model_name}_normloss_n_normaugs": normloss}, step=i)
                    all_losses += normloss.detach().cpu().numpy()
                    # normloss.backward(retain_graph=True)
                    weighted_normloss = normloss * coeff_model
                    weighted_normloss.backward(retain_graph=True)

        # Also run separate loss on the uncolored displacements
        if args.geoloss:
            default_color = torch.zeros(len(mesh.vertices), 3).to(device)
            default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
            sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                                   sampled_mesh.faces)
            for idx, model_name in enumerate(model_name_list):
                clip_model = clip_model_list[idx]
                preprocess = preprocess_list[idx]
                render = render_list[idx]
                INPUT_IMAGE_SHAPE = INPUT_IMAGE_SHAPE_LIST[idx]
                # print(f'INPUT_IMAGE_SHAPE = {INPUT_IMAGE_SHAPE}')
                norm_encoded = norm_encoded_list[idx]
                ensemble_prompt_encoded = ensemble_prompt_encoded_list[idx]
                encoded_text = encoded_text_list[idx]
                clip_transform = transform_dict[('clip_transform', INPUT_IMAGE_SHAPE)]
                augment_transform = transform_dict[('augment_transform', INPUT_IMAGE_SHAPE, back_value)]
                normaugment_transform = transform_dict[('normaugment_transform', INPUT_IMAGE_SHAPE, back_value)]
                displaugment_transform = transform_dict[('displaugment_transform', INPUT_IMAGE_SHAPE, back_value)]

                if args.prompt_based_background:
                    encoded_text = colored_encoded_text_dict[(model_name, back_value)]     
                if args.background_based_prompt:
                    encoded_text = colored_encoded_text_dict[(model_name, background_image_mode)]

                # encoded_image = encoded_image_list[idx]
                device_id = model_device_list[idx]
                coeff_model = coeff_models[idx]
                if args.image:
                    encoded_image = encoded_image_list[idx]
                else:
                    encoded_image = None 
                geo_renders, elev, azim = render.render_front_views(sampled_mesh, num_views=args.n_views,
                                                                    show=args.show,
                                                                    center_azim=args.frontview_center[0],
                                                                    center_elev=args.frontview_center[1],
                                                                    camera_r=args.camera_r,
                                                                    std=args.frontview_std,
                                                                    return_views=True,
                                                                    background=background,
                                                                    frontview_elev_std=args.frontview_elev_std,
                                                                    background_image_mode=background_image_mode)
                if i % args.log_interval == 0:
                    wandb.log({f"{model_name}_geo_rendered_images": wandb.Image(geo_renders)}, step=i)
                                                                    
                if args.n_normaugs > 0:
                    geo_normloss = 0.0
                    ### avgview != aug
                    for _ in range(args.n_normaugs):
                        augmented_image = displaugment_transform(geo_renders)
                        if i % args.log_interval == 0:
                            wandb.log({f"{model_name}_geo_rendered_images_displaugment_transform": wandb.Image(augmented_image)}, step=i)
                        encoded_renders = clip_model.encode_image(augmented_image.to(device_id))
                        if norm_encoded.shape[0] > 1:
                            temp_loss = torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                torch.mean(norm_encoded, dim=0), dim=0)
                            geo_normloss -= temp_loss
                            wandb.log({f"{model_name}_geo_normloss1": -temp_loss}, step=i)
                        else:
                            temp_loss = torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                norm_encoded)
                            geo_normloss -= temp_loss
                            wandb.log({f"{model_name}_geo_normloss2": -temp_loss}, step=i)
                        if args.image:
                            # TODO: is this loss even used??
                            if encoded_image.shape[0] > 1:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                torch.mean(encoded_image, dim=0), dim=0)
                            else:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                encoded_image)  # if args.image:
                            #     loss -= torch.mean(torch.cosine_similarity(encoded_renders,encoded_image))
                    # if not args.no_prompt:
                    all_losses += geo_normloss.detach().cpu().numpy()
                    # geo_normloss.backward(retain_graph=True)
                    weighted_geo_normloss = geo_normloss * coeff_model
                    weighted_geo_normloss.backward(retain_graph=True)
                    wandb.log({f"{model_name}_geo_normloss_backward": geo_normloss}, step=i)

        wandb.log({f"{model_name}_all_losses": all_losses}, step=i)
        wandb.log({"lr_optim": optim.param_groups[0]['lr']}, step=i)
        optim.step()

        if args.neural_style_field == 'NeuralStyleField':
            for param in mlp.mlp_normal.parameters():
                param.requires_grad = True
            for param in mlp.mlp_rgb.parameters():
                param.requires_grad = True

        if activate_scheduler:
            lr_scheduler.step()

        with torch.no_grad():
            losses.append(loss.item())

        # Adjust normweight if set
        if args.decayfreq is not None:
            if i % args.decayfreq == 0:
                normweight *= args.cropdecay

        if i % 100 == 0:
            report_process(args, dir, i, loss, loss_check, losses, rendered_images)

    export_final_results(args, dir, losses, mesh, mlp, network_input, vertices)


def report_process(args, dir, i, loss, loss_check, losses, rendered_images):
    print('iter: {} loss: {}'.format(i, loss.item()))
    torchvision.utils.save_image(rendered_images, os.path.join(dir, 'iter_{}.jpg'.format(i)))
    if args.lr_plateau and loss_check is not None:
        new_loss_check = np.mean(losses[-100:])
        # If avg loss increased or plateaued then reduce LR
        if new_loss_check >= loss_check:
            for g in torch.optim.param_groups:
                g['lr'] *= 0.5
        loss_check = new_loss_check

    elif args.lr_plateau and loss_check is None and len(losses) >= 100:
        loss_check = np.mean(losses[-100:])


def export_final_results(args, dir, losses, mesh, mlp, network_input, vertices):
    with torch.no_grad():
        pred_rgb, pred_normal = mlp(network_input)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal

        objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
        # create a random directory name to save the final mesh to avoid racing issues when running sweeps
        rand_dir = os.path.join(dir, uuid4().hex)
        os.makedirs(rand_dir, exist_ok=True)
        final_obj_path = os.path.join(rand_dir, f"{objbase}_final.obj")
        mesh.export(final_obj_path, color=final_color)
        # NOTE: this might take long for large objects.
        wandb.log({'final_3d_object': wandb.Object3D(open(final_obj_path))})

        # Run renders
        if args.save_render:
            save_rendered_results(args, dir, final_color, mesh)

        # Save final losses
        torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))


def save_rendered_results(args, dir, final_color, mesh):
    default_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                   mesh.faces.to(device))
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, 1280 / 720).to(device),
        dim=(1280, 720))
    MeshNormalizer(mesh, normalizer=args.mesh_normalizer_func)()
    img, mask = kal_render.render_single_view(mesh, args.frontview_center[1], args.frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"init_cluster.png"))
    wandb.log({"init_cluster": wandb.Image(img)})
    MeshNormalizer(mesh, normalizer=args.mesh_normalizer_func)()
    # Vertex colorings
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(final_color.unsqueeze(0).to(device),
                                                                   mesh.faces.to(device))
    
    for azim_diff, elev_diff in zip([0, np.pi], [0, 0]):
        img, mask = kal_render.render_single_view(mesh, args.frontview_center[1], args.frontview_center[0]+azim_diff,
                                                radius=2.5,
                                                background=torch.tensor([1, 1, 1]).to(device).float(),
                                                return_mask=True)
        img = img[0].cpu()
        mask = mask[0].cpu()
        # Manually add alpha channel using background color
        alpha = torch.ones(img.shape[1], img.shape[2])
        alpha[torch.where(mask == 0)] = 0
        img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(dir, f"final_cluster.png"))
        wandb.log({f"final_cluster azim_diff={azim_diff} elev_diff={elev_diff}": wandb.Image(img)})

    if args.save_gif:
        with torch.no_grad():
            kal_render = Renderer(dim=(512, 512))
            rendered_images, elev, azim = kal_render.render_front_views_uniform(mesh, num_views=60, # TODO: add as parameter
                                                                        show=args.show,
                                                                        center_azim=args.frontview_center[0],
                                                                        center_elev=args.frontview_center[1],
                                                                        camera_r=args.camera_r*1.05,
                                                                        azim_dev_degree=360, #TODO: add as parameter
                                                                        return_views=True,
                                                                        background=torch.tensor([1, 1, 1]).to(device).float(),
                                                                        background_image_mode=None)
         
            fname = make_gif(rendered_images, os.path.join(dir, 'final.gif'), duration=3)
            print(f'\n\t Saved gif: {fname}\n')
            wandb.log({"final_gif": wandb.Video(fname)})
            


def update_mesh(mlp, network_input, prior_color, sampled_mesh, vertices, args):
    if args.neural_style_field == 'MultiscaleBACON': 
        pred_rgb, pred_normal = mlp(network_input, specified_layer=bacon_layer_idx)
    else:
        pred_rgb, pred_normal = mlp(network_input)
    sampled_mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    sampled_mesh.vertices = vertices + sampled_mesh.vertex_normals * pred_normal
    MeshNormalizer(sampled_mesh, normalizer=args.mesh_normalizer_func)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='meshes/mesh1.obj')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--normprompt', nargs="+", default=None)
    parser.add_argument('--promptlist', nargs="+", default=None) # Ali: not used
    parser.add_argument('--normpromptlist', nargs="+", default=None) # Ali: not used
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='round2/alpha5')
    parser.add_argument('--traintype', type=str, default="shared")
    parser.add_argument('--sigma', type=float, default=10.0)
    parser.add_argument('--normsigma', type=float, default=10.0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--colordepth', type=int, default=2)
    parser.add_argument('--normdepth', type=int, default=2)
    parser.add_argument('--normwidth', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--normal_learning_rate', type=float, default=0.0005)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--lr_plateau', type=int, default=0)
    parser.add_argument('--pe', type=int, default=1) # progressive_encoding in NSF
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--n_views', type=int, default=5)
    parser.add_argument('--n_augs', type=int, default=0)
    parser.add_argument('--n_normaugs', type=int, default=0)
    parser.add_argument('--n_iter', type=int, default=6000)
    parser.add_argument('--encoding', type=str, default='gaussian')
    parser.add_argument('--normencoding', type=str, default='xyz')
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--run', type=str, default=None) # Ali: not used
    parser.add_argument('--gen', type=int, default=0) # Ali: not used
    parser.add_argument('--clamp', type=str, default="tanh")
    parser.add_argument('--normclamp', type=str, default="tanh")
    parser.add_argument('--normratio', type=float, default=0.1)
    # parser.add_argument('--frontview', type=int, default=0)
    parser.add_argument('--no_prompt', type=int, default=0)
    parser.add_argument('--exclude', type=int, default=0)

    parser.add_argument('--frontview_std', type=float, default=8)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0.0, 0.0]) # horse [3., 0.15], cap/sink [1.57, 0.1]
    parser.add_argument('--clipavg', type=str, default=None)
    parser.add_argument('--geoloss', type=int, default=0)
    parser.add_argument('--samplebary', type=int, default=0) # Ali: not used, but can use it for upsampling 
    parser.add_argument('--promptviews', nargs="+", default=None)
    parser.add_argument('--mincrop', type=float, default=1)
    parser.add_argument('--maxcrop', type=float, default=1)
    parser.add_argument('--normmincrop', type=float, default=0.1)
    parser.add_argument('--normmaxcrop', type=float, default=0.1)
    parser.add_argument('--splitnormloss', type=int, default=0)
    parser.add_argument('--splitcolorloss', type=int, default=0)
    parser.add_argument('--nonorm', type=int, default=0) # Ali: currently not used 
    parser.add_argument('--cropsteps', type=int, default=0)
    parser.add_argument('--cropforward', type=int, default=0)
    parser.add_argument('--cropdecay', type=float, default=1.0)
    parser.add_argument('--decayfreq', type=int, default=None)
    parser.add_argument('--overwrite', type=int, default=0)
    parser.add_argument('--show', type=int, default=0)
    parser.add_argument('--background', nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_render', type=int, default=1)
    parser.add_argument('--input_normals', type=int, default=0)
    parser.add_argument('--symmetry', type=int, default=0)
    parser.add_argument('--only_z', type=int, default=0)
    parser.add_argument('--standardize', type=int, default=0)
    parser.add_argument('--ensemble_prompt', nargs="*", default=None, type=str)
    parser.add_argument('--coeff_models_deviate', default=0.5, type=float) # 0 is only 1st model, 0.5 is all equally weighted, 1 is all last model.
    parser.add_argument('--model_name', type=str, default='ViT-B/32') # ViT-B/32:0,RN50x4:1
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--divers_batch_aug', type=int, default=0)
    parser.add_argument('--random_background', type=int, default=0)
    parser.add_argument('--prompt_based_background', type=int, default=0)
    parser.add_argument('--template_prompt', type=str, default=None)
    parser.add_argument('--mesh_normalizer_func', type=str, default='bounding_sphere')
    parser.add_argument('--neural_style_field', type=str, default='NeuralStyleField') # BACON, MultiscaleBACON, NeuralStyleField
    parser.add_argument('--camera_r', type=float, default=2)
    parser.add_argument('--frontview_elev_std', type=int, default=1)
    parser.add_argument('--background_image_mode', type=str, default=None, help='set to "change_per_iter" if you want to use it.') # ['green', 'white', 'black', 'random_noise', 'checkerboard'], 
    parser.add_argument('--background_based_prompt', type=int, default=0)
    parser.add_argument('--renderer_shape', type=int, default=0) # 0: uses each CLIPs image shape.
    parser.add_argument('--wandb_proj_name', type=str, default="astro_text2mesh")
    parser.add_argument('--save_gif', type=int, default=1)
    args = parser.parse_args()

    # Testing in practice shows a minor decrease in performance when True.
    torch.backends.cudnn.benchmark = False
    # Better reproducibility
    torch.backends.cudnn.deterministic = True

    run_branched(args)
