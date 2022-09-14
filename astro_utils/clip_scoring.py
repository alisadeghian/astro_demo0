import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# print(clip.available_models())

device_id = 'cpu'
clip_model_name = clip.available_models()[5]
# load clip model [0'RN50', 1'RN101', 2'RN50x4', 3'RN50x16', 4'RN50x64', 5'ViT-B/32', 6'ViT-B/16', 7'ViT-L/14', 8'ViT-L/14@336px']
clip_model, preprocess = clip.load(clip_model_name, device_id, jit=False)

INPUT_IMAGE_SHAPE = 224
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
clip_transform = transforms.Compose([
                    transforms.Resize((INPUT_IMAGE_SHAPE, INPUT_IMAGE_SHAPE)),
                    clip_normalizer,
                ])

with torch.no_grad():
    prompt = 'A golden retriever wearing a black cap on his head.\n'
    # image_path = '/Users/amir2/Downloads/dog1_hat1_1.jpg'
    image_path_list = ['/Users/amir2/Downloads/dog1_hat1_1.jpg', '/Users/amir2/Downloads/dog1_hat1_2.jpg']
    # image_path_list = ['/Users/amir2/Downloads/dog2_hat2_1.jpg', '/Users/amir2/Downloads/dog2_hat2_2.jpg']
    scores_list = []

    prompt_token = clip.tokenize([prompt]).to(device_id)
    encoded_text = clip_model.encode_text(prompt_token)
    # print(encoded_text.shape)

    for image_path in image_path_list:
        img = Image.open(image_path)
        print(img)
        img = preprocess(img).to(device_id)
        print(img.max(), img.min())
        encoded_image = clip_model.encode_image(img.unsqueeze(0))
        # print(encoded_image.shape)

        # compute the cosin similarty between the image and the text
        score = torch.cosine_similarity(encoded_image, encoded_text)
        print(f'Score: {score}, image_path: {image_path}')
        scores_list.append(score)

clip_model_print_str = f'Using clip model: {clip_model_name}'
print(clip_model_print_str)
# show both images side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(Image.open(image_path_list[0]))
ax[0].set_title(f'Image 1')
ax[1].imshow(Image.open(image_path_list[1]))
ax[1].set_title(f'Image 2')

print_str = f'{prompt}'
print_str += f'{scores_list}'
if scores_list[0] > scores_list[1]:
    score_diff = scores_list[0] - scores_list[1]
    score_diff = score_diff.item()
    print_str += f'\nImage 1 is better score={scores_list[0].item():.5f}, {score_diff=:.5f}.'
else:
    score_diff = scores_list[1] - scores_list[0]
    score_diff = score_diff.item()
    print_str += f'\nImage 2 is better, score={scores_list[1].item():.5f}, {score_diff=:.5f}.'
    # plt.imshow(Image.open(image_path_list[1]))

print_str += f'\n{clip_model_print_str}'
print(print_str)
# plt.imshow(Image.open(image_path_list[0]))
plt.suptitle(print_str)
plt.show()
