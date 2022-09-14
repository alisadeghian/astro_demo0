

# obj_based_prompts1 = ["a red and blue hat", "a sports hat", "a fancy hat", "a luxury hat", "a golden hat", "super mario's hat", "a dog's hat", "a military hat", "a pirate's hat", "a cowboy hat"]
# obj_based_prompts2 = ["a red and blue cap", "a sports cap", "a fancy cap", "a luxury cap", "a golden cap", "super mario's cap", "a dog's cap", "a military cap"]
# obj_based_prompts3 = ["a women luxury military hat", "Nike cap", "Adidas cap", "Louis Vuitton cap"]


# obj_based_prompts1 = ["a marbel sink", "a luxury sink", "a fancy sink", "a wooden sink", "a stained glass sink", "a crochet sink", "a golden sink", "super mario's sink", "a classic sink", "a futuristic sink"]
# obj_based_prompts2 = ["a red and blue sink", "a kitchen sink", "a bathroom sink", "a women's sink", "a man's sink", "a realistic sink"]
# obj_based_prompts3 = ["Adidas sink", "Louis Vuitton sink", "Nike sink"]
obj_str = 'tea pot'
obj_based_prompts1 = [f"a marbel", f"a luxury", f"a fancy", f"a wooden", f"a stained glass", 
                      f"a crochet", f"a golden", f"a classic", f"a futuristic", f"a red and blue", f"a green and yellow"]
obj_based_prompts2 = [f"Snoop Dogg's", f"Super Mario's", f"Batman's", f"a women's", f"a man's", f"a Roman", f"an Aztec", f"a Persian", f"an American", f"a Chinese" f"a military", f"a sport"]
obj_based_prompts3 = [f"Adidas", f"Louis Vuitton", f"Nike", f"Gucci", f"Lacoste"]

prefix = "A 3D rendering of "
suffix = f" {obj_str} in Unreal Engine."

obj_based_prompts = obj_based_prompts1 + obj_based_prompts2 + obj_based_prompts3
num_prompts = 0
print("[", end="")
prompts_per_line = 3
for idx, obj_prompt in enumerate(obj_based_prompts):
    full_obj_prompt = prefix + obj_prompt.strip().strip('.') + suffix
    print(f'"{full_obj_prompt}"', end=', ')
    num_prompts += 1
    if idx % prompts_per_line == prompts_per_line - 1:
        print("")

for idx, obj_prompt in enumerate(obj_based_prompts):
    # capitalize the first letter
    full_obj_prompt = obj_prompt.strip().strip('.').capitalize() + f' {obj_str}.'
    print(f'"{full_obj_prompt}"', end=', ')
    num_prompts += 1
    if idx % (prompts_per_line+1) == prompts_per_line - 1:
        print("")
print("]")
print(f'{num_prompts} prompts created.')


# #############################################################################

wandb_agent = "wandb agent reza_armand/teapot_factory/0nf83ik1"
suffix = "CUDA_VISIBLE_DEVICES="
N_GPUS = 4
for i in range(N_GPUS):
    print('')
    print(f'{suffix}{i} {wandb_agent}')
    print('')


# #############################################################################

