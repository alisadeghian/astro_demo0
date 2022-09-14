#%%

import wandb
import os
api = wandb.Api(timeout=20)


# %%

# project_name = "teapot_factory"

# object_name = 'teapot1'
# display_names_list = [298, 302, 320, 307, 308, 310, 250, 226, 262, 237, 259, 266]

# object_name = 'teapot2'
# display_names_list = [190, 150, 205, 131, 175, 199, 201, 153, 116, 152, 158, 111]

# object_name = 'teapot3'
# display_names_list = [80, 83, 67, 70, 69, 87]

# %%

# project_name = "chair_factory"

# object_name = 'chair1'
# display_names_list = [110, 92, 41, 25, 4, 74, 36, 3, 10, 11, 59, 70, 69, 62, 56]

# object_name = 'chair2'
# display_names_list = [154, 203, 204, 210, 179, 212, 180, 198, 172, 183, 174, 128, 135]

# object_name = 'chair3'
# display_names_list = [269, 315, 318, 301, 255, 235]


# %%

# project_name = "lamp_factory"

# object_name = 'lamp1'
# display_names_list = [322, 319, 311, 324, 310, 292, 270, 314, 250, 282, 278, 232, 275, 238]

# object_name = 'lamp2'
# display_names_list = [196, 216, 201, 151, 201, 135, 183, 171, 138, 170, 164, 177, 166, 119, 121]

# object_name = 'lamp3'
# display_names_list = [100, 104, 101, 93, 73, 8, 20, 65]

# %%

# project_name = "hat_factory44"

# object_name = 'hat1'
# display_names_list = [76, 67, 42, 40, 50]

# object_name = 'hat2'
# display_names_list = [112, 119, 121, 103, 108, 110, 100, 96, 102]

# object_name = 'hat3'
# display_names_list = [30, 24, 9, 8, 15]

# %%

# project_name = "candle_factory"

# object_name = 'candle1'
# display_names_list = [104, 102, 69, 81, 90, 92, 56, 46, 24, 17]

# %%

project_name = "sink_factory"

object_name = 'sink1'
display_names_list = []

# %%

wandb_user = "reza_armand"
file_extensions = ['.gif', '.obj']
local_root_dir = f"./astroblox_objects/{object_name}"

# if folder exists, ask for confirmation to delete it
if os.path.exists(local_root_dir):
    print(f"{local_root_dir} already exists. Delete it? (y/n)")
    if input() == 'y':
        os.system(f"rm -r {local_root_dir}")
        print(f"{local_root_dir} deleted.")
    else:
        print("Aborting...")
        exit()
# create dir    
os.makedirs(local_root_dir, exist_ok=True)


# %%
run_urls = []
for display_name in display_names_list:
    print('-'*50)
    print(f"Downloading {display_name}")
    # Get runs with display_name following the pattern
    runs = api.runs(
        path=f"{wandb_user}/{project_name}",
        filters={"display_name": {"$regex": f"-{display_name}$"}}
    )

    # assert len(runs) == 1

    # save the relevant files
    for run_num, run in enumerate(runs):
        run_urls.append(run.url)
        files = run.files()
        local_path = os.path.join(local_root_dir, run.name)
        for file in files:
            for extension in file_extensions:
                if file.name.endswith(extension):
                    print(file)
                    # downlod the file
                    file.download(local_path, replace=True)
    
# save run urls to a file
with open(f"{local_root_dir}/run_urls.txt", "w") as f:
    for run_url in run_urls:
        f.write(f"{run_url}\n")
print(f"{len(run_urls)} run urls saved to {local_root_dir}/run_urls.txt")


# %%

# list all the files in local_root_dir and its subdirs
files = []
for dirpath, dirnames, filenames in os.walk(local_root_dir):
    for filename in filenames:
        files.append(os.path.join(dirpath, filename))


files = sorted(files)
obj_files = [file for file in files if file.endswith(".obj")]
gif_files = [file for file in files if file.endswith(".gif")]

assert len(obj_files) == len(gif_files)


# copy all the .obj and .gif files to local_root_dir
# NOTE: this can make the order of run_urls.txt and the order of the files in the local_root_dir inconsistent
file_ctr = 1
for obj_file, gif_file in zip(obj_files, gif_files):
    # rename the files to the same name
    obj_file2 = f"{'/'.join(obj_file.split('/')[:-1])}/{file_ctr}.obj"
    os.rename(obj_file, obj_file2)
    gif_file2 = f"{'/'.join(gif_file.split('/')[:-1])}/{file_ctr}.gif"
    os.rename(gif_file, gif_file2)

    os.system(f"cp {obj_file2} {local_root_dir}")
    os.system(f"cp {gif_file2} {local_root_dir}")

    file_ctr += 1

# delete all the folders in local_root_dir
for dirpath, dirnames, filenames in os.walk(local_root_dir):
    for dirname in dirnames:
        os.system(f"rm -r '{os.path.join(dirpath, dirname)}'")

print('Done!')
# %%
