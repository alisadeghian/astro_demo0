import os
import yaml
from itertools import product
#read a yml file and return the output as a dictionary
def read_yml(file_name):
    with open(file_name, 'r') as stream:
        # load the yml file as a dictionary
        config = yaml.safe_load(stream)
    return config




def get_gpus():
    os.system("nvidia-smi -L")
    gpus = os.popen("nvidia-smi -L").read().splitlines()
    return len(gpus)

def __main__():
    config = read_yml("config_hyperpar.yml") 
    n_gpu = get_gpus()
    # check the config dict and put the keys where their value is list in another list
    list_of_keys = []
    for key in config.keys():
        if isinstance(config[key], list):
            list_of_keys.append(key)

    # create a cartesian product of the list of keys and the values inside the list
    # and return the list of tuples
    cartesian_product = list(product(*(config[key] for key in list_of_keys)))
    print(cartesian_product)
    # create a list of dictionaries
    # each dictionary is a combination of the cartesian product of the list of keys and the values inside the list
    list_of_dict = [dict(zip(list_of_keys, tup)) for tup in cartesian_product]

    # for each dictionary in the list_of_dict run the main.py script with the current combination of parameters
    commands = []
    for id_dict, dictionary in enumerate(list_of_dict):
        print(dictionary)
        # create the command to run the main.py script
        # add to the begining of the command the gpu number enumrated om
        command = "python main.py"
        # for each key in the dictionary append the key and the value to the command
        for key in config.keys():

            value = dictionary[key] if key in dictionary.keys() else config[key]
            if key == "output_dir":
                value = value + "/" + str(id_dict)
            # check the value if it is boolean and True then only add the key
            if isinstance(value, bool):
                if value == True:
                    command += " --" + str(key)
            else:
                command += " --" + str(key) + " " + str(value)
        
    #  add the command to the list of commands
        commands.append(command)
    # run each command on differnt gpu with based on cuda_visible_devices 
    # and if the number of the commands are more than the number of gpu then wait and run the rest accoridnly
    gpu_number = 0
    with open("run_command.sh", "w") as text_file:
        for command in commands:
            # if the gpu number is less than the number of gpu then run the command on that gpu
            if gpu_number < n_gpu-1:
                gpu_number += 1
                command = "CUDA_VISIBLE_DEVICES=" + str(gpu_number) + " " + command + " &"
                # write the command to a bash file
                text_file.write(command)
                text_file.write("\n")
                
                
            # if the number of the commands is greater than the number of gpu then wait until the gpu is free and run the command
            else:
                text_file.write("echo 'waiting for gpus to be free' \n")
                text_file.write("wait \n")
                gpu_number = 0
        text_file.write("wait \n")
        text_file.write("echo 'all taskes are assigned'")
        # zip everything inside config['output_dir'] 
        text_file.write("echo 'zipping'")
        text_file.write("\n")
        text_file.write("zip -r " + config['output_dir'] + "_zip " + config['output_dir'])
        text_file.write("\n")


        










