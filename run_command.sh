# CUDA_VISIBLE_DEVICES=8 python main.py --run branch --obj_path data/source_meshes/person.obj --output_dir results/demo/roman_resnet_multiple_prmp --normratio 0.02 --ensemble_prompt 'a 3D rendering of a Roman soldier' 'a legionaries' 'a person' 'an ancient Roman warrior' --sigma 12.0 --clamp tanh --n_normaugs 4 --n_augs 1 --normmincrop 0.1 --normmaxcrop 0.4 --geoloss --colordepth 2 --normdepth 2 --frontview --frontview_std 100 --clipavg view --lr_decay 0.9 --normclamp tanh --maxcrop 1.0 --save_render --seed 29 --n_iter 2000 --learning_rate 0.0005 --normal_learning_rate 0.0005 --no_pe --symmetry --background 1 1 1
# CUDA_VISIBLE_DEVICES=7 python main.py --run branch --obj_path data/source_meshes/person.obj --output_dir results/demo/roman_resnet_multiple_prmp_/2 --normratio 0.02 --ensemble_prompt 'a 3D rendering of a Roman soldier' 'a legionary standing with open arms' 'a 3D human' 'an ancient Roman warrior' 'a 3D Roman warrior in an unreal engine' --sigma 12.0 --clamp tanh --n_normaugs 4 --n_augs 1 --normmincrop 0.1 --normmaxcrop 0.4 --geoloss --colordepth 2 --normdepth 2 --frontview --frontview_std 4 --clipavg view --lr_decay 0.9 --normclamp tanh --maxcrop 1.0 --save_render --seed 29 --n_iter 2000 --learning_rate 0.0005 --normal_learning_rate 0.0005 --no_pe --symmetry --background 1 1 1
# CUDA_VISIBLE_DEVICES=6 python main.py --run branch --obj_path data/source_meshes/person.obj --output_dir results/demo/roman_resnet_multiple_prmp_/3 --normratio 0.02 --ensemble_prompt 'a 3D rendering of a Roman soldier' 'a legionary standing with open arms' 'a 3D human' 'an ancient Roman warrior' 'a 3D Roman warrior in an unreal engine' --sigma 12.0 --clamp tanh --n_normaugs 4 --n_augs 1 --normmincrop 0.1 --normmaxcrop 0.4 --geoloss --colordepth 2 --normdepth 2 --frontview --frontview_std 4 --clipavg view --lr_decay 0.9 --normclamp tanh --maxcrop 1.0 --save_render --seed 29 --n_iter 2000 --learning_rate 0.0005 --normal_learning_rate 0.0005 --no_pe --symmetry --background 1 1 1
CUDA_VISIBLE_DEVICES=2 wandb agent reza_armand/text2mesh/en4s9scj &
CUDA_VISIBLE_DEVICES=3 wandb agent reza_armand/text2mesh/en4s9scj &
CUDA_VISIBLE_DEVICES=4 wandb agent reza_armand/text2mesh/en4s9scj &
CUDA_VISIBLE_DEVICES=5 wandb agent reza_armand/text2mesh/en4s9scj &
CUDA_VISIBLE_DEVICES=6 wandb agent reza_armand/text2mesh/en4s9scj &
CUDA_VISIBLE_DEVICES=7 wandb agent reza_armand/text2mesh/en4s9scj &
CUDA_VISIBLE_DEVICES=8 wandb agent reza_armand/text2mesh/en4s9scj &
CUDA_VISIBLE_DEVICES=9 wandb agent reza_armand/text2mesh/en4s9scj &
wait
echo "first bacth done"
