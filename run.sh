cd ~/projects/H1-Gym/legged_gym
conda activate h1gym
tensorboard --logdir .

python legged_gym/scripts/train.py --task h1_flat --headless --resume --load_run Oct21_15-59-52_
python legged_gym/scripts/play.py --task h1_flat --load_run Oct05_01-37-34_ --checkpoint 800
