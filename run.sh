cd ~/humanoid/legged_gym
conda activate humanoid
tensorboard --logdir .

python legged_gym/scripts/train.py --task h1_flat --headless
python legged_gym/scripts/play.py --task h1_flat --load_run Oct05_01-37-34_ --checkpoint 800


python legged_gym/scripts/train.py --task h1_rough --headless
python legged_gym/scripts/play.py --task h1_rough --load_run xxxx --checkpoint xxxx