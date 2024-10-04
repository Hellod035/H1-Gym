cd ~/humanoid/legged_gym
conda activate humanoid
tensorboard --logdir .

python legged_gym/scripts/train.py --headless
python legged_gym/scripts/play.py --load_run 0001-locomotion-mlp --checkpoint 1000
