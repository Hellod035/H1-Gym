# Legged_gym environment for unitree H1

Legged gym implementation of unitree H1 environment in IssacLab

## Install

```bash
conda create -n humanoid python=3.8
conda activate humanoid
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd ~/H1-Gym/
pip install -e legged_gym
pip install -e rsl_rl
```

## Run

```bash
cd ~/H1-Gym/legged_gym
python legged_gym/scripts/train.py --headless
python legged_gym/scripts/play.py
```

## Reference

- https://github.com/leggedrobotics/legged_gym
- https://github.com/leggedrobotics/rsl_rl
- https://github.com/isaac-sim/IsaacLab

