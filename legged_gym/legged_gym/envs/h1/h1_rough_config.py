from .h1_flat_config import H1FlatCfg, H1FlatCfgPPO

class H1RoughCfg(H1FlatCfg):
    class env(H1FlatCfg.env):
        num_observations = 253

    class terrain(H1FlatCfg.terrain):
        curriculum = True
        selected = False # select a unique terrain type and pass all arguments

    class rewards(H1FlatCfg.rewards):
        class scales(H1FlatCfg.rewards.scales):
            lin_vel_z = 0.0
            feet_air_time_positive_biped = 0.25
        
        feet_air_time_threshold = 0.4


class H1RoughCfgPPO(H1FlatCfgPPO):
    class runner(H1FlatCfgPPO.runner):
        experiment_name = 'h1_rough'