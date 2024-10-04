from .h1_flat import H1Flat
from .h1_rough_config import H1RoughCfg
import torch

class H1Rough(H1Flat):
    def __init__(self, cfg: H1RoughCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def compute_observations(self):
        """ Computes observations
        """
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        self.obs_buf = torch.cat((
                                self.base_ang_vel  * self.obs_scales.ang_vel, #[0, 3]
                                self.projected_gravity, #[3, 6]
                                self.commands[:, :3] * self.commands_scale, #[6, 9]
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #[9, 28]
                                self.dof_vel * self.obs_scales.dof_vel, #[28, 47]
                                self.actions, #[47, 66]
                                heights #[66, 253]
                                ),dim=-1)
        self.privileged_obs_buf = torch.cat((  
                                self.base_lin_vel * self.obs_scales.lin_vel, #[0, 3]
                                self.base_ang_vel  * self.obs_scales.ang_vel, #[3, 6]
                                self.projected_gravity, #[6, 9]
                                self.contact_filt, #[9, 11]
                                self.commands[:, :3] * self.commands_scale, #[11, 14]
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #[14, 33]
                                self.dof_vel * self.obs_scales.dof_vel, #[33, 52]
                                self.actions, #[52, 71]
                                heights #[71, 258]
                                ),dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

