from os import path
from typing import Dict, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from evorob.utils.geometry import quat2rot
from evorob.utils.logging import log

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.5,
    "lookat": np.array([2.1, 0, 0]),
    "elevation": -25.0,
}


class PassiveWalker(MujocoEnv, utils.EzPickle):
    r"""In this environment a Passive Dynamic Walker is tasked to locomote.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        main_body: Union[int, str] = 1,
        reset_noise_scale: float = 0.0,
        exclude_current_positions_from_observation: bool = False,
        include_cfrc_ext_in_observation: bool = False,
        pert_force=None,
        init_z_offset=0.0,
        verbose: bool = False,
        **kwargs,
    ):
        self.verbose = verbose
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            robot_path,
        )

        utils.EzPickle.__init__(
            self,
            xml_file_path,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            main_body,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            pert_force,
            init_z_offset,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            width=832,
            height=496,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.width = 1200
        self.height = 600
        self.screen_size = (1200, 600)

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += (
            self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }
        z = self.init_qpos.copy()[2]
        self.init_qpos = np.array([ 0.000000000000000e+00,  0.000000000000000e+00,
                                    z,  1.000000000000000e+00,
                                    0.000000000000000e+00,  0.000000000000000e+00,
                                    0.000000000000000e+00, -1.439822065508426e-01,
                                    -3.217110226904367e-01, 7.327784974642921e-01,
                                    -7.567356586881939e-01,])
        self.init_qvel = np.array([2.131535515796873e-04, 9.424507547596443e-05,
                                   -2.551051134047193e-04,2.587135317335107e-04,
                                   0.000000000000000e+00, 0.000000000000000e+00,
                                   8.160239699864630e-02, 1.000000000000000e-01,
                                   7.669169846911073e-02, 0.000000000000000e+00])
        self.init_qpos[2] += init_z_offset

        self.init_z_offset = init_z_offset
        self.body_ids = None
        self.force = None
        self.previous_state = None
        self.prev_x_velocity = 0.0
        self.stuck = 0
        self.fall_penalty = 50.0
        if pert_force is not None:
            self.body_ids , self.force = pert_force

    def step(self, action):
        p_before = self.data.body(self._main_body).xpos.copy()
        if self.body_ids is not None:
            self.apply_force()
        self.do_simulation(action, self.frame_skip)
        p_after = self.data.body(self._main_body).xpos.copy()

        reward, reward_terms = self._get_reward(p_before, p_after)
        observation = self._get_obs()

        terminated = False
        # Check for NaN, Inf, or huge values
        qacc = self.data.qacc
        if np.any(np.isnan(qacc)) or np.any(np.isinf(qacc)) or np.any(np.abs(qacc) > 1e6):
            DOF = np.argwhere((np.isnan(qacc)) + (np.isinf(qacc)) + (np.abs(qacc) > 1e6)).squeeze()
            print(ValueError(f'MuJoCo Warning: Nan, Inf or huge value in QACC at DOF {DOF}'))
            terminated = True
        relative_z = self.init_qpos[2] - self.init_z_offset
        fall_threshold = self.init_z_offset + relative_z * 0.5 - self.data.qpos[0]*np.tan(5*np.pi/180)
        if self.data.qpos[2] < fall_threshold:
            log(f"Walker Fell off the platform at {self.data.qpos[0]} meter!!", self.verbose)
            terminated = True
        if np.abs(self.data.qpos[0] - self.previous_state[0])<1e-4:
            self.stuck += 1
            if self.stuck > 10/self.dt:
                print("Walker not moving for 10 seconds!!")
                terminated = True
        else:
            self.stuck = 0

        info = {
            "reward_progress": reward_terms["progress_reward"],
            "reward_total": reward,
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
        self.prev_x_velocity = reward_terms["x_velocity"]

        self.previous_state = observation

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_reward(self, p_before, p_after):
        """Compute reward and individual reward terms."""
        velocity = (p_after - p_before) / self.dt
        
        # Calculate progress along the slope direction (5 degrees)
        theta = 5 * np.pi / 180
        t_hat = np.array([np.cos(theta), 0.0, -np.sin(theta)])
        progress_velocity = np.dot(velocity, t_hat)

        # Small alive bonus, only awarded if making meaningful forward progress
        alive_bonus = 0.02 if progress_velocity > 0.05 else 0.0
        
        # Penalize moving backward or standing still
        stagnation_penalty = 0.5 if progress_velocity < 1e-3 else 0.0
        
        # Penalize lateral drift: deviation from centerline (y=0) and lateral velocity
        y_position = p_after[1]
        drift_penalty = 0.2 * abs(y_position) + 0.1 * abs(velocity[1])

        # NEW: Adding upright regularisation penalty based on pitch
        body_rot = self.data.body(self._main_body).xmat.reshape(3, 3)
        upright_cos = body_rot[2, 2]
        pitch_penalty = 10.0 * np.square(max(0.0, 0.9 - upright_cos))  # Only penalize severe tilt (> ~45 degrees)
        
        # Scale progress reward by posture quality so it doesn't just "dive forward" to get high scores before falling
        # Also penalize excessively long legs by dividing by initial height (which is proportional to leg length)
        posture_multiplier = 1.0 - (drift_penalty + pitch_penalty)
        posture_multiplier = max(0.1, posture_multiplier) # Ensure it doesn't go below 0.1
        
        # Normalize progress by initial height so longer legs don't get an unfair advantage
        # self.init_qpos[2] is the starting height of the main body
        height_normalization = 1.0 / max(0.1, self.init_qpos[2])
        # print(f"height_normalization: {height_normalization}")
        
        progress_reward = progress_velocity * self._forward_reward_weight if upright_cos > 0.90 else 0.0
        
        reward = progress_reward - pitch_penalty

        reward_terms = {
            "progress_reward": progress_reward,
            "alive_bonus": alive_bonus,
            "drift_penalty": drift_penalty,
            "stagnation_penalty": stagnation_penalty,
            "pitch_penalty": pitch_penalty,
            "x_velocity": velocity[0],
            "progress_velocity": progress_velocity,
        }
        return reward, reward_terms

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, velocity))

    def apply_force(self):
        body_id = self.body_ids
        force = self.force
        pert = self.np_random.uniform(
            low=-0.1, high=0.1, size=3)
        rot = quat2rot([1, *pert])
        force = np.dot(rot, force.reshape(2, 3).T).T.flatten()
        self.data.xfrc_applied[body_id] = force

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        self.previous_state = observation
        self.prev_x_velocity = 0.0
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
