import numpy as np
from gym import utils
from envs import mujoco_env
from mujoco_py import MjViewer
import os

class Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, env_name, rand_seed, maximum_length, params={}):
        self.env_name = 'point_multi'
        self.seeding = False
        self.real_step = True
        self.env_timestep = self.last_switch = 0

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, env_name, rand_seed, \
            maximum_length, curr_dir+'/assets/point.xml', 2, params)
        utils.EzPickle.__init__(self)
        self.observation_dim = 9
        self._old_obs = np.zeros([self.observation_dim])
        self.action_dim = 2

        self.composite_mode = 0
        self.threshold = 0.01

        self.point_sid = self.model.site_name2id("pointmass")
        self.box_sid = self.model.site_name2id("object0")
        self.target0_sid = self.model.site_name2id("target0")
        self.target1_sid = self.model.site_name2id("target1")

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)

        timesteps_since_switch = self.env_timestep - self.last_switch
        if self.composite_mode == 0:
            point_pos = self.data.site_xpos[self.point_sid]
            target_pos = self.data.site_xpos[self.target0_sid]
            dist = np.linalg.norm(point_pos-target_pos)
            reward = - 10.0 * dist - 0.25 * np.linalg.norm(self.data.qvel) + \
                     1.0 * (timesteps_since_switch) ** 2

            if dist < self.threshold:
                self.transition_1_to_2()

        else:
            box_pos = self.data.site_xpos[self.box_sid]
            target_pos = self.data.site_xpos[self.target1_sid]
            dist = np.linalg.norm(box_pos-target_pos)
            reward = - 10.0 * dist - 0.25 * np.linalg.norm(self.data.qvel) + \
                     1.0 * (timesteps_since_switch) ** 2

            if dist < self.threshold:
                self.transition_2_to_1()

        ob = self._get_obs()
        # keep track of env timestep (needed for continual envs)
        self.env_timestep += 1
        return ob, reward, (self.env_timestep == self._maximum_length), self.get_env_infos()

    def step(self, a):
        # overloading to preserve backwards compatibility
        return self._step(a)

    def _get_obs(self):
        if self.composite_mode == 0:
            ret = self._old_obs = np.concatenate([
                self.data.qpos.flat,
                self.data.qvel.flat,
                self.data.site_xpos[self.box_sid],
                self.data.site_xpos[self.target0_sid],
                0
            ])

        else:
            ret = self._old_obs = np.concatenate([
                self.data.qpos.flat,
                self.data.qvel.flat,
                self.data.site_xpos[self.box_sid],
                self.data.site_xpos[self.target1_sid],
                1
            ])

        return ret

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target0_reset(self):
        target0_pos = np.array([0.1, 0.1, 0.1])
        if self.seeding is True:
            target0_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
            target0_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
            target0_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        self.model.site_pos[self.target0_sid] = target0_pos

    def target1_reset(self):
        target1_pos = np.array([0.1, 0.1, 0.1])
        if self.seeding is True:
            target1_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
            target1_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
            target1_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        self.model.site_pos[self.target1_sid] = target1_pos

    def box_reset(self):
        box_pos = np.array([0.1, 0.1, 0.1])
        if self.seeding is True:
            box_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
            box_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
            box_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        self.model.site_pos[self.box_sid] = box_pos

    def transition_1_to_2(self):
        self.target1_reset()
        self.box_reset()
        self.composite_mode = 1
        self.sim.forward()

        self.last_switch = self.env_timestep

    def transition_2_to_1(self):
        self.target1_reset()
        self.box_reset()
        self.composite_mode = 1
        self.sim.forward()

        self.last_switch = self.env_timestep

    def reset(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        self.target0_reset()
        self.target1_reset()
        self.box_reset()
        self.sim.forward()
        self.env_timestep = 0
        return self._get_obs(), 0, False, {}


    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.site_pos[self.target_sid].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos, timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        self.set_state(qp, qv)
        self.model.site_pos[self.target_sid] = target_pos
        self.env_timestep = state['timestep']
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 1.2