from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from pathlib import Path
import mujoco

DEFAULT_SIZE = 500


class Mujoco3Env:
    """Superclass for all MuJoCo3 environments."""

    def __init__(self, mujoco_model=None, frame_skip=15):
        if "<mujoco" in mujoco_model:
            # is string mujoco model
            self.model = mujoco.MjModel.from_xml_string(mujoco_model)
        elif path.exists(mujoco_model):
            # is mujoco path
            self.model = mujoco.MjModel.from_xml_path(mujoco_model)
        elif not path.exists(mujoco_model):
            mujoco_model = path.join(
                Path(__file__).parent.parent.parent.parent,
                "assets/mujoco_models",
                path.basename(mujoco_model),
            )
            if not path.exists(mujoco_model):
                raise IOError("File %s does not exist" % mujoco_model)
            else:
                self.model = mujoco.MjModel.from_xml_path(mujoco_model)

        self.frame_skip = frame_skip
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewers = {}
        self.obs_dim = None
        self.action_space = None
        self.observation_space = None
        self.np_random = None
        self.cur_t = 0  # number of steps taken

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.prev_qpos = None
        self.prev_qvel = None
        self.seed()

    def set_spaces(self):
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        self.obs_dim = observation.size
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------
    def step(self, action):
        """
        Step the environment forward.
        """
        raise NotImplementedError

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self, mode):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self, tracking=False):
        mujoco.mj_resetData(self.model, self.data)
        if not tracking:
            self.cur_t = 0
        ob = self.reset_model(tracking)
        old_viewer = self.viewer
        for mode, v in self._viewers.items():
            self.viewer = v
            self.viewer_setup(mode)
        self.viewer = old_viewer
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == "image":
            self._get_viewer(mode).render(width, height)
            if "human" in self._viewers and not self._viewers['human'] is None:
                self._viewers['image'].cam.lookat[:] = self._viewers['human'].cam.lookat
                self._viewers['image'].cam.azimuth = self._viewers['human'].cam.azimuth
                self._viewers['image'].cam.elevation = self._viewers['human'].cam.elevation
                self._viewers['image'].cam.distance = self._viewers['human'].cam.distance
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it, and the image format is BGR for OpenCV
            return data[::-1, :, [2, 1, 0]]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        return self.viewer

    def set_custom_key_callback(self, key_func):
        self._get_viewer("human").custom_key_callback = key_func

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def vec_body2world(self, body_name, vec):
        body_xmat = self.data.get_body_xmat(body_name)
        vec_world = (body_xmat @ vec[:, None]).ravel()
        return vec_world

    def pos_body2world(self, body_name, pos):
        body_xpos = self.data.get_body_xpos(body_name)
        body_xmat = self.data.get_body_xmat(body_name)
        pos_world = (body_xmat @ pos[:, None]).ravel() + body_xpos
        return pos_world
