"""FlipCoin environment converted to the MuJoCo Playground standard.

Port of ``maniskill_elirobots.tasks.flip_coin_ec.FlipCoinEnv`` to an
``mjx_env.MjxEnv`` following the ``mujoco_playground`` conventions, running on
``mujoco.mjx`` (the JAX backend of MuJoCo). The scene is the one procedurally
built by ``mujoco_elirobots.builder.task`` and serialized to
``xmls/flip_coin_ec.xml``.

Task description
----------------
Flip a two-sided coin (red/blue) over so that it ends up close to a floating
goal sphere, with its face normal aligned with the world ``+z`` axis and at
rest.

Randomizations
--------------
- Coin position: sampled uniformly within ``+-2 * coin_radius`` on the floor.
- Goal position: sampled on a ring of radius ``4 * coin_radius`` around the
  coin.
- Goal height: sampled uniformly in
  ``[coin_half_length, coin_half_length + coin_max_height] + 200 mm``.

Success conditions
------------------
The coin is within ``goal_thresh`` of the goal sphere, its angular distance to
the desired axis is below ``45 deg``, and its linear velocity is below
``qvel_tolerance``.
"""

import pathlib
from dataclasses import asdict, fields
from typing import Any, final, override

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from dacite import from_dict
from ml_collections import ConfigDict, config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import (
    State,  # pylint: disable=g-importing-member
)
from tqdm import trange
from typeguard import check_type, typechecked

from mujoco_elirobots.assets import ASSETS_PATH
from mujoco_elirobots.playground.flip_coin_ec.config import FlipCoinConfig
from mujoco_elirobots.playground.flip_coin_ec.scene_builder import build_task

_XML_PATH = pathlib.Path(__file__).parent / "xmls" / "flip_coin_ec.xml"
_EC63_MESHES_DIR = ASSETS_PATH / "ec63" / "meshes"

_ROBOT_XML_PATH = ASSETS_PATH / "ec63" / "ec63_description.urdf"

_COIN_NORMAL_AXIS = jnp.array([1.0, 0.0, 0.0])
_COIN_DESIRED_AXIS = jnp.array([0.0, 0.0, 1.0])


@final
class FlipCoinEnv(mjx_env.MjxEnv):
    """Flip a two-sided coin over.

    The coin starts flat on the floor (red face up) and must be flipped and
    carried to the goal sphere so that its face normal points along the world
    ``+z`` axis (blue face up).
    """

    def __init__(
        self,
        config: FlipCoinConfig | None = None,
        config_overrides: dict[str, str | int | list[Any]] | None = None,
    ):
        if config is None:
            config = FlipCoinConfig()
        super().__init__(ConfigDict(asdict(config)), config_overrides)

        self._action_scale = config.action_scale

        self._model_assets = self._get_assets()

        mj_model = build_task(str(_ROBOT_XML_PATH), task_args=config.scene_config)
        # mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)

        mj_model.opt.timestep = self.sim_dt
        # The MJX JAX backend only supports the semi-implicit integrators, so the
        # fully implicit integrator cannot be used here. Arm joint damping keeps
        # the stiff position-controlled arm stable under semi-implicit integration.
        mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        arm_dofs = [
            mj_model.jnt_dofadr[mj_model.joint(f"robot_joint{i}").id]
            for i in range(1, 7)
        ]
        mj_model.dof_damping[arm_dofs] = config.arm_damping

        self._mj_model = mj_model
        self._mjx_model = check_type(
            mjx.put_model(mj_model, impl=self._config.impl),
            mjx.Model,
        )
        self._post_init()

    @staticmethod
    def _get_assets() -> dict[str, bytes]:
        assets: dict[str, bytes] = {}
        for f in _EC63_MESHES_DIR.glob("*.STL"):
            assets["meshes/" + f.name] = f.read_bytes()
        return assets

    def _post_init(self) -> None:
        arm_joints = [f"robot_joint{i}" for i in range(1, 7)]
        finger_joints = ["robot_finger_1_joint", "robot_finger_2_joint"]
        all_joints = arm_joints + finger_joints

        arm_jnt_ids = np.array([self._mj_model.joint(j).id for j in arm_joints])
        all_jnt_ids = np.array([self._mj_model.joint(j).id for j in all_joints])
        self._arm_qveladr = self._mj_model.jnt_dofadr[arm_jnt_ids]
        self._robot_qposadr = self._mj_model.jnt_qposadr[all_jnt_ids]

        limited = self._mj_model.jnt_limited[all_jnt_ids]
        ranges = self._mj_model.jnt_range[all_jnt_ids]
        self._lowers = jnp.where(limited, ranges[:, 0], -jnp.inf).astype(jnp.float32)
        self._uppers = jnp.where(limited, ranges[:, 1], jnp.inf).astype(jnp.float32)

        self._coin_body = check_type(self._mj_model.body("coin").id, int)
        self._goal_region_body = check_type(self._mj_model.body("goal_region").id, int)
        self._goal_sphere_body = check_type(self._mj_model.body("goal_sphere").id, int)
        self._tcp_body = check_type(self._mj_model.body("robot_claw_tcp_link").id, int)
        self._finger1_body = check_type(
            self._mj_model.body("robot_claw_finger_1").id, int
        )
        self._finger2_body = check_type(
            self._mj_model.body("robot_claw_finger_2").id, int
        )

        self._coin_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body("coin").jntadr[0]
        ]
        self._goal_region_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body("goal_region").jntadr[0]
        ]
        self._goal_sphere_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body("goal_sphere").jntadr[0]
        ]

        self._init_q = self._mj_model.qpos0
        self._init_ctrl = self._init_q[self._robot_qposadr]

        self._obs_size = (
            self._mjx_model.nq + self._mjx_model.nv + 1 + 3 + 3 + 3 + 4 + 3 + 3 + 1
        )

        init_data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, init_data)
        self._init_tcp_pos = jnp.array(
            init_data.xpos[self._tcp_body], dtype=jnp.float32
        )
        self._init_finger1_pos = jnp.array(
            init_data.xpos[self._finger1_body], dtype=jnp.float32
        )
        self._init_finger2_pos = jnp.array(
            init_data.xpos[self._finger2_body], dtype=jnp.float32
        )

    @property
    @override
    @typechecked
    def xml_path(self) -> str:
        return ""

    @property
    @override
    @typechecked
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    @override
    @typechecked
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    @override
    @typechecked
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    @override
    def observation_size(self) -> int:
        return self._obs_size

    @property
    @typechecked
    def config(self) -> FlipCoinConfig:
        return from_dict(FlipCoinConfig, self._config.to_dict())

    @override
    def reset(self, rng: jax.Array) -> State:
        rng, rng_coin, rng_theta, rng_height = jax.random.split(rng, 4)

        coin_xy = (jax.random.uniform(rng_coin, (2,), minval=-1.0, maxval=1.0)) * (
            2.0 * self.config.coin_radius
        )
        theta = 2.0 * jnp.pi * jax.random.uniform(rng_theta)
        goal_xy = coin_xy + 4.0 * self.config.coin_radius * jnp.array(
            [jnp.sin(theta), jnp.cos(theta)]
        )
        sphere_height = (
            self.config.coin_half_length
            + jax.random.uniform(rng_height) * self.config.coin_max_height
            + 200e-3
        )

        init_q = jnp.array(self._init_q, dtype=jnp.float32)
        init_q = init_q.at[self._coin_qposadr : self._coin_qposadr + 2].set(coin_xy)
        init_q = init_q.at[
            self._goal_region_qposadr : self._goal_region_qposadr + 2
        ].set(goal_xy)
        init_q = init_q.at[
            self._goal_sphere_qposadr : self._goal_sphere_qposadr + 2
        ].set(goal_xy)
        init_q = init_q.at[self._goal_sphere_qposadr + 2].set(sphere_height)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jnp.zeros(self._mjx_model.nv, dtype=jnp.float32),
            ctrl=jnp.array(self._init_ctrl, dtype=jnp.float32),
            impl=self._mjx_model.impl.value,
            naconmax=self.config.naconmax,
            naccdmax=self.config.naccdmax,
            njmax=self.config.njmax,
        )

        goal_pos = data.qpos[self._goal_sphere_qposadr : self._goal_sphere_qposadr + 3]
        info = {"goal_pos": goal_pos}
        metrics = {
            "success": jnp.array(0.0),
            "out_of_bounds": jnp.array(0.0),
            **{
                str(k): jnp.array(0.0) for k in fields(self.config.reward_config.scales)
            },
        }
        obs = self._get_obs(data, info, from_qpos=True)
        reward, done = jnp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    @override
    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = check_type(state.data.ctrl + delta, jax.Array)
        ctrl = jnp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        raw_rewards = self._get_reward(data, state.info)

        rewards = {
            k: v * self.config.reward_config.scales[k] for k, v in raw_rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()), -1e4, 1e4)

        coin_pos = data.xpos[self._coin_body]
        coin_vel = data.cvel[self._coin_body, :3]
        obj_to_goal_dist = math.norm(state.info["goal_pos"] - coin_pos)
        is_obj_placed = obj_to_goal_dist <= self.config.goal_thresh
        is_obj_static = math.norm(coin_vel) <= self.config.qvel_tolerance
        angle_dist = self._coin_angle_deg(data.xquat[self._coin_body])
        is_angle_zero = angle_dist < 45.0
        success = is_obj_placed & is_obj_static & is_angle_zero

        out_of_bounds = jnp.any(jnp.abs(coin_pos) > 2.0) | (coin_pos[2] < -0.5)
        done = (
            success
            | out_of_bounds
            | jnp.isnan(data.qpos).any()
            | jnp.isnan(data.qvel).any()
        )
        done = done.astype(float)

        reward = jnp.where(success, self.config.reward_config.success_bonus, reward)

        metrics = {
            **state.metrics,
            **raw_rewards,
            "success": success.astype(float),
            "out_of_bounds": out_of_bounds.astype(float),
        }

        obs = self._get_obs(data, state.info)
        return State(data, obs, reward, done, metrics, state.info)

    def _get_reward(
        self, data: mjx.Data, info: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        coin_pos = data.xpos[self._coin_body]
        tcp_pos = data.xpos[self._tcp_body]
        goal_pos = info["goal_pos"]
        coin_quat = data.xquat[self._coin_body]
        angle_dist = self._coin_angle_deg(coin_quat)

        reaching = jnp.exp(-5.0 * math.norm(tcp_pos - coin_pos))
        place = jnp.exp(-5.0 * math.norm(goal_pos - coin_pos))
        static = jnp.exp(
            -self.config.qvel_penalty * math.norm(data.qvel[self._arm_qveladr])
        )
        angle = jnp.cos(angle_dist * jnp.pi / 180.0)

        is_grasped = self._is_grasped(
            coin_pos,
            coin_quat,
            data.xpos[self._finger1_body],
            data.xpos[self._finger2_body],
        )
        is_obj_placed = check_type(
            math.norm(goal_pos - coin_pos) <= self.config.goal_thresh, jax.Array
        )
        is_angle_zero = angle_dist < 45.0

        return {
            "reaching": reaching,
            "grasp": is_grasped.astype(float),
            "angle": angle * is_grasped.astype(float),
            "place": place * is_grasped.astype(float) * is_angle_zero.astype(float),
            "static": (
                static
                * is_obj_placed.astype(float)
                * is_grasped.astype(float)
                * is_angle_zero.astype(float)
            ),
        }

    def _get_obs(
        self, data: mjx.Data, info: dict[str, jax.Array], from_qpos: bool = False
    ) -> jax.Array:
        if from_qpos:
            coin_pos = data.qpos[self._coin_qposadr : self._coin_qposadr + 3]
            coin_quat = data.qpos[self._coin_qposadr + 3 : self._coin_qposadr + 7]
            tcp_pos = self._init_tcp_pos
            finger1_pos = self._init_finger1_pos
            finger2_pos = self._init_finger2_pos
        else:
            coin_pos = data.xpos[self._coin_body]
            coin_quat = data.xquat[self._coin_body]
            tcp_pos = data.xpos[self._tcp_body]
            finger1_pos = data.xpos[self._finger1_body]
            finger2_pos = data.xpos[self._finger2_body]
        angle_dist = self._coin_angle_deg(coin_quat)
        is_grasped = self._is_grasped(coin_pos, coin_quat, finger1_pos, finger2_pos)
        return jnp.concatenate(
            [
                data.qpos,
                data.qvel,
                is_grasped.astype(float).reshape(1),
                tcp_pos,
                info["goal_pos"],
                coin_pos,
                coin_quat,
                coin_pos - tcp_pos,
                info["goal_pos"] - coin_pos,
                jnp.array([angle_dist * jnp.pi / 180.0]),
            ]
        )

    def _coin_angle_deg(self, coin_quat: jax.Array) -> jax.Array:
        normal = math.rotate(_COIN_NORMAL_AXIS, coin_quat)
        cos_sim = jnp.sum(normal * _COIN_DESIRED_AXIS) / math.norm(normal)
        cos_sim = jnp.clip(cos_sim, -1.0, 1.0)
        return jnp.arccos(cos_sim) * 180.0 / jnp.pi

    def _is_grasped(
        self,
        coin_pos: jax.Array,
        coin_quat: jax.Array,
        finger1_pos: jax.Array,
        finger2_pos: jax.Array,
    ) -> jax.Array:
        normal = math.rotate(_COIN_NORMAL_AXIS, coin_quat)
        travel_dir = finger1_pos - finger2_pos
        travel_dir = travel_dir / math.norm(travel_dir)
        in_gripper = (math.norm(coin_pos - finger1_pos) < 0.04) & (
            math.norm(coin_pos - finger2_pos) < 0.04
        )
        aligned = jnp.abs(jnp.sum(normal * travel_dir)) < 0.1
        return in_gripper & aligned


if __name__ == "__main__":
    print("Les goo")
    env = FlipCoinEnv()
    print("Env created")

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    print("Env reset")

    trajectory = [state]
    print(
        "Note: the first step JIT-compiles the physics and can take a while on CPU; "
        + "use impl='warp' with a GPU for fast startup.",
        flush=True,
    )
    # while not state.done and len(trajectory) < env.config.episode_length:
    # for _ in trange(env.config.episode_length):
    for _ in trange(10):
        rng, rng_action = jax.random.split(rng)
        # action = jax.random.uniform(
        #     rng_action, (env.action_size,), minval=-1.0, maxval=1.0
        # )

        action = jnp.zeros((env.action_size,))
        state = env.step(state, action)
        trajectory.append(state)
        if state.done:
            print("Done!")
            break

    frames = env.render(trajectory, camera="render_camera", height=480, width=640)
    print(f"Rollout of {len(frames)} frames, done={bool(state.done)}")

    try:
        import mediapy as media
        from PIL import Image

        Image.fromarray(frames[0]).save(
            pathlib.Path(__file__).parent / "flip_coin_ec.gif",
            save_all=True,
            append_images=[Image.fromarray(f) for f in frames[1:]],
            duration=50,
            loop=0,
        )

        media.write_video("robot_run.mp4", frames, fps=25)

    except ImportError:
        pass
