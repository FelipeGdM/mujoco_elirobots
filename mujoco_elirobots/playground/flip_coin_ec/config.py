from dataclasses import dataclass

import numpy as np

INIT_QPOS = (
    0.0,
    4 * np.pi / 8,
    -5 * np.pi / 8,
    3 * np.pi / 8,
    -4 * np.pi / 8,
    0.0,
    0.0,
    0.0,
)


@dataclass(frozen=True)
class FlipCoinSceneConfig:
    coin_half_length: float = 5e-3
    coin_radius: float = 18e-3
    coin_mass: float = 2.6e-2
    coin_max_height: float = 200e-3

    goal_thresh: float = 25e-3
    goal_radius: float = 0.1

    robot_init_pos: tuple[float, float, float] = (-0.4, 0.0, 0.0)
    robot_init_qpos: tuple[float, float, float, float, float, float, float, float] = (
        INIT_QPOS
    )

    gripper_kp: float = 1e3
    gripper_kv: float = 1e2

    initial_coin_pos: tuple[float, float, float] | None = None
    initial_goal_pos: tuple[float, float, float] | None = None
    initial_goal_height: float | None = None


@dataclass(frozen=True)
class FlipCoinScales:
    reaching: float = 1.0
    grasp: float = 1.0
    angle: float = 1.0
    place: float = 1.0
    static: float = 1.0

    def __getitem__(self, key: str):
        return self.__dict__[key]


@dataclass(frozen=True)
class FlipCoinRewardConfig:
    scales: FlipCoinScales = FlipCoinScales()
    success_bonus: float = 25.0


@dataclass(frozen=True)
class FlipCoinConfig:
    ctrl_dt: float = 0.02
    sim_dt: float = 0.005
    episode_length: int = 50
    action_repeat: int = 1
    action_scale: float = 0.1
    goal_thresh: float = 25e-3
    goal_radius: float = 0.1
    coin_half_length: float = 5e-3
    coin_radius: float = 18e-3
    coin_max_height: float = 200e-3
    qvel_tolerance: float = 0.2
    qvel_penalty: float = 1.0
    arm_damping: float = 5.0
    reward_config: FlipCoinRewardConfig = FlipCoinRewardConfig()
    impl: str = "jax"
    naconmax: int = 64 * 256  # 16384
    naccdmax: int = 64 * 256  # 16384
    njmax: int = 128
    scene_config: FlipCoinSceneConfig = FlipCoinSceneConfig()
