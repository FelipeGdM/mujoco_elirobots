"""MuJoCo scene builder for the FlipCoin-v1 task, converted from ManiSkill.

Recreates the scene of ``maniskill_elirobots.tasks.flip_coin_ec`` using MuJoCo's
procedural API (``MjSpec``): an EC63 robot mounted on a table, a two-sided coin
to be flipped, a bullseye goal region on the table and a floating goal sphere
marking where the flipped coin must end up.

ManiSkill element -> MuJoCo equivalent
    coin (two-color peg)      -> free body, box of half size [length, width,
                                 width], split into two colored half-boxes
    goal_region (red/white)   -> free body (gravcomp=1), bullseye of concentric
                                 thin cylinders, collision disabled
    goal_sphere (transparent) -> free body (gravcomp=1), transparent sphere,
                                 collision disabled
    table (TableSceneBuilder) -> static box geom + ground plane
    agent (EC63)              -> URDF robot, gravity-compensated arm, position
                                 actuators for the arm and gripper
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco._specs import MjsBody, MjSpec
from mujoco._structs import MjModel
from typeguard import check_type, typechecked

from mujoco_elirobots.builder.model import RobotModelArgs, build_robot

COIN_RED = [1, 0, 0, 1]
COIN_BLUE = [0, 0, 1, 1]

TARGET_RED = [194 / 255, 19 / 255, 22 / 255, 1]
TARGET_WHITE = [1, 1, 1, 1]

GOAL_SPHERE_GREEN = [0, 1, 0, 0.5]

TABLE_RGBA = [0.55, 0.35, 0.2, 1]
FLOOR_RGBA = [0.4, 0.4, 0.4, 1]

INIT_QPOS = (
    0.0,
    -6 * np.pi / 8,
    5 * np.pi / 8,
    -3 * np.pi / 8,
    4 * np.pi / 8,
    0.0,
    0.0,
    0.0,
)


@dataclass
class FlipCoinArgs:
    coin_half_length: float = 5e-3
    coin_radius: float = 18e-3
    coin_mass: float = 2.6e-2
    coin_max_height: float = 200e-3

    goal_thresh: float = 25e-3
    goal_radius: float = 0.1

    table_length: float = 2.418
    table_width: float = 1.209
    table_height: float = 0.9196429
    table_center_xy: tuple[float, float] = (-0.12, 0.0)

    robot_init_pos: tuple[float, float, float] = (-0.4, 0.0, 0.0)
    robot_init_qpos: tuple[float, float, float, float, float, float, float, float] = INIT_QPOS

    gripper_kp: float = 1e3
    gripper_kv: float = 1e2

    initial_coin_pos: tuple[float, float, float] | None = None
    initial_goal_pos: tuple[float, float, float] | None = None
    initial_goal_height: float | None = None


def _euler_y_quat(angle: float) -> tuple[float, float, float, float]:
    return (np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0)


def _look_at_quat(eye: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    z_axis = -forward
    up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s

    return (float(w), float(x), float(y), float(z))


@typechecked
def build_coin(spec: MjSpec, args: FlipCoinArgs) -> MjsBody:
    half_length = args.coin_half_length
    width = args.coin_radius

    initial_coin_pos = args.initial_coin_pos if args.initial_coin_pos is not None else (0.0, 0.0, 2 * half_length)

    coin = spec.worldbody.add_body(
        name="coin",
        pos=initial_coin_pos,
        quat=_euler_y_quat(np.pi / 2),
    )
    _ = coin.add_freejoint()

    _ = coin.add_geom(
        name="coin_red",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half_length, width, width],
        pos=[-half_length, 0, 0],
        rgba=COIN_RED,
        mass=args.coin_mass / 2,
    )
    _ = coin.add_geom(
        name="coin_blue",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half_length, width, width],
        pos=[half_length, 0, 0],
        rgba=COIN_BLUE,
        mass=args.coin_mass / 2,
    )

    return coin


@typechecked
def build_goal_region(spec: MjSpec, args: FlipCoinArgs) -> MjsBody:
    initial_goal_pos = args.initial_goal_pos if args.initial_goal_pos is not None else (0.1, 0.0, 1e-5)

    region = spec.worldbody.add_body(
        name="goal_region",
        pos=initial_goal_pos,
        quat=_euler_y_quat(np.pi / 2),
    )
    _ = region.add_freejoint()
    region.gravcomp = 1.0

    thickness = 1e-5
    radii = [args.goal_radius, args.goal_radius * 4 / 5, args.goal_radius * 3 / 5, args.goal_radius * 2 / 5, args.goal_radius / 5]
    colors = [TARGET_RED, TARGET_WHITE, TARGET_RED, TARGET_WHITE, TARGET_RED]

    for i, (radius, color) in enumerate(zip(radii, colors)):
        _ = region.add_geom(
            name=f"goal_region_ring_{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, thickness / 2 + i * 1e-5],
            rgba=color,
            contype=0,
            conaffinity=0,
        )

    return region


@typechecked
def build_goal_sphere(spec: MjSpec, args: FlipCoinArgs) -> MjsBody:
    initial_goal_pos = args.initial_goal_pos if args.initial_goal_pos is not None else (0.1, 0.0, 1e-5)
    initial_goal_height = (
        args.initial_goal_height
        if args.initial_goal_height is not None
        else args.coin_half_length + args.coin_max_height + 200e-3
    )

    sphere = spec.worldbody.add_body(
        name="goal_sphere",
        pos=[initial_goal_pos[0], initial_goal_pos[1], initial_goal_height],
    )
    _ = sphere.add_freejoint()
    sphere.gravcomp = 1.0

    _ = sphere.add_geom(
        name="goal_sphere_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[args.goal_thresh],
        rgba=GOAL_SPHERE_GREEN,
        contype=0,
        conaffinity=0,
    )

    return sphere


@typechecked
def build_table(spec: MjSpec, args: FlipCoinArgs) -> None:
    cx, cy = args.table_center_xy

    _ = spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[args.table_width / 2, args.table_length / 2, args.table_height / 2],
        pos=[cx, cy, -args.table_height / 2],
        rgba=TABLE_RGBA,
    )

    _ = spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20, 20, 0.1],
        pos=[0, 0, -args.table_height],
        rgba=FLOOR_RGBA,
    )


@typechecked
def _add_cameras(spec: MjSpec) -> None:
    _ = spec.worldbody.add_camera(
        name="base_camera",
        mode=mujoco.mjtCamera.mjCAMERA_FIXED,
        pos=[0.3, 0, 0.6],
        quat=_look_at_quat(np.array([0.3, 0, 0.6]), np.array([-0.1, 0, 0.1])),
        fovy=np.pi / 2,
    )
    _ = spec.worldbody.add_camera(
        name="render_camera",
        mode=mujoco.mjtCamera.mjCAMERA_FIXED,
        pos=[0.15, 0, 0.5],
        quat=_look_at_quat(np.array([0.15, 0, 0.5]), np.array([-0.2, 0, 0.2])),
        fovy=1.5,
    )


@typechecked
def _set_robot_init_qpos(robot_spec: MjSpec, args: FlipCoinArgs) -> None:
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "finger_1_joint", "finger_2_joint"]
    for name, value in zip(joint_names, args.robot_init_qpos):
        robot_spec.joint(name).ref = value


@typechecked
def _add_gripper_actuators(spec: MjSpec, args: FlipCoinArgs) -> None:
    for name, joint in [("robot_actuator7", "robot_finger_1_joint"), ("robot_actuator8", "robot_finger_2_joint")]:
        actuator = spec.add_actuator(
            name=name,
            target=joint,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
        )
        actuator.set_to_position(args.gripper_kp, args.gripper_kv)


@typechecked
def build_task(
    robot_filename: str,
    robot_model_args: RobotModelArgs | None = None,
    task_args: FlipCoinArgs | None = None,
) -> MjModel:
    if robot_model_args is None:
        robot_model_args = RobotModelArgs()
    if task_args is None:
        task_args = FlipCoinArgs()

    scene_spec = mujoco.MjSpec()
    scene_spec.compiler.degree = False
    scene_spec.compiler.balanceinertia = True

    build_table(scene_spec, task_args)
    _add_cameras(scene_spec)

    robot_spec = build_robot(robot_filename, robot_model_args)
    _set_robot_init_qpos(robot_spec, task_args)

    robot_attach_frame = scene_spec.worldbody.add_frame(pos=task_args.robot_init_pos)
    _ = robot_attach_frame.attach_body(robot_spec.body("base_link"), prefix="robot_")
    _add_gripper_actuators(scene_spec, task_args)

    _ = build_coin(scene_spec, task_args)
    _ = build_goal_region(scene_spec, task_args)
    _ = build_goal_sphere(scene_spec, task_args)

    return check_type(scene_spec.compile(), MjModel)
