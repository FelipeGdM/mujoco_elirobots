from dataclasses import dataclass, field
from pyexpat import model

import mujoco
from mujoco._specs import MjSpec
from mujoco._structs import MjModel
from typeguard import check_type, typechecked


@dataclass
class MotorArgs:
    kp: float = 400.0
    kv: float = 80.0


@dataclass
class RobotModelArgs:
    joint1_params: MotorArgs = field(default_factory=MotorArgs)
    joint2_params: MotorArgs = field(default_factory=MotorArgs)
    joint3_params: MotorArgs = field(default_factory=MotorArgs)
    joint4_params: MotorArgs = field(default_factory=MotorArgs)
    joint5_params: MotorArgs = field(default_factory=MotorArgs)
    joint6_params: MotorArgs = field(default_factory=MotorArgs)


@typechecked
def build_robot(
    robot_filename: str, robot_model_args: RobotModelArgs | None = None
) -> MjSpec:
    if robot_model_args is None:
        robot_model_args = RobotModelArgs()

    robot_spec = mujoco.MjSpec.from_file(robot_filename)

    def _disable_gravity(n: int):
        robot_spec.body(f"link{n}").gravcomp = 1

    _ = [_disable_gravity(n) for n in range(1, 7)]

    def _create_actuator(n: int, joint_params: MotorArgs):

        act = robot_spec.add_actuator(
            name=f"actuator{n}",
            target=f"joint{n}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
        )

        act.set_to_position(joint_params.kp, joint_params.kv)

        return act

    _ = [
        _create_actuator(n, joint_params)
        for n, joint_params in enumerate(
            [
                robot_model_args.joint1_params,
                robot_model_args.joint2_params,
                robot_model_args.joint3_params,
                robot_model_args.joint4_params,
                robot_model_args.joint5_params,
                robot_model_args.joint6_params,
            ],
            start=1,
        )
    ]

    return robot_spec


def build_cube(size: float = 20e-3, mass: float = 50e-3):
    spec = mujoco.MjSpec()

    # Cube body with a free joint
    body = spec.worldbody.add_body(name="cube", pos=[0, 0, size])
    _ = body.add_freejoint()

    # Two half-boxes forming one cube
    _ = body.add_geom(
        name="cube_red",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[size / 2, size, size],
        pos=[-size / 2, 0, 0],
        rgba=[1, 0, 0, 1],
        mass=mass / 2,
    )
    _ = body.add_geom(
        name="cube_blue",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[size / 2, size, size],
        pos=[size / 2, 0, 0],
        rgba=[0, 0, 1, 1],
        mass=mass / 2,
    )

    return spec


def build_world(
    scene_filename: str,
    robot_filename: str,
    robot_model_args: RobotModelArgs | None = None,
) -> MjModel:

    scene_spec = mujoco.MjSpec.from_file(scene_filename)

    robot_attach_frame = scene_spec.worldbody.add_frame(
        pos=[0, 0, 0],  # world position
    )

    robot_spec = build_robot(robot_filename, robot_model_args)

    _ = robot_attach_frame.attach_body(
        robot_spec.body("base_link"),
        prefix="robot_",  # avoids name clashes
        # pos=[0, 0, 0],
    )

    cube_spec = build_cube()

    _ = scene_spec.attach(
        cube_spec,
        frame=scene_spec.worldbody.add_frame(
            pos=[0, -0.5, 0],  # world position
        ),
    )

    return check_type(scene_spec.compile(), mujoco.MjModel)  # pyright: ignore[reportAny]
