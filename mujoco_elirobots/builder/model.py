from dataclasses import dataclass, field

import mujoco
from mujoco._structs import MjModel
from typeguard import check_type


@dataclass
class MotorArgs:
    kp: float = 400.0
    kv: float = 80.0


@dataclass
class ModelArgs:
    joint1_params: MotorArgs = field(default_factory=MotorArgs)
    joint2_params: MotorArgs = field(default_factory=MotorArgs)
    joint3_params: MotorArgs = field(default_factory=MotorArgs)
    joint4_params: MotorArgs = field(default_factory=MotorArgs)
    joint5_params: MotorArgs = field(default_factory=MotorArgs)
    joint6_params: MotorArgs = field(default_factory=MotorArgs)


def build_model(
    scene_filename: str, robot_filename: str, model_args: ModelArgs | None = None
) -> MjModel:

    if model_args is None:
        model_args = ModelArgs()

    scene_spec = mujoco.MjSpec.from_file(scene_filename)
    robot_spec = mujoco.MjSpec.from_file(robot_filename)

    attach_frame = scene_spec.worldbody.add_frame(
        pos=[0, 0, 0],  # world position
    )

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
                model_args.joint1_params,
                model_args.joint2_params,
                model_args.joint3_params,
                model_args.joint4_params,
                model_args.joint5_params,
                model_args.joint6_params,
            ],
            start=1,
        )
    ]

    _ = attach_frame.attach_body(
        robot_spec.body("base_link"),
        prefix="robot_",  # avoids name clashes
        # pos=[0, 0, 0],
    )

    return check_type(scene_spec.compile(), mujoco.MjModel)  # pyright: ignore[reportAny]
