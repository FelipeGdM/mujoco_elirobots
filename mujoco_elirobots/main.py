import mujoco
import mujoco.viewer
from mujoco._specs import MjsActuator
from typeguard import check_type

SCENE_FILENAME = "mujoco_elirobots/scene.xml"
ROBOT_FILENAME = "mujoco_elirobots/assets/ec63/ec63_description.urdf"

n_frames = 180
height = 240
width = 320
frames = []
fps = 60.0
times = []
sensordata = []


def main():

    # 1. Load the scene and robot as Spec objects
    scene_spec = mujoco.MjSpec.from_file(SCENE_FILENAME)
    robot_spec = mujoco.MjSpec.from_file(ROBOT_FILENAME)

    attach_frame = scene_spec.worldbody.add_frame(
        pos=[0, 0, 0],  # world position
    )

    # joint1 = robot_spec.joint("joint1")

    act = robot_spec.add_actuator(
        name="actuator1",
        target="joint2",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
    )

    act.set_to_position(100.0, 1.0)

    _ = attach_frame.attach_body(
        robot_spec.body("base_link"),
        prefix="robot_",  # avoids name clashes
        # pos=[0, 0, 0],
    )

    model = check_type(scene_spec.compile(), mujoco.MjModel)  # pyright: ignore[reportAny]

    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            aid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "robot_actuator1",
            )

            # print(f"{aid=}")
            data.ctrl[aid] = -3 / 4
            # model.actuator("robot_actuator1")

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
