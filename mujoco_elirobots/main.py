import mujoco
import mujoco.viewer
from mujoco._structs import MjModel
from typeguard import check_type

from mujoco_elirobots.builder.model import build_world

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

    model: MjModel = build_world(SCENE_FILENAME, ROBOT_FILENAME)

    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            aid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "robot_actuator2",
            )

            for i in range(len(data.ctrl)):
                data.ctrl[i] = 0

            data.ctrl[aid] = -3 / 4

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
