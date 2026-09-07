from typing import Any, cast

import jax
import mediapy as media
import mujoco
import mujoco.viewer
from mujoco import mjx
from mujoco._structs import MjModel
from tqdm import trange
from typeguard import check_type

from mujoco_elirobots.builder.model import build_world
from mujoco_elirobots.builder.task import build_task

SCENE_FILENAME = "mujoco_elirobots/scene.xml"
ROBOT_FILENAME = "mujoco_elirobots/assets/ec63/ec63_description.urdf"

n_frames = 180
height = 240
width = 320
frames = []
fps = 60.0
times = []
sensordata = []

jit_step = jax.jit(mjx.step)


def main():

    # model: MjModel = build_world(SCENE_FILENAME, ROBOT_FILENAME)
    model: MjModel = build_task(ROBOT_FILENAME)

    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

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


def mj_main():

    framerate = 50

    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # model: MjModel = build_world(SCENE_FILENAME, ROBOT_FILENAME)
    mj_model: MjModel = build_task(ROBOT_FILENAME)

    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mj_model.opt.timestep = 1e-3

    total_timesteps = 1

    mj_data = mujoco.MjData(mj_model)

    frames: list[Any] = []
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    for _ in trange(total_timesteps):
        aid = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "robot_actuator2",
        )

        for i in range(len(mj_data.ctrl)):
            mj_data.ctrl[i] = 0

        mj_data.ctrl[aid] = -3 / 4

        mujoco.mj_step(mj_model, mj_data)

        renderer.update_scene(
            data=mj_data,
            camera="render_camera",
            scene_option=scene_option,
        )
        pixels = renderer.render()
        frames.append(pixels)

    _ = media.set_show_save_dir("/home/felipe/Documents/Poli/RL/agent/mujoco_elirobots")
    _ = media.show_video(frames, title="robot_run", fps=framerate)
    # media.write_video("robot_run.mp4", frames, fps=framerate)
    media.write_image("render.png", image=frames[0])


def mjx_main():

    framerate = 50

    # enable joint visualization option:
    scene_option = mujoco.MjvOption()
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # model: MjModel = build_world(SCENE_FILENAME, ROBOT_FILENAME)
    mj_model: MjModel = build_task(ROBOT_FILENAME)

    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mj_model.opt.timestep = 1e-3

    total_timesteps = 50

    mj_data = mujoco.MjData(mj_model)

    mjx_model = mjx.put_model(mj_model)

    mjx_data = cast(Any, mjx.put_data(mj_model, mj_data))  # pyright: ignore[reportAttributeAccessIssue]

    print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())  # pyright: ignore[reportUnknownArgumentType]

    frames: list[Any] = []
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    for _ in trange(total_timesteps):
        mjx_data = jit_step(mjx_model, mjx_data)  # pyright: ignore[reportUnknownVariableType]
        mj_data = cast(Any, mjx.get_data(mj_model, mjx_data))  # pyright: ignore[reportAttributeAccessIssue]
        renderer.update_scene(
            data=mj_data,
            camera="render_camera",
            scene_option=scene_option,
        )
        pixels = renderer.render()
        frames.append(pixels)

    _ = media.set_show_save_dir("/home/felipe/Documents/Poli/RL/agent/mujoco_elirobots")
    _ = media.show_video(frames, title="robot_run", fps=framerate)
    media.write_video("robot_run.mp4", frames, fps=framerate)
    # media.write_image("render.png", image=frames[0])


if __name__ == "__main__":
    mjx_main()
    # mj_main()
    # main()
