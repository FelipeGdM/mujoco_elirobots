import mujoco
import mujoco.viewer
from typeguard import check_type

SCENE_FILENAME = "mujoco_elirobots/scene.xml"
ROBOT_FILENAME = "mujoco_elirobots/assets/ec63/ec63_description.urdf"

def main():

    # 1. Load the scene and robot as Spec objects
    scene_spec = mujoco.MjSpec.from_file(SCENE_FILENAME)
    robot_spec = mujoco.MjSpec.from_file(ROBOT_FILENAME)

    attach_frame = scene_spec.worldbody.add_frame(
        pos=[0, 0, 0],          # world position
    )

    _ = attach_frame.attach_body(
        robot_spec.body("base_link"),
        prefix="robot_",                # avoids name clashes
        # pos=[0, 0, 0], 
    )

    model = check_type(scene_spec.compile(), mujoco.MjModel)  # pyright: ignore[reportAny]
    
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()

