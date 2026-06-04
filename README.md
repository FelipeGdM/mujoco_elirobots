# Mujoco 


```bash
python3 -c "import mujoco; model = mujoco.MjModel.from_xml_path('robot.urdf'); mujoco.mj_saveLastXML('robot.xml', model)"

pybind11-stubgen mujoco -o ~/typings/  # Installs mujoco stubs dir in $HOME/typings/
```
