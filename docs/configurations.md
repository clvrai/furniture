# Configuration

We offer many configuration options on both the MuJoCo simulation and Unity rendering. The simulation configurations will be exposed through command line arguments.

## Robots
To switch between agent configuration, simply select the corresponding python script.
```
# sawyer
$ python -m env.furniture_sawyer ...

# cursor
$ python -m env.furniture_cursor ...
```
OR use the demo_\<yourtask\>.py script for your task
  ```
# manipulating agent manually
$ python -m demo_manual ...

# RL training
$ python -m demo_rl ...
```

## Furniture Models
Preferably use the `--furniture_name` argument to choose a furniture model. `--furniture_id` may also be used but is not recommended, because the ids are determined dynamically by sorting the xml files in the directory. Therefore, if more furniture is added, the IDs may change. Use the `furniture_name` argument to get the exact furniture you want. See [`furniture/env/models/__init__.py`](../env/models/__init__.py) for more details.

Some furniture pieces (e.g. flat plane) are difficult to grasp using grippers we currently support.
This can be addressed by initializing the difficult parts in a predefined, easy to grasp way. See
[Designing a new task](creating_task.md) for how to customize initialization.

## Assembly Configuration
Two parts will be assembled when an agent activates `connect` action and two parts are well-aligned.
The thresholds for determining successful connection are defined by distance between two connectors `pos_dist`, cosine distance between up vectors of connectors `rot_dist_up`, cosine distance between forward vectors of connectors `rot_dist_forward`, and relative pose of two connectors `project_dist`. These values are configurable by changing those values in [`furniture/env/furniture.py`](../env/furniture.py). Please refer to `_is_aligned(connector1, connector2)` method in [`furniture/env/furniture.py`](../env/furniture.py) for details.

## Background Scenes

<img src="img/env/allenv.gif" width="200">

Use `--background` argument to choose a background scene.

- Garage: flat ground environment
- Interior: flat ground, interior decoration
- Lab: flat ground, bright lighting
- Industrial: cluttered objects, directional lighting
- NightTime: flat ground, dim lighting
- Ambient: flat ground, colored lighting

The next update will make lighting and material changes programmatic, so the user does not need to rebuild the binary for Unity changes.
Note that objects in the Unity scene are not physically simulated. For example, the table is just an overlay of the invisible mujoco ground plane.

## Meshes
To use 3d meshes for collision detection, we reference the STL meshes for each part in the assets tag. Then we use mesh geoms for each body to give it a collider shaped like the mesh.
Many times, the mesh collider is not accurate enough for contact physics. So, we use the meshes for just rendering, and add invisible MuJoCo primitive geometries to create the colliders.

## Connectors, Welding
MuJoCo has a *weld* mechanism that can attach bodies together. We define weld constraints between parts that should be attached together in the equality tag.
The connectors on a part is defined through the site tag. There can be multiple connectors for each part. The naming for the connector is special, and should follow a template:
Given two parts A and B, we want to connect A to B:
The site tag for A should be named “A-B,conn_site” and the site tag for B should be “B-A,conn_site”.
Look at the XMLs in [`furniture/env/models/assets/objects`](../env/models/assets/objects) for more examples.
<img src="img/readme/conn_sites.png">
## Example

```xml
<mujoco model="chair_bernhard_0146">
    <asset>
        <mesh file="chair_bernhard_0146/complete_chair.stl" name="chair_complete" scale="0.576 0.576 0.576" />
        <mesh file="chair_bernhard_0146/left_leg.stl" name="1_chair_leg_left" scale="0.576 0.576 0.576" />
        <mesh file="chair_bernhard_0146/right_leg.stl" name="2_chair_leg_right" scale="0.576 0.576 0.576" />
        <mesh file="chair_bernhard_0146/seat.stl" name="3_chair_seat" scale="0.576 0.576 0.576" />
        <texture file="../textures/light-wood.png" name="tex-light-wood" type="2d" />
        <material name="light-wood" reflectance="0.5" texrepeat="20 20" texture="tex-light-wood" texuniform="true" />
    </asset>
    <equality>
      <weld active="false" body1="1_chair_leg_left" body2="3_chair_seat" solimp="1 1 0.5" solref="0.01 0.3" />
      <weld active="false" body1="2_chair_leg_right" body2="3_chair_seat" solimp="1 1 0.5" solref="0.01 0.3" />
    </equality>

    <worldbody>
        <body name="1_chair_leg_left" pos="-0.115 0.0 0.1727" quat="1 0 0 0">
            <geom density="100" material="light-wood" mesh="1_chair_leg_left" name="1_chair_leg_left_geom" pos="0 0 0" rgba="0.82 0.71 0.55 1" solref="0.001 1" type="mesh" />
            <site name="leg_left-seat,conn_site1" pos="0.0 0.0 0.1401" quat="0.707 0 -0.707 0" rgba="1 0 1 0.3" size="0.0057" />
            <site name="1_chair_leg_left_corner_site1" pos="-0.019 0.1305 -0.144" rgba="0 0 1 0.3" size="0.0057" />
            <site name="1_chair_leg_left_corner_site2" pos="-0.019 -0.129 -0.144" rgba="0 0 1 0.3" size="0.0057" />
            <site name="1_chair_leg_left_corner_site3" pos="0.0307 0.0230 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="1_chair_leg_left_corner_site4" pos="0.0307 -0.022 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="1_chair_leg_left_corner_site5" pos="-0.019 0.0230 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="1_chair_leg_left_corner_site6" pos="-0.019 -0.022 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="1_chair_leg_left_bottom_site" pos="-0.019 0.0 -0.134" rgba="0 0 1 0.3" size="0.0057" />
            <site name="1_chair_leg_left_top_site" pos="0.0 0.0 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="1_chair_leg_left_horizontal_radius_site" pos="-0.019 0.0 0.0" rgba="0 0 1 0.3" size="0.0057" />
        </body>
        <body name="2_chair_leg_right" pos="0.1152 0.0 0.1727" quat="1 0 0 0">
            <geom density="100" material="light-wood" mesh="2_chair_leg_right" name="2_chair_leg_right_geom" pos="0 0 0" rgba="0.82 0.71 0.55 1" solref="0.001 1" type="mesh" />
            <site name="leg_right-seat,conn_site2" pos="0.0 0.0 0.1401" quat="0.707 0 -0.707 0" rgba="0 1 1 0.3" size="0.0057" />
            <site name="2_chair_leg_right_corner_site1" pos="0.0192 0.1305 -0.144" rgba="0 0 1 0.3" size="0.0057" />
            <site name="2_chair_leg_right_corner_site2" pos="0.0192 -0.129 -0.144" rgba="0 0 1 0.3" size="0.0057" />
            <site name="2_chair_leg_right_corner_site3" pos="0.0268 0.0230 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="2_chair_leg_right_corner_site4" pos="0.0268 -0.022 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="2_chair_leg_right_corner_site5" pos="-0.028 0.0230 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="2_chair_leg_right_corner_site6" pos="-0.028 -0.022 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="2_chair_leg_right_bottom_site" pos="0.0192 0.0 -0.134" rgba="0 0 1 0.3" size="0.0057" />
            <site name="2_chair_leg_right_top_site" pos="0.0 0.0 0.1382" rgba="0 0 1 0.3" size="0.0057" />
            <site name="2_chair_leg_right_horizontal_radius_site" pos="0.0172 0.0057 0.0" rgba="0 0 1 0.3" size="0.0057" />
        </body>
        <body name="3_chair_seat" pos="0.0 0.0 0.4416" quat="1 0 0 0">
            <geom density="100" material="light-wood" mesh="3_chair_seat" name="3_chair_seat_geom" pos="0 0 0" rgba="0.82 0.71 0.55 1" solref="0.001 1" type="mesh" />
            <site name="seat-leg_left,conn_site1" pos="-0.115 0.0 -0.126" quat="0.707 0 -0.707 0" rgba="1 0 1 0.3" size="0.0057" />
            <site name="seat-leg_right,conn_site2" pos="0.1152 0.0 -0.126" quat="0.707 0 -0.707 0" rgba="0 1 1 0.3" size="0.0057" />

            <site name="3_chair_seat_corner_site1" pos="0.1420 -0.116 0.1267" rgba="0 0 1 0.3" size="0.0057" />
            <site name="3_chair_seat_corner_site2" pos="0.0 -0.116 0.1267" rgba="0 0 1 0.3" size="0.0057" />
            <site name="3_chair_seat_corner_site3" pos="-0.141 -0.116 0.1267" rgba="0 0 1 0.3" size="0.0057" />
            <site name="3_chair_seat_corner_site4" pos="0.1420 -0.116 -0.115" rgba="0 0 1 0.3" size="0.0057" />
            <site name="3_chair_seat_corner_site5" pos="-0.141 -0.116 -0.115" rgba="0 0 1 0.3" size="0.0057" />
            <site name="3_chair_seat_corner_site6" pos="0.1420 0.1267 -0.109" rgba="0 0 1 0.3" size="0.0057" />
            <site name="3_chair_seat_corner_site7" pos="-0.141 0.1267 -0.109" rgba="0 0 1 0.3" size="0.0057" />

            <site name="3_chair_seat_bottom_site" pos="0.0 0.0 -0.115" rgba="0 0 1 0.3" size="0.0057" />
            <site name="3_chair_seat_top_site" pos="0.0 -0.116 0.1267" rgba="0 0 1 0.3" size="0.0057" />
            <site name="3_chair_seat_horizontal_radius_site" pos="0.0057 0.0057 0.0" rgba="0 0 1 0.3" size="0.0057" />
        </body>
    </worldbody>
</mujoco>
```



