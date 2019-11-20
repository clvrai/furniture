# DoorGym Unity Environment
## Overview
This is a modified and extended version of the standard Unity environment for MuJoCo by Emo Todorov
For basic usage information, see the MuJoCo [Unity Plugin](http://mujoco.org/book/unity.html) page
The intended use for this environment is rendering *remote* simulations, specifically for our [DoorGym Environment](https://github.com/PSVL/DoorGym)
Please see the DoorGym repository for an example integration.

## New features
* The MJCF import process is replaced with with a runtime loading mechanism triggerd through the TCP interface.
  * As a consequence, environments can no longer be imported and modified, but they will be merged into existing scenes.
* A best effort is now made to import any lights specified in the MJCF file, but be warned Unity and MuJoCo have different lighting models and not every lighting scenario is supported.
* Materials specified in MJCF files are mapped to Unity materials by name when possible instead of recreating the MJCF material.
* Material colors can be randomized on command via the TCP interface.
* Per-pixel object segmentation data can be output on command via the TCP interface.
* Imported models have their normals recalculated, with seams automatically created with thresholding.
  * This may not be desirable behavior for your specific models, or you may want a different threshold.  Please create an issue if you'd like this behavior to be configurable.

## Configuration
The configuration file is Assets/StreamingAssets/settings.xml
The `<material>` nodes define min and max values for HSV and should be in [0,1].  Materials are instanced per object, which automatically necessitates adding `(Instance)` to the material name.  If you want to add a new material, be sure to use the `material (Instance)` naming convention in the configuration file AND the MJCF file.

`<segment>` nodes define the IDs used when generating object segmentation images. Segments are mapped to MJCF `<body>` by tags (and subsequently, their meshes) name, and will match up the first underscore or end of string. So, `robot`, `robot_shoulder` and `robot_arm` would all be placed in the same segment if there is a `robot` segment specified.  If a body has a name with no matching segment name, it will inherit segmentation from its parent, so you can easily segment an entire heirarchy by properly naming its parent node.

## Acknowledgements
Triplanar mapping code by github user keijiro, used under public domain

Contains normal recalulating code from http://schemingdeveloper.com, used under license

Mouse orbit code from the Unify Community Wiki, used under CC BY-SA 3.0 license

Textures from 3dtextures.me used under CC0 license
