^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package jaco_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.0.25 (2016-02-23)
-------------------
* Merge branch 'develop' of https://github.com/RIVeR-Lab/wpi_jaco into develop
* Added initial support for the Jaco2 arm, see readme for details
* Contributors: David Kent

0.0.24 (2015-08-18)
-------------------
* reverted changelog
* changelog updated
* Contributors: Russell Toris

0.0.23 (2015-05-04)
-------------------

0.0.22 (2015-04-22)
-------------------

0.0.21 (2015-04-17)
-------------------

0.0.20 (2015-04-14)
-------------------
* removed .urdf file
* urdf file
* urdf file
* Contributors: Mathijs de Langen

0.0.19 (2015-04-10)
-------------------

0.0.18 (2015-04-03)
-------------------

0.0.17 (2015-03-27)
-------------------

0.0.16 (2015-03-24)
-------------------
* copied old meshes. manual texturing failed on web.
* Minor fixes for better placement of items.
  Also:
  - Actually launches joint controllers in the Gazebo launch
  - Allows the robot to be specified, this allows mounting the Jaco arm on
  a variety of bases using the same gazebo launch.
* Added Gazebo support.
  Added necessary elements for Gazebo simulation:
  - Mass (currently made up)
  - Inertia (currently made up)
  - Joint controllers for the Gazebo robot using ROS Control
  - Transmissions on necessary joints
  - Gripper tag to support grasping (may need tuning)
* Contributors: Alex Henning, Peter

0.0.15 (2015-02-17)
-------------------

0.0.14 (2015-02-06)
-------------------

0.0.13 (2015-02-03)
-------------------
* fixed catkin install bug
* min
* remin
* moved to .jpg
* fixed small ring & minified
* Merge remote-tracking branch 'upstream/develop' into manual-texturing
* finished retexturing
* link 4,5,small ring. swapped colors
* link 2
* finished link 1
* large ring complete
* textred large ring manually
* added materials
* changed to .jpg
* changed to jpegs
* minifinified and fixed textures
* replaced with recolored dae's
* added materials
* Contributors: Peter, Russell Toris

0.0.12 (2015-01-20)
-------------------

0.0.11 (2014-12-18)
-------------------
* minify script
* Contributors: Russell Toris

0.0.10 (2014-12-12)
-------------------

0.0.9 (2014-12-02)
------------------

0.0.8 (2014-10-22)
------------------

0.0.7 (2014-09-19)
------------------

0.0.6 (2014-09-02)
------------------
* start position fixed
* added rviz to install
* fixed start position of arm
* Updated urdf and regenerated urdf file
* rebuild of URDF
* minified XML
* fixed collision models in URDF
* Contributors: Russell Toris, dekent

0.0.5 (2014-08-25)
------------------

0.0.4 (2014-08-05)
------------------
* fixed install cmake bug
* recompiled URDF
* fixed URDF to load minified files
* updated pre-compiled URDF
* minified XML
* removed large unused mesh originals
* removed large unused mesh originals
* Removed soft links
* Using low poly version of all meshes
* Removed duplicate meshes
* Improved meshes and URDF
* Contributors: Russell Toris, Steven Kordell

0.0.3 (2014-08-01)
------------------

0.0.2 (2014-08-01)
------------------
* diff fixed
* Updated xacro files to use jaco_description instead of jaco_model, generated urdf for standalone_arm
* Contributors: Russell Toris, dekent

0.0.1 (2014-07-31)
------------------
* renamed JACO to WPI packages
* Contributors: Russell Toris
