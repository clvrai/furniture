# Converting a New Furniture Model

The IKEA Furniture Assembly Environment implements a furniture model using MuJoCo XML schema and STL meshes for furniture pieces.
To create a new furniture model for the environment, a STL mesh for each furniture piece is required.  
The general steps to adding new furniture are:
1. Obtain 3D model (mesh) of furniture
1. Generate a mujoco XML with [welds](furniture_details.md#Connectors-Welding)    

## 3D models
### Creating 3D Models
Preferably, a desired furniture model will be designed using 3D modeling software like Rhino or Blender such that  
1. All individual furniture parts can be saved as individual .stl files.
1. All parts have the same relative positions and sizes as the real life furniture. 
1. The file size (polygon count) of the meshes is as small as possible without significant quality loss 
1. Screws, pegs, and all pin-like small parts are removed from the model.
1. Convex design is used as much as possible.
### Convex Mesh 
The preference for convex meshes comes from the MuJoCo's ability to infer only a convex mesh's intertial properties and from collision physics being more accurate and reliable with convex meshes.  
### Individual Parts
furniture parts must be individual files so that the XML schema can function properly.   
### Quality and Performance
## Creating mujoco XML


