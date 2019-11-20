# Test installation

## Test MuJoCo-Unity
First, download pre-compiled Unity binary from [this link](https://drive.google.com/open?id=1ofnw_zid9zlfkjBLY_gl-CozwLUco2ib) if you haven't already and extract files to `furniture` directory.

```bash
$ python -m demo_manual --unity True
```
Now that we’ve installed the dependencies, we can first see if the environment is working. A window should pop up, and you can use the WASDQE keys to move the robot arm around. Refer to the documentation for detailed controls. You can skip the next two sections if you don't care about testing the Unity editor or the MuJoCo viewer.


## Test MuJoCo-Unity Editor
For developmental purposes, we recommend installing the Unity Editor to visualize and inspect the Unity scene during runtime.
Install [Unity Hub](https://unity3d.com/get-unity/download), a software for managing Unity versions, sign up for an account, and then install the __2018.3.14f1__ version.
1. Open the Unity project `furniture-unity` under this repository.
2. Double click `All` scene in the `Scenes` directory in the Project view.
3. Click the play button in the Unity editor.
4. Run the python command with `unity` and `unity_editor` set to `True`.
```bash
$ python -m demo --unity True --unity_editor True
```
5. In the Game view of the Unity Editor, try pressing WASDQE to move around.
