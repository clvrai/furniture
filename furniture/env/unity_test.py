# ml-agents-envs/mlagents_envs/environment.py

from pathlib import Path
import os
from sys import platform

from mlagents_envs.environment import UnityEnvironment


def _find_unity_path():
    """Finds path to Unity app."""
    binary_dir = Path(__file__).parents[2] / "binary"
    launch_string = None
    if platform == "linux" or platform == "linux2":
        candidates = list(binary_dir.glob("Furniture.x86_64"))
        if len(candidates) == 0:
            candidates = list(binary_dir.glob("Furniture.x86"))
        if len(candidates) == 0:
            candidates = list(Path(".").glob("Furniture.x86_64"))
        if len(candidates) == 0:
            candidates = list(Path(".").glob("Furniture.x86"))

    elif platform == "darwin":
        app_path = "Furniture.app/Contents/MacOS"
        candidates = list((binary_dir / app_path).glob("Furniture"))
        if len(list(candidates)) == 0:
            candidates = list(Path(app_path).glob("Furniture"))

    elif platform == "win32":
        candidates = list(binary_dir.glob("Furniture.exe"))

    if len(candidates) > 0:
        launch_string = str(candidates[0])
    return launch_string


# launch_string = _find_unity_path()
launch_string = "D:\\Documents\\USC\\Furniture\\furniture\\furniture-unity-main\\furniture-unity-main\\build\\Furniture.exe"
envs = UnityEnvironment(file_name=None, base_port=5004, seed=1, side_channels=[])

envs.reset()
behavior_names = env.behavior_specs.keys()
print(behavior_names)
