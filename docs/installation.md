# Installation

## Prerequisites
* Ubuntu 18.04 or macOS
- Python 3.7 (pybullet may not work with Python 3.8 or higher)
* Mujoco 2.0
- Unity 2018.4.23f1 ([Install using Unity Hub](https://unity3d.com/get-unity/download))

## Installation

1. Install mujoco 2.0 and add the following environment variables into `~/.bashrc` or `~/.zshrc`
```bash
# download mujoco 2.0
wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip
unzip mujoco.zip -d ~/.mujoco
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200

# copy mujoco license key `mjkey.txt` to `~/.mujoco`

# add mujoco to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

# for GPU rendering (replace 418 with your nvidia driver version)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418

# only for a headless server
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-418/libGL.so
```

For macOS Catalina, you first have to make `libmujoco200.dylib` and `libglfw.dylib` in `~/.mujoco/mujoco200/bin` executable. Otherwise, the files cannot be opened because they are from an unidentified developer. To resolve this issue, navigate to the directory `~/.mujoco/mujoco200/bin`, right click each file, click `open` in the menu, and click the `open` button.

2. Install python dependencies
```bash
# Run the next line for Ubuntu
sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libopenmpi-dev libglew-dev python3-pip python3-numpy python3-scipy

# Run the next line for macOS
brew install gcc
brew install openmpi

# Run the rest for both Ubuntu and macOS
pip install -r requirements.txt
```

3. Download MuJoCo-Unity binary
The Unity binary will be automatically downloaded if the Unity binary is found in `furniture/binary` directory.
You can also manually download pre-compiled Unity binary for your OS from [this link](https://drive.google.com/drive/folders/1w0RHRYNG8P5nIDXq0Ko5ZshQ2EYS47Zc?usp=sharing) and extract files to `furniture` directory.
```bash
# inside the furniture directory
unzip [os]_binary.zip
```
Inside `furniture/binary` there should be `Furniture.app` for macOS, and `Furniture.x86_64, Furniture_Data` for Ubuntu, and `Furniture.exe, Furniture_Data` for Windows.

4. Download demonstrations for imitation learning
Download generated demonstrations `demos.zip` from [this link](https://drive.google.com/drive/folders/1w0RHRYNG8P5nIDXq0Ko5ZshQ2EYS47Zc?usp=sharing) and extract files to `furniture` directory.
The demonstration pickle files can be found in `furniture/demos/Sawyer_[furniture name]/`.
The following python script downloads and unzip the demonstrations.
```bash
python scripts/download_demos.py
```

5. Use virtual display for headless servers (optional)
On servers, you don’t have a monitor. Use this to get a virtual monitor for rendering. Set the `--virtual_display` flag to
`:1` when you run the environment.
```bash
# Run the next line for Ubuntu
sudo apt-get install xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev

# Configure nvidia-x
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

# Launch a virtual display
sudo /usr/bin/X :1 &

# Set virtual display flag
python -m furniture.demo_manual --virtual_display :1
```
