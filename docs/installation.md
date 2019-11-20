# Installation

## Prerequisites
* Ubuntu 18.04 or macOS
* Python 3.6
* Mujoco 2.0
* Unity 2018.3.14f1  (x86_64)

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

2. Install python dependencies
```bash
# Run the next line for Ubuntu
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libopenmpi-dev libglew-dev python3-pip python3-numpy python3-scipy

# Run the rest for both Ubuntu and macOS
pip install -r requirements.txt

# for RL
$ pip install torch torchvision h5py wandb
```

3. Download MuJoCo-Unity binary
Download pre-compiled Unity binary from [this link](https://drive.google.com/open?id=1ofnw_zid9zlfkjBLY_gl-CozwLUco2ib) and extract files to `furniture` directory.
```
# inside the furniture directory
unzip binary.zip
```
Inside `furniture/binary` there should be `Furniture.app` for macOS and `Furniture.x86_64` and `Furniture_Data` folder for Ubuntu.

4. Virtual screen (on headless machines)

On servers, you don’t have a monitor. Use this to get a virtual monitor for rendering. Set the `--virtual_display` flag to
`True` when you run the environment.
```bash
sudo apt-get install xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev
# configure nvidia-x
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
# use virtual display flag when running FurnitureEnvironment
python -m demo_manual --unity True --virtual_display True
```

