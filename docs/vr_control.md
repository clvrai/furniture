# HTC Vive Controller
We offer support for integrating the Vive controller with the IK controller of robots.
See the `run_vr` method in `furniture.py` for more details. The VR tests were conducted on a workstation running Ubuntu 18.04 and SteamVR.


## Installation
Here, we briefly describe how to setup SteamVR and openvr. See this [guide](https://www.cgl.ucsf.edu/chimera/data/linux-vr-oct2018/linuxvr.html) for detail.

1. Install Steam and SteamVR
```bash
$ sudo add-apt-repository multiverse
$ sudo apt install steam
```
Refer this article for more detail: [How to Install STEAM to Play Games on Ubuntu 18.04 LTS](https://linuxhint.com/install_steam_games_ubuntu_1804/).

2. Install 32-bit graphics driver
```bash
$ sudo apt install libnvidia-gl-440:i386
```

3. Run Steam and log into Steam account
```bash
$ steam
```

4. Install SteamVR in Steam, switch to SteamVR beta by right clicking SteamVR within Steam, choosing Properties, then Betas tab.

5. Start SteamVR by pluging HTC Vive headset and turning on hand-controllers. Then, run SteamVR room setup.

6. Add path to DLLs
```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/share/Steam/steamapps/common/SteamVR/bin/linux64
```
