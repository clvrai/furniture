# Triad OpenVR Python Wrapper

This is an enhanced wrapper for the already excellent [pyopenvr library](https://github.com/cmbruns/pyopenvr) by [cmbruns](https://github.com/cmbruns).  The goal of this library is to create easy to use python functions for any SteamVR tracked system.

# Getting Started

```python
import triad_openvr as vr
import pylab as plt
v = vr.triad_openvr()
data = v.devices["controller_1"].sample(1000,250)
plt.plot(data.time,data.x)
plt.title('Controller X Coordinate')
plt.xlabel('Time (seconds)')
plt.ylabel('X Coordinate (meters)')
```

![Example plot of captured data](images/simple_xcoord_plot.png "Example Plot")

# Configuration file

The goal is to identify devices by serial, in order to keep the same name for the same physical device. for maing it work, you just have to change serials and names in the 'config.json' file. Here is an example of config file :

```
{
    "devices":[
        {
          "name": "hmd",
          "type": "HMD",
          "serial":"XXX-XXXXXXXX"
        },
        {
          "name": "tracking_reference_1",
          "type": "Tracking Reference",
          "serial":"LHB-XXXXXXXX"
        },
        {
          "name": "controller_1",
          "type": "Controller",
          "serial":"XXX-XXXXXXXX"
        },
        {
          "name": "tracker_1",
          "type": "Tracker",
          "serial":"LHR-XXXXXXXX"
        }
    ]
}
```
