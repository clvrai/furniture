# IKEA Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks

[Youngwoon Lee](https://youngwoon.github.io), [Edward S. Hu](https://www.edwardshu.com), [Joseph J. Lim](https://clvrai.com) at [USC CLVR lab](https://clvrai.com)<br/>
[[Environment website (https://clvrai.com/furniture)](https://clvrai.com/furniture)]<br/>
[[arXiv Paper](https://arxiv.org/abs/1911.07246)]

|![](docs/img/agents/video_sawyer_swivel_chair.gif)|![](docs/img/agents/video_baxter_chair.gif)|![](docs/img/agents/video_cursor_round_table.gif)|![](docs/img/agents/video_jaco_tvunit.gif)|![](docs/img/agents/video_panda_table.gif)|
| :---: | :---: | :---: |:---: |:---: |
| Sawyer | Baxter | Cursors | Jaco | Panda |


This code contains the **IKEA Furniture Assembly environment** as a first-of-its-kind benchmark for testing and accelerating the automation of physical assembly processes.
An agent (Sawyer, Baxter, Panda, Jaco, Cursor) is required to move, align, and connect furniture parts sequentially.
The task is completed when all parts are connected.


The IKEA Furniture Assembly environment provides:
- Comprehensive modeling of **furniture assembly** task
- 60+ furniture models
- Configurable and randomized backgrounds, lighting, textures
- Realistic robot simulation for Baxter, Sawyer, Jaco, Panda, and more
- Gym interface for easy RL training
- Reinforcement learning and imitation learning benchmarks
- Teleopration with 3D mouse/VR controller

<br>

## Directories
The structure of the repository:
- `docs`: Detail documentation
- `furniture`:
  - `config`: Configuration files for the environments
  - `env`: Envrionment code of the IKEA furniture assembly environment
  - `util`: Utility code
  - `demo_manual.py`: Script for testing the environment with keyboard control
- `furniture-unity`: Unity code for the IKEA furniture assembly environment (excluded in this repo due to the size of files, instead download pre-built Unity app)
- `method`: Reinforcement learning and imitation learning code (will be updated soon)


## (0) Installation

### Prerequisites
- Ubuntu 18.04, MacOS Catalina, Windows10
- Python 3.7 (pybullet may not work with Python 3.8 or higher)
- Mujoco 2.0
- Unity 2018.4.23f1 ([Install using Unity Hub](https://unity3d.com/get-unity/download))

### Installation
```bash
git clone https://github.com/clvrai/furniture.git
cd furniture
pip install -e .
```

See [`docs/installation.md`](docs/installation.md) for more detailed instruction and troubleshooting.<br/>
If you are on a headless server, make sure you run a [virtual display](docs/installation.md#virtual-display-on-headless-machines) and use `--virtual_display` to specify the display number (e.g. :0 or :1).


## (1) Human control
You can use WASDQE keys for moving and IJKLUO keys for rotating an end-effector of an agent. SPACE and ENTER are closing and opening the gripper, respectively. C key will connect two aligned parts.

```bash
python -m furniture.demo_manual
```

## (2) Gym interface
Gym interface for the IKEA Furniture Assembly environment is also provided. The environment parameters, such as furniture, background, and episode length, can be specified via parameters. (see `register` functions in [`furniture/env/__init__.py`](furniture/env/__init__.py).
```py
import gym
import furniture

# make an environment
env = gym.make('IKEASawyer-v0', furniture_name="table_lack_0825")

done = False

# reset environment
observation = env.reset()

while not done:
    # simulate environment
    observation, reward, done, info = env.step(env.action_space.sample())
```

## (3) Demonstration generation
We provide the demonstration generation script for 10 furniture models.
``` bash
python -m furniture.env.furniture_sawyer_gen --furniture_name table_lack_0825 --start_count 0 --n_demos 100
```

## IL Training


### BC
We provide behavioral cloning (BC) benchmark. You can simply change the furniture name to test on other furniture models.

```bash
$ python -m run --algo bc --run_prefix ik_table_dockstra_0279 --env furniture-sawyer-v0 --gpu 0 --max_episode_steps 500 --furniture_name table_dockstra_0279 --demo_path demos/Sawyer_table_dockstra_0279/Sawyer
```

For evaluation, you can add `--is_train False --num_eval 50` to the training command:
```bash
$ python -m run --algo bc --run_prefix ik_table_dockstra_0279 --env furniture-sawyer-v0 --gpu 0 --max_episode_steps 500 --furniture_name table_dockstra_0279 --demo_path demos/Sawyer_table_dockstra_0279/Sawyer --is_train False --num_eval 50
```

## RL Training

### SAC
```
$ python -m run --algo sac --run_prefix sac_table_dockstra_0279 --env furniture-sawyer-v0 --gpu 0 --max_episode_steps 100 --furniture_name table_dockstra_0279
```
For evaluation, you can add `--is_train False --num_eval 50` to the training command.

### PPO
```
$ python -m run --algo ppo --run_prefix ppo_table_dockstra_0279 --env furniture-sawyer-v0 --gpu 0 --max_episode_steps 100 --furniture_name table_dockstra_0279
```
For evaluation, you can add `--is_train False --num_eval 50` to the training command.

<br>

## (2) Documentation
See [documentation](docs/readme.md) for installation and configuration details.

<br>

## (3) References
Our Mujoco environment is developed based on Robosuite and Unity implementation from DoorGym-Unity is used.

* Robosuite environment: https://github.com/StanfordVL/robosuite
* MuJoCo-Unity plugin: http://www.mujoco.org/book/unity.html
* DoorGym-Unity: https://github.com/PSVL/DoorGym-Unity

<br>

## (4) Citation
```
@inproceedings{lee2021ikea,
  title={{IKEA} Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks},
  author={Lee, Youngwoon and Hu, Edward S and Lim, Joseph J},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  url={https://clvrai.com/furniture},
}
```

## Contributors
We thank Alex Yin and Zhengyu Yang for their contributions. We would like to thank everyone who has helped IKEA Furniture Assembly Environment in any way.
