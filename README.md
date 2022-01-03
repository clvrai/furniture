# IKEA Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks

[Youngwoon Lee](https://youngwoon.github.io), [Edward S. Hu](https://www.edwardshu.com), [Joseph J. Lim](https://clvrai.com) at [USC CLVR lab](https://clvrai.com)<br/>
[[Environment website (https://clvrai.com/furniture)](https://clvrai.com/furniture)]<br/>
[[arXiv Paper](https://arxiv.org/abs/1911.07246)]

|![](docs/img/agents/video_sawyer_swivel_chair.gif)|![](docs/img/agents/video_baxter_chair.gif)|![](docs/img/agents/video_cursor_round_table.gif)|![](docs/img/agents/video_jaco_tvunit.gif)|![](docs/img/agents/video_panda_table.gif)|
| :---: | :---: | :---: |:---: |:---: |
| Sawyer | Baxter | Cursors | Jaco | Panda |


We are announcing the launch of the **IKEA Furniture Assembly environment** as a first-of-its-kind benchmark for testing and accelerating the automation of physical assembly processes.
An agent (Sawyer, Baxter, Jaco, Panda, Fetch) is required to move, align, and connect furniture parts sequentially.
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
- Python 3.9
- Mujoco 2.1.1
- Unity 2018.4.23f1 ([Install using Unity Hub](https://unity3d.com/get-unity/download))

### Installation
```bash
# from github (latest)
pip install git+git//github.com/clvrai/furniture.git

# from pypi
pip install furniture

# from code
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
env = gym.make("IKEASawyer-v0", furniture_name="table_lack_0825")

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

## (4) Benchmarking

We provide example commands for `table_lack_0825`. You can simply change the furniture name to test on other furniture models.
For evaluation, you can add `--is_train False --num_eval 50` to the training command:

### IL Training

### BC
```bash
python -m run --algo bc --run_prefix bc_table_lack_0825 --env IKEASawyerDense-v0 --furniture_name table_lack_0825 --demo_path demos/Sawyer_table_lack_0825
```

### GAIL
```bash
mpirun -np 32 python -m run --algo gail --run_prefix gail_table_lack_0825 --env IKEASawyerDense-v0 --furniture_name table_lack_0825 --demo_path demos/Sawyer_table_lack_0825
```

### GAIL + PPO
```bash
mpirun -np 32 python -m run --algo gail --run_prefix gailppo_table_lack_0825 --env IKEASawyerDense-v0 --furniture_name table_lack_0825 --demo_path demos/Sawyer_table_lack_0825 --gail_env_reward 0.5
```

### RL Training

### SAC
```bash
python -m run --algo sac --run_prefix sac_table_lack_0825 --env IKEASawyerDense-v0 --furniture_name table_dockstra_0279
```

### PPO
```bash
mpirun -np 32 python -m run --algo ppo --run_prefix ppo_table_lack_0825 --env IKEASawyerDense-v0 --furniture_name table_dockstra_0279
```

<br>

## (5) Documentation
See [documentation](docs/readme.md) for installation and configuration details.

<br>

## (6) References
Our Mujoco environment is developed based on Robosuite and Unity implementation from DoorGym-Unity is used.

* Robosuite environment: https://github.com/StanfordVL/robosuite
* MuJoCo-Unity plugin: http://www.mujoco.org/book/unity.html
* DoorGym-Unity: https://github.com/PSVL/DoorGym-Unity

<br>

## (7) Citation
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
We thank [Alex Yin](https://www.linkedin.com/in/alexyin1/) and [Zhengyu Yang](https://zhengyuyang.com) for their contributions. We would like to thank everyone who has helped IKEA Furniture Assembly Environment in any way.
