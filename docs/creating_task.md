# Designing a new task / env
This tutorial is aimed at teaching you how to define tasks in the
environment or extending the environment itself. We usually inherit from
`FurnitureEnv`, `FurnitureBaxterEnv`, `FurnitureSawyerEnv`, or `FurnitureCursorEnv`
and override certain functions to define the new task or environment.


## `__init__`
In the constructor, we usually define task / environment specific parameters, such as
reward function parameters in the `_env_config` dictionary to avoid polluting the
global configuration dictionary. We recommend either changing the env parameters directly
in the constructor, or reading from a file to load the env config.

## `_step`
The `_step` function takes in an action, and outputs 4 items.

1. next state
2. reward of taking current action
3. episode termination status
4. environment information

If you look at the step function in `FurnitureEnv`, it will first
calculate the change in state, then compute the reward, then log information,
and finally return the 4 items.

## `_reset`
This function resets the robot and furniture to a starting configuration.
Usually you will override the `_place_objects` function to define how the
furniture parts are initialized.

## `_get_obs`
This function returns the observations seen by the agent.

## `_place_objects`
This function by default will attempt to initialize the furniture pieces
in a random position and orientation without collision. You should override
this for your own task if you want to control the furniture initialization.


## `_compute_reward`
This function is called by the `_step` function to compute the reward at
the current state and action. By default it is a sparse reward that depends
on the number of connected parts.


# Block Picking with Baxter
We will look at [`furniture/env/furniture_baxter_block.py`](../env/furniture_baxter_block.py) as a case study. In this file,
want to teach the Baxter agent how to pick up a block and move it towards a target.

We extend the `FurnitureBaxterEnv` to add block picking logic.
##`__init__`
Here, we define all of the dense reward parameters in the `_env_config` dictionary.

## `_step`
The `_step` function is quite standard, computing the reward given the action and
logging info. We zero out the left arm to make the task easier. It calls the `_compute_reward`.

## `_reset`
This function resets the robot and furniture to a starting configuration. It calls the
`_place_objects` function in the `super._reset` call, which we override.

## `_place_objects`
This function overrides `FurnitureEnv` by fixing the initial poses of the furniture parts.
The original logic attempts to find a random configuration of poses for the parts, which
can make RL very slow to learn.

## `_compute_reward`
The dense reward for picking up is structured in the following phases:

0. Put the arm above the block
1. Lower the arm slightly
2. Lower the arm s.t. the block is between fingers
3. Lower the arm more
4. Hold the block
5. Pick up the block
6. Move the block towards the target position
