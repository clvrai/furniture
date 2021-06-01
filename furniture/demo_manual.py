"""
Human control for the IKEA furniture assembly environment.
The user will learn how to configure the environment and control it.
If you're coming from the demo_demonstration script:
Pass --record_demo True and press Y to save the the current scene
into demos/test.pkl.
"""

import gym

import furniture
from furniture import agent_names, background_names, furniture_names
from furniture.util import str2bool

# available agents
agent_names

# available furnitures
furniture_names

# available background scenes
background_names


def main():
    """
    Inputs types of agent, furniture model, and background and simulates the environment.
    """
    print("IKEA Furniture Assembly Environment!")

    # choose an agent
    print()
    print("Supported robots:\n")
    for i, agent in enumerate(agent_names):
        print("{}: {}".format(i, agent))
    print()
    try:
        s = input(
            "Choose an agent (enter a number from 0 to {}): ".format(
                len(agent_names) - 1
            )
        )
        k = int(s)
        agent_name = agent_names[k]
    except:
        print("Input is not valid. Use 0 by default.")
        agent_name = agent_names[0]

    # choose a furniture model
    print()
    print("Supported furniture:\n")
    for i, furniture_name in enumerate(furniture_names):
        print("{}: {}".format(i, furniture_name))
    print()
    try:
        s = input(
            "Choose a furniture model (enter a number from 0 to {}): ".format(
                len(furniture_names) - 1
            )
        )
        furniture_id = int(s)
        furniture_name = furniture_names[furniture_id]
    except:
        print("Input is not valid. Use 'three_blocks' by default.")
        furniture_id = 57
        furniture_name = "three_blocks"

    # choose a background scene
    print()
    print("Supported backgrounds:\n")
    for i, background in enumerate(background_names):
        print("{}: {}".format(i, background))
    print()
    try:
        s = input(
            "Choose a background (enter a number from 0 to {}): ".format(
                len(background_names) - 1
            )
        )
        k = int(s)
        background_name = background_names[k]
    except:
        print("Input is not valid. Use 0 by default.")
        background_name = background_names[0]

    # set correct environment name based on agent_name
    env_name = "IKEA{}-v0".format(agent_name)

    print()
    print(
        "Creating environment (robot: {}, furniture: {}, background: {})".format(
            env_name, furniture_name, background_name
        )
    )

    # make environment following arguments
    env = gym.make(env_name, furniture_name=furniture_name, background=background_name)

    # print brief instruction
    print()
    print("=" * 80)
    print("Instruction:\n")
    print(
        "Move - WASDQE, Rotate - IJKLUO\n"
        "Grasp - SPACE, Release - ENTER (RETURN), Attach - C\n"
        "Switch baxter arms or cursors - 1 or 2\n"
        "Screenshot - T, Video recording - R, Save Demo - Y"
    )
    print("=" * 80)
    print()

    # manual control of agent using keyboard
    env.run_manual()
    # close the environment instance
    env.close()


if __name__ == "__main__":
    main()
