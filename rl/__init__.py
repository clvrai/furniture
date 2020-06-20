import matplotlib

matplotlib.use("Agg")


def get_agent_by_name(algo):
    """
    Returns RL or IL agent.
    """
    if algo == "sac":
        from rl.sac_agent import SACAgent

        return SACAgent
    elif algo == "ppo":
        from rl.ppo_agent import PPOAgent

        return PPOAgent
    elif algo == "ddpg":
        from rl.ddpg_agent import DDPGAgent

        return DDPGAgent
    elif algo == "bc":
        from il.bc_agent import BCAgent

        return BCAgent
    elif algo == "gail":
        from il.gail_agent import GAILAgent

        return GAILAgent
