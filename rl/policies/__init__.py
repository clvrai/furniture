def get_actor_critic_by_name(name, algo):
    actor = critic = None
    if name == "mlp":
        from .mlp_actor_critic import MlpActor, MlpCritic, NoisyMlpActor

        if algo == "ddpg":  # add exploratory noise to actor
            actor = NoisyMlpActor
        elif algo in ["bc"]:
            return MlpActor, None
        else:
            actor = MlpActor
        return actor, MlpCritic
    else:
        raise ValueError("--policy %s is not supported." % name)
