def get_actor_critic_by_name(name, algo):
    actor = critic = None
    if name == 'mlp':
        from .mlp_actor_critic import MlpActor, MlpCritic, NoisyMlpActor
        if algo == 'ddpg': # add exploratory noise to actor
            actor = NoisyMlpActor
        else:
            actor = MlpActor
        return actor, MlpCritic
    else:
        raise NotImplementedError()
