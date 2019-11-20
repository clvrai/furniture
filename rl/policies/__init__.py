
def get_actor_critic_by_name(name):
    if name == 'mlp':
        from .mlp_actor_critic import MlpActor, MlpCritic
        return MlpActor, MlpCritic
    else:
        raise NotImplementedError()

