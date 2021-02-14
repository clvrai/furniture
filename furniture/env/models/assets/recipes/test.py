import yaml


class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u"tag:yaml.org,2002:python/tuple", PrettySafeLoader.construct_python_tuple
)

with open("env/models/assets/recipes/toy_table.yaml", "r") as stream:
    p = yaml.load(stream, Loader=PrettySafeLoader)

print(p)
