__all__ = [
    'DATASET_REGISTRY',
    'LOSS_REGISTRY',
    'TRAINER_REGISTRY',
    'MODEL_REGISTRY',
    'METRIC_REGISTRY',
]

class Registry:
    def __init__(self, name):
        self._name = name
        self._objs = {}

    def __call__(self, obj):
        name = obj.__name__.lower()
        assert name not in self._objs, f'Object {name} has registered.'
        self._objs[name] = obj
        return obj

    def __contains__(self, name):
        return name.lower() in self._objs

    def __iter__(self):
        return iter(self._objs.items())

    def get(self, name):
        obj = self._objs.get(name.lower())
        assert obj is not None, f'No such obj named {name.lower()}.'
        return obj

    def key(self):
        return self._objs.keys()


DATASET_REGISTRY = Registry('dataset')
LOSS_REGISTRY = Registry('loss')
TRAINER_REGISTRY = Registry('trainer')
MODEL_REGISTRY = Registry('model')
METRIC_REGISTRY = Registry('metric')
