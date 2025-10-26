import inspect
from abc import ABC, abstractmethod

class PickleCompatible(ABC):
    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._bo_storage = {}
        if hasattr(cls, '_big_objects'):
            for field in cls._big_objects:
                def make_prop(name):
                    return property(lambda self, name=name: type(self).lazy_load(name))
                setattr(cls, field, make_prop(field))

    def __setstate__(self, state):
        self.__dict__.update(state)
        type(self)._bo_storage = {}
        sig = inspect.signature(self.__init__)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            if not hasattr(self, param_name) and param.default != inspect.Parameter.empty:
                setattr(self, param_name, param.default)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in list(type(self)._bo_storage.keys()):
            state[key] = None
        return state

    @classmethod
    @abstractmethod
    def load_big_object(cls, name):
        pass

    @classmethod
    def lazy_load(cls, name):
        if cls._bo_storage.get(name, None) is None:
            cls._bo_storage[name] = cls.load_big_object(name)
        return cls._bo_storage[name]
