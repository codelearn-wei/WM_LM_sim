from typing import Dict, Type, Any, Optional
from functools import wraps

class Registry:
    """A registry to map strings to classes."""
    
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key: str):
        return key in self._module_dict

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, items={list(self._module_dict.keys())})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key: str) -> Optional[Type]:
        """Get the registry record."""
        return self._module_dict.get(key, None)

    def register_module(self, cls: Type = None, module_name: str = None):
        """Register a module."""
        if cls is None:
            return lambda x: self.register_module(x, module_name)
        
        if module_name is None:
            module_name = cls.__name__
        
        if module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self.name}')
        
        self._module_dict[module_name] = cls
        return cls

    def build(self, *args, **kwargs):
        """Build a module from registry."""
        if not isinstance(args[0], str):
            raise TypeError(f'Expected str, but got {type(args[0])}')
        
        module_name = args[0]
        if module_name not in self._module_dict:
            raise KeyError(f'{module_name} is not registered in {self.name}')
        
        module_cls = self._module_dict[module_name]
        return module_cls(*args[1:], **kwargs)

# Create registries for different components
MODELS = Registry('models')
DATASETS = Registry('datasets')
LOSSES = Registry('losses')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers') 