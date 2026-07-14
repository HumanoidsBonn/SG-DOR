# pepper_scene_graphs/models/registry.py

from typing import Callable, Type, Any, Dict

_BACKBONES: Dict[str, Type] = {}


def register_backbone(name: str) -> Callable[[Type], Type]:
  """Decorator to register a backbone model class.

  Args:
      name: The name key to register the backbone under.

  Returns:
      The decorator function.
  """
  def _wrap(cls: Type) -> Type:
    _BACKBONES[name] = cls
    return cls
  return _wrap


def make_backbone(name: str, **kwargs: Any) -> Any:
  """Factory function to instantiate a backbone.

  Args:
      name: The name of the backbone to instantiate.
      **kwargs: Keyword arguments passed to the backbone constructor.

  Returns:
      The instantiated backbone model.

  Raises:
      ValueError: If the backbone name is not registered.
  """
  if name not in _BACKBONES:
    raise ValueError(f"Unknown backbone: {name}. Available: {list(_BACKBONES.keys())}")
  return _BACKBONES[name](**kwargs)
