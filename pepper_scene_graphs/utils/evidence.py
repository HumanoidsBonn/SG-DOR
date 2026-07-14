# pepper_scene_graphs/utils/evidence.py

import torch
import torch.nn.functional as F


from typing import Any, Callable, Type

def relu_evidence(logits: torch.Tensor) -> torch.Tensor:
  """Apply ReLU activation to obtain evidence."""
  return F.relu(logits)


def exp_evidence(logits: torch.Tensor) -> torch.Tensor:
  """Apply exponential activation with clamping to obtain evidence."""
  return torch.exp(torch.clamp(logits, min=-10, max=10))


def softplus_evidence(logits: torch.Tensor) -> torch.Tensor:
  """Apply softplus activation to logits to get non-negative evidence.
  
  Args:
      logits: Input tensor.
      
  Returns:
      Tensor with softplus applied.
  """
  return F.softplus(logits)


def get_evidence_fn(name: str) -> Any:
  """Retrieve the evidence activation function by name.

  Args:
      name: One of 'relu', 'exp', 'softplus', 'none'.

  Returns:
      The activation function or None.

  Raises:
      ValueError: If name is unknown.
  """
  name = (name or "none").lower()
  if name == "relu":
    return relu_evidence
  if name == "exp":
    return exp_evidence
  if name == "softplus":
    return softplus_evidence
  if name == "none":
    return None
  raise ValueError(f"Invalid evidence function: {name}")
