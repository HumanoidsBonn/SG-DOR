# pepper_scene_graphs/utils/seed.py

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
  """Seeding for reproducibility.
  
  Args:
      seed: Integer seed for random, numpy, and torch.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
