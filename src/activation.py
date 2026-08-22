import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
  @abstractmethod
  def forward(self, inp: np.ndarray) -> np.ndarray:
    ...

  @abstractmethod
  def backward(self, inp: np.ndarray) -> np.ndarray:
    ...

class ReLU(Activation):
  def forward(self, inp):
    """
    Performs the Rectified Linear Unit function on some outputs.

    Args:
      inp (ndarray): An ndarray of shape (samples, features).
    
    Returns:
      (ndarray): An ndarray of shape (samples, features).
    """
    return np.maximum(inp, 0)

  def backward(self, inp) -> np.ndarray:
    ...