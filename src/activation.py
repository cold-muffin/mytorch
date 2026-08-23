import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
  @abstractmethod
  def forward(self, inp: np.ndarray) -> np.ndarray:
    ...

  @abstractmethod
  def backward(self, inc: np.ndarray) -> np.ndarray:
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
    self.inp = inp
    return np.maximum(inp, 0)

  def backward(self, inc):
    """
    Returns the derivative of the Rectified Linear Unit function.

    Args:
      inc (ndarray): An ndarray of any shape representing the incoming gradient.
    
    Returns:
      (ndarray): An ndarray of any shape representing the outgoing gradient.
    """
    return np.where(self.inp > 0, inc, 0)