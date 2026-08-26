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
  def __repr__(self):
    return "ReLU"

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

  def backward(self, inc, learn_rate: float = -1):
    """
    Returns the derivative of the Rectified Linear Unit function.

    Args:
      inc (ndarray): An ndarray of any shape representing the incoming gradient.
      learn_rate (float): The learning rate.
    
    Returns:
      (ndarray): An ndarray of any shape representing the outgoing gradient.
    """
    return np.where(self.inp > 0, inc, 0)

class Sigmoid(Activation):
  def forward(self, inp):
    """
    Performs the Sigmoid activation function.

    Args:
      inc (ndarray): An ndarray of any shape representing the incoming gradient.
    
    Returns:
      (ndarray): An ndarray of any shape representing the outgoing gradient.
    """
    self.inp = inp
    self.out = 1/(1+np.exp(-inp))
    return self.out

  def backward(self, inc, learn_rate: float = -1):
    """
    Returns the derivative of the Sigmoid function.

    Args:
      inc (ndarray): An ndarray of any shape representing the incoming gradient.
      learn_rate (float): The learning rate.
    
    Returns:
      (ndarray): An ndarray of any shape representing the outgoing gradient.
    """
    return inc * self.out * (1 - self.out)