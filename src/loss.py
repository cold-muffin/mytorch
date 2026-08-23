import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
  @abstractmethod
  def forward(self, truth: np.ndarray, pred: np.ndarray) -> float:
    ...

  @abstractmethod
  def backward(self) -> np.ndarray:
    ...

class SquaredError(Loss):
  def forward(self, truth, pred):
    """
    Calculates the squared error given some trurth and predicted values.

    Args:
      truth (ndarray): An ndarray of size (samples, features) representing the ground truth.
      pred (ndarray): An ndarray of size (samples, features) representing the predicted values.

    Returns:
      float: The calculated squared error loss.
    """
    self.truth, self.pred = truth, pred
    return np.sum(np.square(truth - pred))

  def backward(self):
    """
    Returns the derivative of the squared error function.

    Returns:
      ndarray: The outgoing gradient representing the derivative of the squared error function.
    """
    return -2 * (self.truth - self.pred)
