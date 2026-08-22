import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
  @abstractmethod
  def forward(self, truth: np.ndarray, inp: np.ndarray) -> np.ndarray:
    ...

  @abstractmethod
  def backward(self, inp: np.ndarray) -> np.ndarray:
    ...