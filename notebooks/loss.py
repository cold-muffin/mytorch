import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
  @abstractmethod
  def forward(self, truth: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    "Calculates the loss given some truth datapoints"
  
  def __call__(self, truth, inputs):
    return self.forward(truth, inputs)

class CrossEntropy(Loss):
  def forward(self, truth, inputs):
    tolerance = 1e-7
    inputs = np.clip(inputs, tolerance, 1-tolerance)
    
    # Sparse
    if len(truth.shape) == 1:
      return -np.log(
        inputs[range(len(truth)), truth]
      )
    # One-hot encoded
    else:
      return -np.log(
        inputs[range(len(truth)), np.argmax(truth, axis=1)]
      )