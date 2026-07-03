import numpy as np
from abc import ABC, abstractmethod

def accuracy(truth: np.ndarray, inputs: np.ndarray) -> float:
  y_pred = np.argmax(inputs, axis=1)
  # Convert one-hot encoded to sparse
  if len(truth.shape) != 1:
    truth = np.argmax(truth, axis=1)
  return np.mean(truth==y_pred)

class Loss(ABC):
  @abstractmethod
  def forward(self, truth: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    "Calculates the loss given some truth datapoints"
  
  def __call__(self, truth, inputs):
    return self.forward(truth, inputs)

  def calculate(self, truth, inputs):
    loss = self.forward(truth, inputs)
    return np.mean(loss)

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