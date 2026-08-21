import numpy as np
from typing import Optional

class DenseLayer:
  def __init__(self, num_features: int, num_neurons: int):
    self._num_neurons = num_neurons
    self._num_features = num_features

    self._random_weights()
    self._random_biases()
  
  @property
  def num_neurons(self) -> int:
    return self._num_neurons
  
  @property
  def num_features(self) -> int:
    return self._num_features
  
  def _random_weights(self) -> None:
    self.weights = np.random.rand(self.num_features, self.num_neurons)
  
  def _random_biases(self) -> None:
    self.biases = np.random.rand(1, self.num_neurons)

  def forward(self, samples: np.ndarray) -> np.ndarray:
    self.inp = samples
    return samples @ self.weights + self.biases
  
  def backprop(self, gradient: Optional[np.ndarray] = None, stepsize: float = 0.001) -> np.ndarray:
    if gradient is None:
      return np.ones((len(self.inp), self.num_features))

    outp = gradient @ self.weights.T
    self.weights -= self.inp.T @ gradient * stepsize
    self.biases -= np.sum(gradient, axis=0, keepdims=True) * stepsize
    
    return outp