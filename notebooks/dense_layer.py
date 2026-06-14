import numpy as np

class DenseLayer:
  def __init__(self, num_neurons: int, num_features: int):
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
    self.weights = np.random.rand(self.num_neurons, self.num_features)
  
  def _random_biases(self) -> None:
    self.biases = np.random.rand(1, self.num_neurons)

  def forward(self, samples: np.ndarray) -> np.ndarray:
    return samples @ self.weights.T + self.biases