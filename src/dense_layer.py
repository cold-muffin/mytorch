import numpy as np

class DenseLayer:
  """
  A neuron layer where all the neurons are connected to the previous inputs.

  Args:
    features (int): The number of inputs going into the layer.
    neurons (int): The number of outputs from the layer.
  """
  def __init__(self, features: int, neurons: int):
    self.features = features
    self.neurons = neurons

    self._random_weights()
    self._random_biases()

  def _random_weights(self) -> None:
    # w_11, w_12, ...
    # w_21, w_22, ...
    # ...
    self.weights = np.random.rand(self.features, self.neurons)
    self.weights *= np.random.choice([-1, 1], size=(self.features, self.neurons))

  def _random_biases(self) -> None:
    self.biases = np.random.rand(1, self.neurons)
    self.biases *= np.random.choice([-1, 1], size=(1, self.neurons))

  def forward(self, inp: np.ndarray):
    """
    Performs one forward pass on the layer given some input.

    Args:
      inp (ndarray): An ndarray of shape (samples, features) representing the input to the layer.
    
    Returns:
      ndarray: An ndarray of shape (samples, neurons) representing the output to the layer.
    """
    self.inp = inp
    # (samples, features) @ (features, neurons) = (samples, neurons)
    # (samples, neurons) + (1, neurons) = (samples, neurons)
    return inp @ self.weights + self.biases
  
  def backward(self, inc: np.ndarray, learn_rate: float = 0.001) -> np.ndarray:
    """
    Backpropogates on the entire dense layer given some incoming gradient.

    Args:
      inc (ndarray): An ndarray of shape (samples, neurons) representing the incoming gradient.
    
    Returns:
      ndarray: An ndarray of shape (samples, features) representing the outgoing gradient.
    """
    # (incoming gradient) @ (transposed weights) = (outgoing gradient)
    # (samples, neurons)  @ (neurons, features)  = (samples, features)
    outg = inc @ self.weights.T

    # (inputs)            @ (incoming gradient) = (weights gradient)
    # (features, samples) @ (samples, neurons)  = (features, neurons)
    dw = self.inp.T @ inc
    self.weights -= dw * learn_rate

    self.biases -= np.sum(inc, axis=0) * learn_rate

    return outg