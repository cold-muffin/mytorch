import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
  def __init__(self, features: int, neurons: int):
    self.features = features
    self.neurons = neurons
  
  @abstractmethod
  def forward(self, inp: np.ndarray) -> np.ndarray:
    """
    Args:
      inp (ndarray): An ndarray of shape (samples, features) representing the input to the layer.
    
    Returns:
      ndarray: An ndarray of shape (samples, neurons) representing the output to the layer.
    """
    ...
  
  @abstractmethod
  def backward(self, inc: np.ndarray, learn_rate: float) -> np.ndarray:
    """
    Args:
      inc (ndarray): An ndarray of shape (samples, neurons) representing the incoming gradient.
    
    Returns:
      ndarray: An ndarray of shape (samples, features) representing the outgoing gradient.
    """
    ...

  def __repr__(self):
    return f"DenseLayer: {self.features} => {self.neurons}"

  def _random_weights(self) -> None:
    # w_11, w_12, ...
    # w_21, w_22, ...
    # ...
    self.weights = np.random.rand(self.features, self.neurons)
    self.weights *= np.random.choice([-1, 1], size=(self.features, self.neurons))

  def _random_biases(self) -> None:
    self.biases = np.random.rand(1, self.neurons)
    self.biases *= np.random.choice([-1, 1], size=(1, self.neurons))

class DenseLayer(Layer):
  """
  A neuron layer where all the neurons are connected to the previous inputs.

  Args:
    features (int): The number of inputs going into the layer.
    neurons (int): The number of outputs from the layer.
  """
  def __init__(self, features: int, neurons: int):
    super().__init__(features, neurons)
    self._random_weights()
    self._random_biases()

  def forward(self, inp):
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
  
  def backward(self, inc, learn_rate = 0.001):
    """
    Backpropogates on the entire dense layer given some incoming gradient.

    Args:
      inc (ndarray): An ndarray of shape (samples, neurons) representing the incoming gradient.
      learn_rate (float): A float representing the learning rate of the network.
    
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

class DropoutLayer(Layer):
  """
  A neuron layer where a probability of neurons are "dropped", resulting in some neurons losing connections with others.

  Args:
    features (int): The number of inputs going into the layer.
    neurons (int): The number of outputs from the layer.
    keep (float): A float [0, 1] representing the percentage of neurons to keep.
  """
  def __init__(self, features: int, neurons: int, keep: float):
    super().__init__(features, neurons)
    self.keep = keep
    self.training = True
    self._random_weights()
    self._random_biases()
  
  def forward(self, inp):
    """
    Performs one forward pass on the layer where `keep` percent of outputs become 0.

    Args:
      inp (ndarray): An ndarray of shape (samples, features) representing the input to the layer.
    
    Returns:
      ndarray: An ndarray of shape (samples, neurons) representing the output to the layer.
    """
    self.inp = inp
    outp = (inp @ self.weights + self.biases)
    if self.training:
      self.drop = np.random.binomial(1, self.keep, size=outp.shape) / self.keep
      outp *= self.drop
    return outp

  def backward(self, inc, learn_rate = 0.001):
    """
    Backpropogates on the entire dense layer given some incoming gradient.

    Args:
      inc (ndarray): An ndarray of shape (samples, neurons) representing the incoming gradient.
      learn_rate (float): A float representing the learning rate of the network.
    
    Returns:
      ndarray: An ndarray of shape (samples, features) representing the outgoing gradient.
    """
    inc *= self.drop

    outg = inc @ self.weights.T

    dw = self.inp.T @ inc
    self.weights -= dw * learn_rate

    self.biases -= np.sum(inc, axis=0) * learn_rate

    return outg