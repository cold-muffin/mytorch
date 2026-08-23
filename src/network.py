import numpy as np
from typing import Sequence

from .dense_layer import DenseLayer
from .activation import Activation, ReLU
from .loss import Loss, SquaredError

class Network:
  def __init__(
    self, 
    *neurons: int, 
    activation: type[Activation] = ReLU, 
    loss: type[Loss] = SquaredError
  ):
    network = []
    for i in range(len(neurons)-1):
      network.append(DenseLayer(neurons[i], neurons[i+1]))
      if i != len(neurons) - 2:
        network.append(activation())

    self.network = network
    self.loss = loss()

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
    Performs a forward pass on the entire network given some input data.

    Args:
      x (ndarray): An ndarray of shape (samples, features) representing the input data.
    
    Returns:
      ndarray: An ndarray of shape (samples, outp_neurons) representing the outputted prediction.
    """
    for layer in self.network:
      x = layer.forward(x)
    return x
  
  def cost(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Performs a loss calculation given the output of the network.

    Args:
      y_test (ndarray): The ground truth of which we compare the predicted values to.
      y_pred (ndarray): The predicted output from the neural network which we compare with the ground truth.
    
    Returns:
      float: A single value which represents the loss of the prediction.
    """
    return self.loss.forward(y_test, y_pred)
  
  def backprop(self, learn_rate: float = 0.001) -> None:
    """
    Performs a backward pass on the entire network.

    Precondition:
      A single forward pass and cost must have been calculated at least once.
    """
    grad = self.loss.backward()
    for layer in self.network[::-1]:
      grad = layer.backward(grad, learn_rate)
  
  def train(self, epochs: int) -> None:
    ...