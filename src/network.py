import numpy as np
from typing import Sequence

from .layer import DenseLayer, DropoutLayer
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
      keep = 1 if neurons[i] < 6 else .7
      network.append(DropoutLayer(neurons[i], neurons[i+1], keep))
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
  
  def cost(self, y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Performs a loss calculation given the output of the network.

    Args:
      y (ndarray): The ground truth of which we compare the predicted values to.
      y_pred (ndarray): The predicted output from the neural network which we compare with the ground truth.
    
    Returns:
      float: A single value which represents the loss of the prediction.
    """
    return self.loss.forward(y, y_pred)
  
  def backprop(self, learn_rate: float = 0.001) -> None:
    """
    Performs a backward pass on the entire network.

    Precondition:
      A single forward pass and cost must have been calculated at least once.

    Args:
      learn_rate (float): The learning rate of the network.
    """
    grad = self.loss.backward()
    for layer in self.network[::-1]:
      grad = layer.backward(grad, learn_rate)
  
  def train(
    self, 
    x_train: np.ndarray, 
    y_train: np.ndarray, 
    epochs: int, 
    learn_rate: float = 0.001
  ) -> tuple[Sequence[float], Sequence[float]]:
    """
    Trains the network by the number of epochs.

    Args:
      x_train (ndarray): The training data to train the model on.
      epochs (int): The number of training steps to put the model through.
      learn_rate (float): The learning rate of the model.
    """
    costs = []
    learn_rates = []
    for epoch in range(epochs):
      y_pred = self.forward(x_train)
      costs.append(self.cost(y_train, y_pred))
      learn_rates.append(learn_rate)
      self.backprop(learn_rate)
    return costs, learn_rates