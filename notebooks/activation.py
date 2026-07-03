import math
import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
  @abstractmethod
  def forward(self, inputs: np.ndarray) -> np.ndarray:
    "Calculates activation function of an input"

  def __call__(self, inputs: np.ndarray):
    return self.forward(inputs)

class ReLU(Activation):
  def forward(self, inputs):
    return np.maximum(0, inputs)

class SoftMax(Activation):
  def forward(self, inputs):
    expo = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    norm = expo / np.sum(expo, axis=1, keepdims=True)
    return norm