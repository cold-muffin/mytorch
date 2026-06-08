from abc import ABC, abstractmethod
from typing import Iterable

from .observer import Observer

class Observable(ABC):
  "An object that can update an `Iterable` of observers"
  observers: Iterable[Observer]

  @abstractmethod
  def update(self, *args, **kwargs):
    "Performs the `update` method on certain/all observers"