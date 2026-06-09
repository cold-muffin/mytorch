from abc import ABC, abstractmethod

class Observer(ABC):
  "Performs updates called by an `Observable` object"
  @abstractmethod
  def update(self, *args, **kwargs):
    "Accepts an objects as arguments and performs an update upon the `Observable`'s call"