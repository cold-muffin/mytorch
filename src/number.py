from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING
import uuid

from src.models.observable import Observable
from src.models.node import Node
from src.operations import Operation, Add, Subtract, Multiply, Divide, Power

if TYPE_CHECKING:
  from src.datatypes import Scalar

class Number(Observable):
    observers = []
    def __repr__(self):
        return "Number({})".format(self.data)

    def __init__(self, obj: Scalar, *, creator: Optional[Operation] = None):
        """ Parameters
            ----------
            obj : Union[int, float, Number, NumPy-numeric]
                The numerical object used as the value of this Number
            
            creator : Optional[Operation]
                The Operation-instance that produced this Number. By specifying a `creator`,
                you are effectively setting the edge from an Operation node to this Number node,
                in the computational graph being created. This allows the back-propagation process
                to 'retrace' back through the graph.
                
                Note: creator must be specified as a named variable: i.e. Number(2, creator=ref)"""
        assert isinstance(obj, (Number, int, float, np.generic))
        self.data = obj.data if isinstance(obj, Number) else obj
        self._creator = creator
        self.grad = None

    @property
    def creator(self):
        """ Number.creator is a read-only property """
        return self._creator
    
    def _op(self, Op, a, b):
        """_op "wraps" (i.e. mediates) all of the operations performed between `Number` instances.
           
           Parameters
           ----------
           Op : subclass of Operation class. E.g. Add or Multiply
                
           a : Union[int, float, Number]
            
           b : Union[int, float, Number]
           
           Returns
           -------
           Number
               The number produced by the creator f(a, b), where f = Op().
            """
        """ Make `a` and `b` instances of `Number` if they aren't already. """
        if (not isinstance(a, Number)):
            a = Number(a)
        if (not isinstance(b, Number)):
            b = Number(b)
        
        """ Initialize Op, using `f` as its reference"""
        f = Op()

        self.update(event=Op, a=a, b=b)
        
        """ Get the output of the operation's forward pass, which is an int or float.
            Make it ans instance of `Number`, whose creator is f. Return this result."""
        return Number(f(a, b), creator=f)

    def __add__(self, other) -> Number:
        return self._op(Add, self, other)

    def __radd__(self, other) -> Number:
        return self._op(Add, other, self)

    def __mul__(self, other) -> Number:
        return self._op(Multiply, self, other)

    def __rmul__(self, other) -> Number:
        return self._op(Multiply, other, self)

    def __truediv__(self, other) -> Number:
        return self._op(Divide, self, other)

    def __rtruediv__(self, other) -> Number:
        return self._op(Divide, other, self)

    def __sub__(self, other) -> Number:
        return self._op(Subtract, self, other)

    def __rsub__(self, other) -> Number:
        return self._op(Subtract, other, self)

    def __pow__(self, other) -> Number:
        return self._op(Power, self, other)

    def __rpow__(self, other) -> Number:
        return self._op(Power, other, self)

    def __neg__(self) -> Number:
        return -1*self
    
    def __eq__(self, value):
        if isinstance(value, Number):
            value = value.data
        return self.data == value

    def backprop(self, grad=1):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        
        if self._creator is not None:
            self._creator.backprop(grad)
    
    def null_gradients(self):
        self.grad = None
        if self._creator is not None:
            self._creator.null_gradients()

    def display(self, depth: int=0, previous: Optional[Node] = None):
      node = Node(str(uuid.uuid4()), disp=str(round(self.data, 2)))
      self.update(event="graph", this=node, layer=depth, prev=previous)
      if self.creator is not None:
        self.creator.display(depth+1, previous=node)

    def update(self, *args, **kwargs):
      for observer in self.observers:
        observer.update(*args, **kwargs)