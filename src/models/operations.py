from __future__ import annotations

import math
from abc import abstractmethod
from typing import Optional, TYPE_CHECKING
import uuid

from src.models.observable import Observable
from src.models.node import Node

if TYPE_CHECKING:
  from .number import Number
  from src.models.datatypes import Scalar

class Operation(Observable): # Observable inherits from ABC
    a: Number
    b: Number
    observers = []
    """ All `Operations` represent two-variable mathematical functions, e.g. +,-,*,/,** 
        __call__ accepts two `Number` objects, and returns a Python-numeric result (int, float)
    """
    @abstractmethod
    def partial_a(self) -> Scalar:
        """ Computes the partial derivative of this operation with respect to a: d(op)/da"""

    @abstractmethod
    def partial_b(self) -> Scalar:
        """ Computes the partial derivative of this operation with respect to b: d(op)/db"""

    @abstractmethod
    def __call__(self, a, b) -> Scalar:
        """ Computes the forward pass of this operation: op(a, b) -> output"""

    def backprop(self, grad) -> None:
        """ Calls .backprop for self.a and self.b, passing to it dF/da and
            dF/db, respectively, where F represents the terminal `Number`-instance node 
            in the computational graph, which originally invoked `F.backprop().
             
            (In short: F is the variable with respect to which all of the partial derivatives 
            are being computed, for this back propagation)
            
            Parameters
            ----------
            grad : Union[int, float]
                dF/d(op) - The partial derivative of F with respect to the output of this operation.
            """
        self.a.backprop(self.partial_a() * grad)  # backprop: dF/d(op)*d(op)/da -> dF/da
        self.b.backprop(self.partial_b() * grad)

    def null_gradients(self) -> None:
        for attr in self.__dict__:
            var = getattr(self, attr)
            if hasattr(var, 'null_gradients'):
                var.null_gradients()

    def display(self, depth: int=0, previous: Optional[Node] = None):
      node = Node(str(uuid.uuid4()), disp=self.__repr__())
      self.update(event="graph", this=node, layer=depth, prev=previous)
      self.a.display(depth+1, previous=node)
      self.b.display(depth+1, previous=node)

    def update(self, *args, **kwargs):
      for observer in self.observers:
        observer.update(*args, **kwargs)


class Add(Operation):
    def __repr__(self): return "+"

    def __call__(self, a, b):
        """ Adds two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return a.data + b.data
    
    def partial_a(self):
        """ Returns d(a + b)/da """
        return 1
    
    def partial_b(self):
        """ Returns d(a + b)/db """
        return 1


class Multiply(Operation):
    def __repr__(self): return "*"

    def __call__(self, a, b):
        """ Nultiplies two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return a.data * b.data
    
    def partial_a(self):
        """ Returns d(a * b)/da as int or float"""
        return self.b.data
    
    def partial_b(self):
        """ Returns d(a * b)/db as int or float"""
        return self.a.data


class Subtract(Operation):
    def __repr__(self): return "-"

    def __call__(self, a, b):
        """ Subtracts two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return self.a.data - self.b.data
    
    def partial_a(self):
        """ Returns d(a - b)/da as int or float"""
        return 1
    
    def partial_b(self):
        """ Returns d(a - b)/db as int or float"""
        return -1


class Divide(Operation):
    def __repr__(self): return "/"

    def __call__(self, a, b):
        """ Divides two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return self.a.data / self.b.data
    
    def partial_a(self):
        """ Returns d(a / b)/da as int or float"""
        return 1/self.b.data
    
    def partial_b(self):
        """ Returns d(a / b)/db as int or float"""
        return -self.a.data / self.b.data**2


class Power(Operation):
    def __repr__(self): return "**"

    def __call__(self, a, b):
        """ Exponentiates two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return self.a.data ** self.b.data
    
    def partial_a(self):
        """ Returns d(a ** b)/da as int or float"""
        return self.b.data * self.a.data**(self.b.data-1)
    
    def partial_b(self):
        """ Returns d(a ** b)/db as int or float
            Reference: http://tutorial.math.lamar.edu/Classes/CalcI/DiffExpLogFcns.aspx
        """
        return self.a.data ** self.b.data * math.log(self.a.data)