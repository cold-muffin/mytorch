import numpy as np
from typing import Sequence, Callable

from src.models.observable import Observable
from src.models.number import Number
from src.models.operations import *

class Train(Observable):
    @staticmethod
    def l2_loss(truth: Sequence[int], model: Sequence[Number]) -> Number:
        ''' Simple L2 loss for a single sample.
        
            Parameters
            ----------
            truth : Sequence[float]
                A single data point.
                
            model : Sequence[float]
                The parameters of a model.
                
            Returns
            -------
            Number : the L2 loss (squared error) of `model_params` evaluated on `truth`
        '''
        l = truth[0] - model[0] - sum(truth[i]*model[i] for i in range(1, len(model)))
        return l**2 if l.data > 0 else (-1*l)**2

    @staticmethod
    def create_model(num_params: int) -> Sequence[Number]:
        ''' Create a model by randomly initializing `num_params` parameters. 
            
            Our model, then, is represented by tuple containing all of its
            parameters. Each parameter is an instance of the `Number` class
            so that its gradient can be backpropagated to it from the loss,
            and we can thus updated the parameter, via gradient descent, so
            that the model will produce better predictions.'''
        return tuple(Number(np.random.rand()) for _ in range(num_params))

    def train_epoch(self, model: Sequence[Number], data_set: Sequence[Sequence[int]], loss_fn: Callable[[Sequence[int], Sequence[Number]], Number], lr=0.001):
        ''' Train a model for a single pass (epoch) through the provided data.
        
            Parameters
            ----------
            model : Sequence[Number]
                The parameters of a model
                
            data_set : Sequence[Sequence[float]]
                The datapoints in a dataset
                
            Returns
            -------
            float : the mean loss for the epoch
        '''
        # compute the mean error over the dataset
        mean_loss = sum(loss_fn(sample, model) for sample in data_set) / len(data_set)
        
        # compute gradients for our parameters
        assert isinstance(mean_loss, Number)
        mean_loss.null_gradients()
        mean_loss.backprop()
        mean_loss.display()
        
        # update the model parameters using gradient descent
        for param in model:
            # recall: param.grad is d(L)/d(param)
            # thus this computes:
            # param_new = param_old - step-size * d(L)/d(param)
            assert param.grad is not None
            param.data -= lr*param.grad
            
        # return the loss for visualization
        return mean_loss.data

    def update(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)