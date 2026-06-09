import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
  from src.models.number import Number

Scalar = Union[int, float, Number]