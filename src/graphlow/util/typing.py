from enum import Enum

import numpy as np
import scipy.sparse as sp
import torch

ArrayDataType = np.ndarray | sp.sparray | torch.Tensor
NumpyScipyArray = np.ndarray | sp.sparray

KeyType = str | Enum
