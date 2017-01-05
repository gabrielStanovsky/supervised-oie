import numpy as np

# Common symbols to be used across models
UNK_SYMBOL = "UNK"
UNK_INDEX = 0
UNK_VALUE = lambda dim: np.zeros(dim) # get an UNK of a specificed dimension
