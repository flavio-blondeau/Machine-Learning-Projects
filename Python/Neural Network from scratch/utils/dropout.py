import numpy as np
from .basics import Operation


class Dropout(Operation):

    def __init__(self, keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    
    def _output(self, inference: bool) -> np.ndarray:
        if inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob, size = self.input_.shape)

            return self.input_ * self.mask
        

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self.mask